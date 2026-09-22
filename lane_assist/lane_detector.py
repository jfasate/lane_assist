#!/usr/bin/env python3
"""Camera lane keeping: painted lines -> /planning/ref_path.

Same role as window_republisher (publish an OPEN local horizon the LPV-MPC can
track in ref_source='topic' mode), but the geometry comes from the camera
instead of a CSV. Run with ref_generator='external' so the two never both own
the topic.

Pipeline per frame:
  RGB -> white mask -> per-row paint runs -> ray/ground intersection (X,Y in
  base_link) -> near-to-far lane bracketing -> anchored quadratic fit -> EMA
  -> (N,6) [s, x, y, psi, kappa, vx] in WORLD frame.

The road model this was written against (superspeedway, measured off its
model.sdf): solid edge lines at +/-1.44 m, dashed lane lines at +/-0.49 m, so a
1.00 m middle lane between two 0.95 m outer lanes, and a 0.20 m car. Nothing
here hardcodes those offsets -- the detector brackets whatever two lines
straddle the car -- but lane_width is the fallback when only one side is
visible, which is common because the dashes have 0.32 m gaps.

Measured accuracy on a live frame at the entry to the tightest corner
(R=4.94 m): lane centre within 0.075 m mean / 0.17 m max of the true centre
derived from the SDF paint positions.

Two geometry facts this camera forces, both handled here rather than tuned
around:
  * At 0.2025 m mount height the ground from 3 m to the horizon occupies ~42 of
    480 rows, so lookahead past ~3 m carries no usable geometry.
  * Both lane lines only enter the 69.4 deg FOV beyond lane_min_range() (~0.7 m
    here), so x_min is raised to that automatically from camera_info.

Fail-safe: after blind_ticks_stop consecutive undetected frames it publishes the
last good geometry at vx=0, which lpv_mpc_node treats as an explicit HOLD and
stops the car. Silence would instead leave the MPC tracking a stale horizon.

Self-check (no ROS needed):  python3 lane_detector.py --selfcheck
"""

import math
import os
import sys
import time

import numpy as np

# ── Pure geometry (ROS-free so the self-check and any offline tool can use it) ──


def ground_from_pixel(u, v, K, cam_h, cam_pitch, cam_x):
    """Pixel(s) -> (X, Y) where the ray hits z=0, in base_link (x fwd, y left).

    Rays at or above the horizon return NaN. u, v may be arrays.
    """
    fx, fy, cx, cy = K
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    # optical frame (X right, Y down, Z fwd) -> camera_link (x fwd, y left, z up)
    dx = np.ones_like(u)
    dy = -(u - cx) / fx
    dz = -(v - cy) / fy
    # camera_link -> base_link: rotate by +pitch about +y (nose-down mount)
    ct, st = math.cos(cam_pitch), math.sin(cam_pitch)
    bx = ct * dx + st * dz
    by = dy
    bz = -st * dx + ct * dz
    with np.errstate(divide='ignore', invalid='ignore'):
        s = np.where(bz < -1e-6, cam_h / np.maximum(-bz, 1e-9), np.nan)
    return cam_x + s * bx, s * by


def pixel_from_ground(X, Y, K, cam_h, cam_pitch, cam_x):
    """Inverse of ground_from_pixel — used by the debug overlay and self-check."""
    fx, fy, cx, cy = K
    dx = np.asarray(X, dtype=float) - cam_x
    dy = np.asarray(Y, dtype=float)
    dz = -cam_h
    ct, st = math.cos(cam_pitch), math.sin(cam_pitch)
    cxx = ct * dx - st * dz
    cyy = dy
    czz = st * dx + ct * dz
    with np.errstate(divide='ignore', invalid='ignore'):
        u = cx - fx * cyy / cxx
        v = cy - fy * czz / cxx
    return u, v


def paint_runs(mask_row):
    """(first, last) pixel of each white run in one mask row.

    Endpoints, not the centre: the caller projects both to the ground and drops
    runs spanning more than a stripe's width, which is the only way to tell a
    0.08 m marking from glare or a sunlit wall in a resolution- and
    distance-independent way.
    """
    idx = np.flatnonzero(mask_row)
    if idx.size == 0:
        return []
    splits = np.flatnonzero(np.diff(idx) > 1)
    return [(float(run[0]), float(run[-1]))
            for run in np.split(idx, splits + 1) if run.size]


def poly(coef, x, x0):
    """Lane centre model Y(X) = a + b*u + c*u^2, u = X - x0.

    The basis is ANCHORED at x0 (mid-lookahead) rather than at the car. The
    camera cannot see its own lane lines nearer than lane_min_range(), so a
    basis at X=0 makes `a` a pure extrapolation from data that starts ~0.9 m
    out: the fit then trades a/b/c off against each other and drives c into its
    clamp. Anchored at x0, `a` is the lane offset where the data actually is.
    """
    a, b, c = coef
    u = np.asarray(x, dtype=float) - x0
    return a + b * u + c * u * u


def lane_min_range(K, lane_width):
    """Nearest X at which both of the ego lane's boundaries are inside the
    horizontal FOV. Below this the camera sees only bare road between the lines,
    so bins there are empty no matter how good the threshold is.

    half_visible(X) = X * tan(hfov/2) = X * (W/2) / fx, and W/2 ~ cx.
    """
    fx, _, cx, _ = K
    return 0.5 * lane_width * fx / max(cx, 1.0)


def fit_lane_center(X, Y, n_bins, x_min, x_max, lane_width, coef0=None,
                    min_bins=3, max_abs_b=0.70, max_abs_c=0.15,
                    w_lo=0.5, w_hi=1.6, x0=None):
    """Paint points -> quadratic lane centre, propagated NEAR TO FAR.

    Bracketing every bin against y=0 only works on a straight. This track's
    tightest turn is R=4.94 m, where the lane centre moves +0.91 m over the 3 m
    lookahead — nearly a full lane width — so in the far bins BOTH of the ego
    lane's boundaries sit on the same side of y=0. Classifying those against
    y=0 labels the right-hand boundary as the left one and the fit collapses
    (measured: psi0 = +41 deg on a real frame).

    So walk the bins outward instead and classify each one against the centre
    found in the previous bin. Bins are ~0.32 m apart and curvature shifts the
    centre by <0.1 m over that step, so the previous bin is always a good local
    predictor no matter how tight the corner. Bin 0 is seeded from the previous
    frame's fit (coef0) when there is one, else 0.

    A point further than 0.75*lane_width from the running centre belongs to
    another lane's markings and is ignored: the ego boundaries sit at +/-0.475 m
    and the neighbours' at +/-1.425 m, so the cut is unambiguous.

    Returns (coef, x_bins, y_centres) or None if the fit is not trustworthy.
    """
    if x0 is None:
        x0 = 0.5 * (x_min + x_max)
    edges = np.linspace(x_min, x_max, n_bins + 1)
    band = 0.75 * lane_width
    half = 0.5 * lane_width
    xs, ys = [], []
    for i in range(n_bins):
        xb = 0.5 * (edges[i] + edges[i + 1])
        # Predict this bin's centre from the last accepted bin, else the
        # previous frame's curve, else straight ahead.
        if ys:
            yc = ys[-1]
        elif coef0 is not None:
            yc = float(poly(np.asarray(coef0, dtype=float), xb, x0))
        else:
            yc = 0.0
        m = (X >= edges[i]) & (X < edges[i + 1])
        if not m.any():
            continue
        rb = Y[m] - yc
        rb = rb[np.abs(rb) <= band]
        if rb.size == 0:
            continue
        right = rb[rb < 0.0]
        left = rb[rb > 0.0]
        if right.size and left.size:
            r, l = right.max(), left.min()
            if not (w_lo <= (l - r) <= w_hi):
                continue
            dc = 0.5 * (l + r)
        elif right.size:
            dc = right.max() + half
        else:
            dc = left.min() - half
        xs.append(xb)
        ys.append(yc + dc)

    if len(xs) < min_bins:
        return None
    bx, by = np.asarray(xs), np.asarray(ys)
    deg = 2 if bx.size >= 4 else 1
    f = np.polyfit(bx - x0, by, deg)[::-1]
    coef = np.array([f[0], f[1], f[2] if deg == 2 else 0.0])
    # Curvature is bounded by the track, not by the fit: 2c ~ kappa, so cap |c|
    # instead of letting one sparse far bin bend the line into nonsense.
    # ponytail: 2c over-reads kappa by ~0.05-0.10 1/m. The `band` gate clips the
    # far side of one line before the near side of the other, so single-sided
    # bins carry a small outward bias that the fit turns into curvature. The
    # offset (a) is unaffected (~0.015 m) and lpv_mpc re-derives psi/kappa from
    # the points, so this only costs a little speed via the a_lat cap. Fix by
    # fitting each lane line separately and averaging, if it ever matters.
    coef[2] = float(np.clip(coef[2], -max_abs_c, max_abs_c))

    # Reject rather than clamp the offset/heading: a lane centre more than a
    # lane away, or a heading error past max_abs_b, means we latched onto the
    # wrong markings. The caller then holds the last good fit (and eventually
    # commands HOLD), which is safer than steering to a bad line.
    if abs(coef[0]) > lane_width or abs(coef[1]) > max_abs_b:
        return None
    return coef, bx, by


def horizon_row(K, cam_pitch):
    """Image row where the ground plane vanishes. Everything above it is sky or
    wall and carries no lane geometry, so the mask starts below it."""
    _, fy, _, cy = K
    return cy - fy * math.tan(cam_pitch)


def horizon_to_path(coef, x_min, x_max, n_ref, target_speed, a_lat_max, x0=None):
    """Quadratic lane centre Y(X) -> ego (N,6) [s, x, y, psi, kappa, vx]."""
    if x0 is None:
        x0 = 0.5 * (x_min + x_max)
    a, b, c = coef
    X = np.linspace(x_min, x_max, n_ref)
    u = X - x0
    Y = a + b * u + c * u * u
    dY = b + 2.0 * c * u
    psi = np.arctan(dY)
    kappa = 2.0 * c / np.power(1.0 + dY * dY, 1.5)
    v_curv = np.sqrt(a_lat_max / np.maximum(np.abs(kappa), 1e-6))
    vx = np.minimum(target_speed, v_curv)
    seg = np.hypot(np.diff(X), np.diff(Y))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    return np.column_stack([s, X, Y, psi, kappa, vx])


def ego_to_world(path, pose):
    """(N,6) ego horizon -> world frame, given pose (x, y, yaw)."""
    px, py, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    out = path.copy()
    out[:, 1] = px + path[:, 1] * c - path[:, 2] * s
    out[:, 2] = py + path[:, 1] * s + path[:, 2] * c
    out[:, 3] = path[:, 3] + yaw
    return out


# ── ROS node ──────────────────────────────────────────────────────────────────

def _main_ros():
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from sensor_msgs.msg import Image, CameraInfo
    from nav_msgs.msg import Odometry, Path
    from geometry_msgs.msg import PoseStamped
    from std_msgs.msg import Float64MultiArray, MultiArrayDimension
    from visualization_msgs.msg import Marker

    def _dim(label, size, stride):
        d = MultiArrayDimension()
        d.label, d.size, d.stride = label, int(size), int(stride)
        return d


    class LaneDetector(Node):
        def __init__(self):
            super().__init__('lane_detector')
            self.declare_parameter('camera_topic', '/camera/color/image_raw')
            self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
            self.declare_parameter('odom_topic', '/ego_racecar/odom')
            self.declare_parameter('ref_path_topic', '/planning/ref_path')
            self.declare_parameter('debug_image_topic', '/lane_keeping/debug_image')
            self.declare_parameter('publish_debug_image', True)
            self.declare_parameter('cam_height', 0.2025)
            self.declare_parameter('cam_pitch', 0.05)
            self.declare_parameter('cam_x', 0.275)
            self.declare_parameter('white_k', 2.0)
            self.declare_parameter('white_min', 140.0)
            self.declare_parameter('max_paint_width_m', 0.20)
            self.declare_parameter('n_rows', 48)
            self.declare_parameter('n_bins', 8)
            self.declare_parameter('x_min', 0.40)
            self.declare_parameter('x_max', 3.00)
            self.declare_parameter('y_max', 2.00)
            self.declare_parameter('lane_width', 1.00)
            self.declare_parameter('min_bins', 3)
            self.declare_parameter('max_abs_b', 0.70)
            self.declare_parameter('max_abs_c', 0.15)
            self.declare_parameter('n_ref', 20)
            self.declare_parameter('target_speed', 1.5)
            self.declare_parameter('a_lat_max', 3.0)
            self.declare_parameter('ema_alpha', 0.35)
            self.declare_parameter('publish_rate_hz', 20.0)
            self.declare_parameter('blind_ticks_stop', 10)
            self.declare_parameter('enable_csv_log', True)
            self.declare_parameter('log_dir', '')
            self.declare_parameter('diag_topic', '/lane_assist/debug')
            self.declare_parameter('publish_diag', True)
            self.declare_parameter('ref_viz_topic', '/planning/ref_path_viz')
            self.declare_parameter('mpc_pred_topic', '/lpv_mpc_gazebo/pred_path')
            # The horizon is PUBLISHED from here, while the fit still uses
            # x_min. The camera cannot see nearer than x_min (~0.72 m), but the
            # car is physically on the lane between 0 and x_min, so evaluating
            # the same quadratic there is interpolation toward the car, not
            # invention. Without it the reference never covers the ground just
            # ahead of the car and lpv_mpc locks onto the far end of the stub
            # (measured: aiming 2.69 m out on a 2.3 m horizon = 118%), which
            # makes it start turning ~2 m before the corner.
            self.declare_parameter('ref_start_m', 0.0)

            g = lambda k: self.get_parameter(k).value            # noqa: E731
            self.cam_h = float(g('cam_height'))
            self.cam_pitch = float(g('cam_pitch'))
            self.cam_x = float(g('cam_x'))
            self.white_k = float(g('white_k'))
            self.white_min = float(g('white_min'))
            self.max_paint_w = float(g('max_paint_width_m'))
            self.n_rows = int(g('n_rows'))
            self.n_bins = int(g('n_bins'))
            self.x_min = float(g('x_min'))
            self.x_max = float(g('x_max'))
            self.x0 = 0.5 * (self.x_min + self.x_max)
            self.y_max = float(g('y_max'))
            self.lane_width = float(g('lane_width'))
            self.min_bins = int(g('min_bins'))
            self.max_abs_b = float(g('max_abs_b'))
            self.max_abs_c = float(g('max_abs_c'))
            self.ref_start = float(g('ref_start_m'))
            self.n_ref = int(g('n_ref'))
            self.target_speed = float(g('target_speed'))
            self.a_lat_max = float(g('a_lat_max'))
            self.ema_alpha = float(g('ema_alpha'))
            self.blind_stop = int(g('blind_ticks_stop'))
            self.debug_on = bool(g('publish_debug_image'))

            self.K = None
            self.img = None
            self.pose = None
            self.twist = (0.0, 0.0, 0.0)
            self.coef = None
            self.blind = 0
            self.pred = None
            self.pred_t = float('nan')

            best = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
            latched = QoSProfile(
                depth=1, reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL)

            self._bridge = None
            try:
                from cv_bridge import CvBridge
                self._bridge = CvBridge()
            except ImportError:
                self.get_logger().warn('cv_bridge unavailable -> manual decode')

            self.create_subscription(Image, str(g('camera_topic')),
                                     self._img_cb, best)
            self.create_subscription(CameraInfo, str(g('camera_info_topic')),
                                     self._info_cb, best)
            self.create_subscription(Odometry, str(g('odom_topic')),
                                     self._odom_cb, best)
            self.create_subscription(Marker, str(g('mpc_pred_topic')),
                                     self._pred_cb, best)
            self._pub = self.create_publisher(
                Float64MultiArray, str(g('ref_path_topic')), latched)
            self._dbg = self.create_publisher(
                Image, str(g('debug_image_topic')), best) if self.debug_on else None
            # RELIABLE, not best-effort: this is the analysis record and a
            # dropped tick is a hole in the data. Depth 50 rides out a recorder
            # that briefly falls behind.
            self._viz = self.create_publisher(
                Path, str(g('ref_viz_topic')), best)
            self._diag = (self.create_publisher(
                Float64MultiArray, str(g('diag_topic')),
                QoSProfile(depth=50, reliability=ReliabilityPolicy.RELIABLE))
                if bool(g('publish_diag')) else None)

            rate = float(g('publish_rate_hz'))
            self.create_timer(1.0 / max(rate, 1e-3), self._tick)
            self._init_csv_log()
            self.get_logger().info(
                f'lane_detector up @ {rate:.0f} Hz  x=[{self.x_min:.2f},'
                f'{self.x_max:.2f}] m  lane_width={self.lane_width:.2f} m  '
                f'mount h={self.cam_h:.4f} m pitch={self.cam_pitch:.4f} rad')

        # ── callbacks ──
        def _info_cb(self, m):
            if self.K is None:
                self.K = (m.k[0], m.k[4], m.k[2], m.k[5])
                self.get_logger().info(
                    f'intrinsics fx={self.K[0]:.1f} fy={self.K[1]:.1f} '
                    f'cx={self.K[2]:.1f} cy={self.K[3]:.1f} ({m.width}x{m.height})')
                # Below this range both lane lines fall outside the horizontal
                # FOV, so those bins are always empty and only drag the fit.
                geo = lane_min_range(self.K, self.lane_width)
                if self.x_min < geo:
                    self.get_logger().info(
                        f'x_min {self.x_min:.2f} -> {geo:.2f} m: nearer than '
                        f'that this FOV cannot see both lane lines')
                    self.x_min = geo
                self.x0 = 0.5 * (self.x_min + self.x_max)

        def _img_cb(self, m):
            if self._bridge is not None:
                self.img = self._bridge.imgmsg_to_cv2(m, desired_encoding='rgb8')
            else:
                a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1)
                if m.encoding.lower().startswith('bgr'):
                    a = a[:, :, ::-1]
                self.img = np.ascontiguousarray(a[:, :, :3])

        def _odom_cb(self, m):
            q = m.pose.pose.orientation
            yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                             1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            self.pose = (m.pose.pose.position.x, m.pose.pose.position.y, yaw)
            t = m.twist.twist
            self.twist = (t.linear.x, t.linear.y, t.angular.z)

        def _pred_cb(self, m):
            self.pred = np.array([(p.x, p.y) for p in m.points], dtype=float)
            self.pred_t = self.get_clock().now().nanoseconds * 1e-9

        # ── detection ──
        def _detect(self):
            """Latest frame -> (X, Y) paint points in base_link, plus the mask."""
            img = self.img
            h, w = img.shape[:2]
            top = int(max(0, math.ceil(horizon_row(self.K, self.cam_pitch)) + 2))
            if top >= h - 2:
                return None, None, None
            crop = img[top:, :, :]
            gray = crop.max(axis=2).astype(np.float32)
            thr = max(self.white_min, gray.mean() + self.white_k * gray.std())
            mask = gray >= thr

            rows = np.unique(np.linspace(0, mask.shape[0] - 1,
                                         self.n_rows).astype(int))
            u0, u1, vv = [], [], []
            for r in rows:
                for a, b in paint_runs(mask[r]):
                    u0.append(a)
                    u1.append(b)
                    vv.append(r + top)
            if not u0:
                return None, None, mask
            u0 = np.asarray(u0)
            u1 = np.asarray(u1)
            vv = np.asarray(vv)
            # Reject a run by the ground width it spans, measured by projecting
            # BOTH its endpoints. Estimating metres-per-pixel from the row's
            # image edges instead allows ~164 px at 0.7 m, which is no filter at
            # all; projecting the endpoints is both simpler and actually bounded.
            X0, Y0 = ground_from_pixel(u0, vv, self.K, self.cam_h,
                                       self.cam_pitch, self.cam_x)
            X1, Y1 = ground_from_pixel(u1, vv, self.K, self.cam_h,
                                       self.cam_pitch, self.cam_x)
            X = 0.5 * (X0 + X1)
            Y = 0.5 * (Y0 + Y1)
            keep = (np.isfinite(X) & np.isfinite(Y)
                    & (np.abs(Y1 - Y0) <= self.max_paint_w)
                    & (X >= self.x_min) & (X <= self.x_max)
                    & (np.abs(Y) <= self.y_max))
            return X[keep], Y[keep], mask

        def _tick(self):
            if self.K is None or self.img is None or self.pose is None:
                return
            X, Y, mask = self._detect()
            coef = None
            bins = None
            if X is not None and X.size >= self.min_bins:
                # Seed from the previous fit so the left/right classification
                # starts on the right curve instead of re-converging each frame.
                got = fit_lane_center(
                    X, Y, self.n_bins, self.x_min, self.x_max, self.lane_width,
                    coef0=self.coef, min_bins=self.min_bins,
                    max_abs_b=self.max_abs_b, max_abs_c=self.max_abs_c,
                    x0=self.x0)
                if got is not None:
                    coef = got[0]
                    bins = (got[1], got[2])

            if coef is None:
                self.blind += 1
                if self.coef is None:
                    self._record(X, Y, None, 0, None)
                    return                      # nothing good ever seen yet
                if self.blind >= self.blind_stop:
                    # Explicit HOLD: lpv_mpc_node stops the car on a zero-speed
                    # horizon, which is safer than letting it track a stale one.
                    path = horizon_to_path(self.coef, self.ref_start, self.x_max,
                                           self.n_ref, 0.0, self.a_lat_max,
                                           x0=self.x0)
                    self._publish(path)
                    if self.blind % 20 == self.blind_stop % 20:
                        self.get_logger().warn(
                            f'no lane for {self.blind} ticks -> commanding HOLD')
                    self._record(X, Y, None, 0, path)
                    return
            else:
                self.blind = 0
                a = self.ema_alpha
                self.coef = coef if self.coef is None else a * coef + (1 - a) * self.coef

            path = horizon_to_path(self.coef, self.ref_start, self.x_max,
                                   self.n_ref, self.target_speed,
                                   self.a_lat_max, x0=self.x0)
            self._publish(path)
            self._record(X, Y, bins, int(coef is not None), path)
            if self._dbg is not None and mask is not None:
                self._publish_debug(mask)

        def _publish(self, path):
            # EGO-CENTRIC: the controller gets raw base_link coordinates. Going
            # via the map frame meant the measurement was rotated by a pose
            # sampled at a different instant than the camera frame, which shows
            # up as a phantom lateral error that no controller can reject.
            msg = Float64MultiArray()
            msg.data = path.ravel().tolist()
            self._pub.publish(msg)
            # Same geometry as a nav_msgs/Path purely so RViz can draw it:
            # Float64MultiArray has no RViz display, which is why the reference
            # was invisible after the switch from the CSV republisher. This one
            # still needs the map frame or the line renders at the origin.
            if self._viz is not None:
                world_path = ego_to_world(path, self.pose)
                pa = Path()
                pa.header.frame_id = 'map'
                pa.header.stamp = self.get_clock().now().to_msg()
                for row in world_path:
                    ps = PoseStamped()
                    ps.header = pa.header
                    ps.pose.position.x = float(row[1])
                    ps.pose.position.y = float(row[2])
                    ps.pose.orientation.z = math.sin(0.5 * float(row[3]))
                    ps.pose.orientation.w = math.cos(0.5 * float(row[3]))
                    pa.poses.append(ps)
                self._viz.publish(pa)

        # Everything the detector knows that never reaches another node. A bag
        # records only what is on the wire, so without this it would be missing
        # exactly the signal that separates a bad camera from a bad fit.
        # MultiArrayDimension labels each section, so a reader splits it by name
        # and no schema is kept in sync by hand -- which is how the CSV column
        # list silently drifted.
        DEBUG_SCALARS = [
            'sim_t', 'X', 'Y', 'yaw', 'vx', 'vy', 'omega',
            'fit_ok', 'blind', 'x_min', 'x0', 'lane_offset_m',
        ]

        def _record(self, X, Y, bins, fit_ok, path):
            """Log + publish diagnostics. Never let recording kill the node.

            A bug in here previously took the detector down on its first real
            tick, which left the MPC tracking a stale horizon and cost a whole
            run. Diagnostics are not worth the control loop: warn once and keep
            driving.
            """
            try:
                self._log_row(X, Y, bins, fit_ok, path)
                self._publish_diag(X, Y, bins, fit_ok, path)
            except Exception as e:                       # noqa: BLE001
                if not getattr(self, '_record_broken', False):
                    self._record_broken = True
                    self.get_logger().error(
                        f'recording disabled after {type(e).__name__}: {e}')

        def _publish_diag(self, X, Y, bins, fit_ok, path):
            if self._diag is None:
                return
            c = self.coef if self.coef is not None else np.full(3, np.nan)
            off = (float(poly(self.coef, 0.0, self.x0))
                   if self.coef is not None else float('nan'))
            px, py, yaw = self.pose
            scal = [self.get_clock().now().nanoseconds * 1e-9, px, py, yaw,
                    self.twist[0], self.twist[1], self.twist[2],
                    float(fit_ok), float(self.blind), self.x_min, self.x0, off]
            bx, by = bins if bins is not None else (np.empty(0), np.empty(0))
            pxs, pys = ((X, Y) if X is not None and X.size
                        else (np.empty(0), np.empty(0)))
            hz = (ego_to_world(path, self.pose).ravel()
                  if path is not None else np.empty(0))
            msg = Float64MultiArray()
            msg.layout.dim = [
                _dim('scalars', len(scal), 1),
                _dim('coef', 3, 1),
                _dim('bins_xy', int(bx.size), 2),
                _dim('paint_xy', int(pxs.size), 2),
                _dim('horizon', 0 if path is None else int(path.shape[0]), 6),
            ]
            msg.data = (list(scal) + [float(v) for v in c]
                        + np.column_stack([bx, by]).ravel().tolist()
                        + np.column_stack([pxs, pys]).ravel().tolist()
                        + hz.tolist())
            self._diag.publish(msg)

        LOG_COLUMNS = [
            'wall_t', 'sim_t',
            # measured pose, so the log alone is enough to score a run offline
            'X', 'Y', 'yaw', 'vx', 'vy', 'omega',
            # what the camera found this tick
            'n_paint', 'n_bins', 'fit_ok', 'blind', 'x_min', 'x0',
            # the accepted lane model (anchored at x0) and what it means at the car
            'coef_a', 'coef_b', 'coef_c', 'lane_offset_m', 'psi0_deg', 'kappa0',
            # first point of the horizon actually published
            'ref_X', 'ref_Y', 'ref_psi', 'ref_vx', 'ref_arc_m',
            # full series, space-separated in one cell (lpv_mpc's log convention).
            # Logging only ref_X/ref_Y hides a horizon whose SHAPE is wrong,
            # which is exactly the failure mode to look for in a turn.
            'ref_hx', 'ref_hy',          # whole published horizon, WORLD frame
            # what lpv_mpc predicted it would drive, same WORLD frame, from
            # /lpv_mpc_gazebo/pred_path. pred_age_s is how stale it was at this
            # tick: the MPC runs at 50 Hz and this node at 20 Hz, so a healthy
            # age is < 0.05 s. A growing age means the MPC stopped publishing.
            'pred_hx', 'pred_hy', 'pred_age_s',
            'bin_x', 'bin_y',            # per-bin lane centres, ego frame
            'paint_x', 'paint_y',        # every detected paint point, ego frame
        ]

        @staticmethod
        def _series(a):
            return ' '.join(f'{v:.4f}' for v in np.asarray(a).ravel())

        def _init_csv_log(self):
            self._csv_file = None
            self._csv_writer = None
            if not bool(self.get_parameter('enable_csv_log').value):
                return
            import csv
            import datetime
            d = str(self.get_parameter('log_dir').value) or os.getcwd()
            try:
                os.makedirs(d, exist_ok=True)
                stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                path = os.path.join(d, f'lane_{stamp}.csv')
                self._csv_file = open(path, 'w', newline='')
                self._csv_writer = csv.writer(self._csv_file)
                self._csv_writer.writerow(self.LOG_COLUMNS)
                self.get_logger().info(f'lane log -> {path}')
            except OSError as e:
                self.get_logger().warn(f'lane log disabled: {e}')
                self._csv_file = None
                self._csv_writer = None

        def _log_row(self, X, Y, bins, fit_ok, path):
            if self._csv_writer is None:
                return
            n_paint = 0 if X is None else int(X.size)
            n_bins = 0 if bins is None else int(bins[0].size)
            c = self.coef if self.coef is not None else (float('nan'),) * 3
            off = (float(poly(self.coef, 0.0, self.x0))
                   if self.coef is not None else float('nan'))
            # px/py, NOT X/Y: X and Y are this tick's paint points and are still
            # needed at the bottom of this method. Unpacking the pose into them
            # replaced the arrays with floats and killed the node on the first
            # real tick (float has no .size).
            px, py, yaw = self.pose
            row = [time.time(), self.get_clock().now().nanoseconds * 1e-9,
                   px, py, yaw, self.twist[0], self.twist[1], self.twist[2],
                   n_paint, n_bins, fit_ok, self.blind, self.x_min, self.x0,
                   c[0], c[1], c[2], off]
            if path is not None:
                arc = float(np.hypot(*np.diff(path[:, 1:3], axis=0).T).sum())
                w = ego_to_world(path, self.pose)
                row += [math.degrees(path[0, 3]), path[0, 4],
                        w[0, 1], w[0, 2], w[0, 3], path[0, 5], arc]
            else:
                row += [float('nan')] * 7
            row = [f'{v:.6f}' if isinstance(v, float) else v for v in row]
            if path is not None:
                w = ego_to_world(path, self.pose)
                row += [self._series(w[:, 1]), self._series(w[:, 2])]
            else:
                row += ['', '']
            if self.pred is not None and self.pred.size:
                age = self.get_clock().now().nanoseconds * 1e-9 - self.pred_t
                row += [self._series(self.pred[:, 0]),
                        self._series(self.pred[:, 1]), f'{age:.4f}']
            else:
                row += ['', '', '']
            row += ([self._series(bins[0]), self._series(bins[1])]
                    if bins is not None else ['', ''])
            row += ([self._series(X), self._series(Y)]
                    if X is not None and X.size else ['', ''])
            self._csv_writer.writerow(row)
            self._csv_file.flush()

        def _close_log(self):
            if self._csv_file is not None:
                self._csv_file.close()
                self._csv_file = None
                self._csv_writer = None

        def _publish_debug(self, mask):
            h, w = self.img.shape[:2]
            out = (self.img.astype(np.float32) * 0.5).astype(np.uint8)
            out[h - mask.shape[0]:, :, 0] = np.where(mask, 255,
                                                     out[h - mask.shape[0]:, :, 0])
            if self.coef is not None:
                Xs = np.linspace(self.x_min, self.x_max, 40)
                Ys = poly(self.coef, Xs, self.x0)
                u, v = pixel_from_ground(Xs, Ys, self.K, self.cam_h,
                                         self.cam_pitch, self.cam_x)
                ok = np.isfinite(u) & np.isfinite(v)
                ui = np.clip(u[ok].astype(int), 0, w - 1)
                vi = np.clip(v[ok].astype(int), 0, h - 1)
                out[vi, ui] = (0, 255, 0)
            m = Image()
            m.height, m.width = h, w
            m.encoding = 'rgb8'
            m.step = 3 * w
            m.data = out.tobytes()
            self._dbg.publish(m)

    rclpy.init()
    node = LaneDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._close_log()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


# ── Self-check ────────────────────────────────────────────────────────────────

def selfcheck():
    """Round-trip the projection and recover a known lane centre from a synthetic
    superspeedway frame. Fails loudly if the geometry or the bracketing breaks."""
    W, H = 640, 480
    hfov = 1.2112
    fx = (W / 2) / math.tan(hfov / 2)
    K = (fx, fx, W / 2, H / 2)
    h, pitch, cam_x = 0.2025, 0.05, 0.275

    # 1. pixel <-> ground round trip
    Xg = np.array([0.5, 1.0, 2.0, 3.0, 5.0])
    Yg = np.array([-0.5, 0.0, 0.49, -1.44, 1.45])
    u, v = pixel_from_ground(Xg, Yg, K, h, pitch, cam_x)
    Xb, Yb = ground_from_pixel(u, v, K, h, pitch, cam_x)
    assert np.allclose(Xg, Xb, atol=1e-6), (Xg, Xb)
    assert np.allclose(Yg, Yb, atol=1e-6), (Yg, Yb)

    # 2. horizon row is above every finite ground projection
    XMIN = lane_min_range(K, 1.00)
    assert 0.65 < XMIN < 0.80, f'unexpected geometric x_min {XMIN:.3f}'
    hr = horizon_row(K, pitch)
    assert np.all(v > hr), (v, hr)
    assert not np.isfinite(ground_from_pixel(W / 2, hr - 1.0, K, h, pitch, cam_x)[0])

    # 3/4. synthetic superspeedway frames -> recover offset AND curvature.
    # kappa=0.2025 is the track's tightest turn, where the lane centre moves
    # +0.91 m over the 3 m lookahead. That case is why the fit brackets against
    # the curve instead of y=0, so it has to be in the check.
    def synth(offset, kappa):
        """Paint the four lines of a lane curving at kappa, car `offset` m right
        of the lane centre. Returns in-window (X, Y) paint points."""
        mask = np.zeros((H, W), bool)
        for line_y in (-1.44, -0.49, 0.51, 1.45):
            for Xs in np.linspace(0.3, 4.0, 600):
                yc = 0.5 * kappa * Xs * Xs          # lane centre on the curve
                for dy in (-0.04, 0.0, 0.04):       # 0.08 m stripe width
                    uu, vv = pixel_from_ground(Xs, yc + line_y - offset + dy,
                                               K, h, pitch, cam_x)
                    if (np.isfinite(uu) and np.isfinite(vv)
                            and 0 <= int(uu) < W and 0 <= int(vv) < H):
                        mask[int(vv), int(uu)] = True
        rows = np.unique(np.linspace(int(hr) + 2, H - 1, 48).astype(int))
        us, vs = [], []
        for r in rows:
            for a, b in paint_runs(mask[r]):
                us.append(0.5 * (a + b))
                vs.append(r)
        Xp, Yp = ground_from_pixel(np.asarray(us), np.asarray(vs), K, h,
                                   pitch, cam_x)
        k = (np.isfinite(Xp) & np.isfinite(Yp) & (Xp >= XMIN) & (Xp <= 3.0)
             & (np.abs(Yp) <= 2.0))
        return Xp[k], Yp[k]

    for offset, kappa in [(-0.20, 0.0), (0.0, 0.2025), (0.25, -0.2025),
                          (-0.30, 0.0827)]:
        Xp, Yp = synth(offset, kappa)
        got = fit_lane_center(Xp, Yp, 8, XMIN, 3.0, 1.00)
        assert got is not None, f'no fit at offset={offset} kappa={kappa}'
        coef, bx, by = got
        # coef is anchored at x0, so `a` is the lane centre at mid-lookahead:
        # the curve has already carried it sideways by kappa*x0^2/2.
        x0 = 0.5 * (XMIN + 3.0)
        want_a = 0.5 * kappa * x0 * x0 - offset
        assert abs(coef[0] - want_a) < 0.12, \
            f'offset={offset} kappa={kappa}: a={coef[0]:+.3f} want {want_a:+.3f}'
        # 2c should recover kappa (clamped at max_abs_c=0.15 -> kappa 0.30).
        assert abs(2.0 * coef[2] - kappa) < 0.10, \
            f'offset={offset} kappa={kappa}: 2c={2*coef[2]:+.3f}'

        path = horizon_to_path(coef, XMIN, 3.0, 20, 1.5, 3.0)
        assert path.shape == (20, 6)
        arc = float(np.hypot(*np.diff(path[:, 1:3], axis=0).T).sum())
        assert 1.0 <= arc <= 60.0, f'arc {arc:.2f} m outside MPC accept band'
        assert np.isfinite(ego_to_world(path, (12.0, 1.5, 1.5795))).all()
        print(f'  offset={offset:+.2f} kappa={kappa:+.4f} -> a={coef[0]:+.3f} '
              f'b={coef[1]:+.3f} 2c={2*coef[2]:+.4f} bins={bx.size} '
              f'arc={arc:.2f} m v=[{path[:, 5].min():.2f},{path[:, 5].max():.2f}]')

    # 5. garbage in -> None out, so the node holds instead of steering to it.
    assert fit_lane_center(np.array([1.0, 2.0]), np.array([0.1, 0.2]),
                           8, XMIN, 3.0, 1.00) is None, 'too few bins accepted'
    Xp, Yp = synth(0.0, 0.0)
    assert fit_lane_center(Xp, Yp + 5.0, 8, XMIN, 3.0, 1.00) is None, \
        'lane 5 m off-centre accepted'
    print('selfcheck OK')


def main(args=None):
    if '--selfcheck' in sys.argv:
        selfcheck()
        return
    _main_ros()


if __name__ == '__main__':
    main()
