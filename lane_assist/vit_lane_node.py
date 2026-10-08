#!/usr/bin/env python3
"""ViT lane node: camera frame -> trained ViT -> /planning/ref_path.

Replaces the classical lane detector in the loop. No paint mask, no line fitting: the
fine-tuned ViT (tools/train/train_vit.py) looks at the frame cropped below the
horizon and predicts the path the car should follow -- lane-centre Y at
vit_lane.XS (0..3 m ahead, base_link) -- including lane changes around
obstacles it was trained on.

Per tick (publish_rate_hz):
  latest frame -> crop_resize -> ViT -> 16 Y values -> EMA
  -> (N,6) [s, x, y, psi, kappa, vx] in the EGO frame, same contract as
     classical detector, so lane_follow_node / lpv_mpc track it unchanged.
  vx = min(target_speed, sqrt(a_lat_max / |kappa|)), and with a stop-head
  checkpoint (17 outputs) also <= sqrt(2 * brake_decel * (free - stop_margin)):
  the predicted free distance brings the car to a smooth stop before an
  obstacle it cannot pass.

Fail-safe: no fresh frame for frame_timeout_s -> nothing is published, and
lane_follow_node's ref_timeout stops the car.

Also publishes, for humans: /planning/ref_path_viz (map frame, RViz) and a
debug image with the prediction drawn on the camera frame.

Run log (enable_csv_log), in log_dir -- vit_follow_launch.py gives every run
its own log/vit_run_<stamp>/ shared with lane_follow_node's follow_*.csv:
  vit.csv    one row per tick: sim time, odom pose + twist, camera frame age,
             inference time, raw and smoothed 16-point prediction
  frames/    the raw camera frame (the model's input) at log_frames_hz, named by sim time
tools/analysis/analyze_run.py scores a run folder against the map's ground truth.
"""

import math
import os
import time

import numpy as np


def path_from_pred(Y, xs, target_speed, a_lat_max):
    """Predicted lane-centre Y at xs -> ego (N,6) [s, x, y, psi, kappa, vx]."""
    dY = np.gradient(Y, xs, edge_order=2)
    d2Y = np.gradient(dY, xs, edge_order=2)
    psi = np.arctan(dY)
    kappa = d2Y / np.power(1.0 + dY * dY, 1.5)
    vx = np.minimum(target_speed, np.sqrt(a_lat_max / np.maximum(np.abs(kappa), 1e-6)))
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(xs), np.diff(Y)))])
    return np.column_stack([s, xs, Y, psi, kappa, vx])


def _main_ros():
    import rclpy
    import torch
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from sensor_msgs.msg import Image, CameraInfo
    from nav_msgs.msg import Odometry, Path
    from geometry_msgs.msg import PoseStamped
    from std_msgs.msg import Float64MultiArray
    from lane_assist.camera_geometry import horizon_row, pixel_from_ground, ego_to_world
    from lane_assist.vit_lane import FREE_SCALE, XS, build_lane_model, preprocess

    class VitLaneNode(Node):
        def __init__(self):
            super().__init__('vit_lane_node')
            self.declare_parameter('vit_checkpoint', '')
            self.declare_parameter('device', 'cuda')
            self.declare_parameter('camera_topic', '/camera/color/image_raw')
            self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
            self.declare_parameter('odom_topic', '/ego_racecar/odom')
            self.declare_parameter('ref_path_topic', '/planning/ref_path')
            self.declare_parameter('ref_viz_topic', '/planning/ref_path_viz')
            self.declare_parameter('debug_image_topic', '/vit_lane/debug_image')
            self.declare_parameter('publish_debug_image', True)
            self.declare_parameter('cam_height', 0.40)
            self.declare_parameter('cam_pitch', 0.0463)
            self.declare_parameter('cam_x', 0.275)
            self.declare_parameter('target_speed', 1.5)
            self.declare_parameter('a_lat_max', 3.0)
            self.declare_parameter('ema_alpha', 0.6)
            self.declare_parameter('publish_rate_hz', 20.0)
            self.declare_parameter('frame_timeout_s', 0.5)
            self.declare_parameter('enable_csv_log', True)
            self.declare_parameter('log_dir', '')
            self.declare_parameter('log_frames_hz', 2.0)
            self.declare_parameter('brake_decel', 1.0)
            self.declare_parameter('stop_margin', 0.4)

            g = lambda k: self.get_parameter(k).value            # noqa: E731
            ckpt_path = os.path.expanduser(str(g('vit_checkpoint')))
            if not os.path.isfile(ckpt_path):
                raise SystemExit(f'vit_checkpoint not found: "{ckpt_path}" '
                                 '(train one with tools/train/train_vit.py)')
            dev = str(g('device'))
            self.device = dev if (dev != 'cuda' or torch.cuda.is_available()) else 'cpu'
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            self.n_out = int(ckpt['state_dict']['head.weight'].shape[0])   # 16, or 17 with the stop head
            self.model = build_lane_model(npz=None, n_out=self.n_out)
            self.model.load_state_dict(ckpt['state_dict'])
            self.model.to(self.device).eval()
            self.xs = np.asarray(ckpt.get('xs', XS), float)
            self.torch = torch

            self.cam = (float(g('cam_height')), float(g('cam_pitch')), float(g('cam_x')))
            self.target_speed = float(g('target_speed'))
            self.a_lat_max = float(g('a_lat_max'))
            self.alpha = float(g('ema_alpha'))
            self.timeout = float(g('frame_timeout_s'))
            self.brake = float(g('brake_decel'))
            self.stop_margin = float(g('stop_margin'))
            self.free = float('nan')
            self.debug_on = bool(g('publish_debug_image'))

            self.K = None
            self.top = None
            self.img = None
            self.img_t = None
            self.pose = None
            self.Y = None
            self.twist = (0.0, 0.0, 0.0)
            self.ckpt_path = ckpt_path
            self._init_log(str(g('log_dir')) if bool(g('enable_csv_log')) else '',
                           float(g('log_frames_hz')))

            best = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
            latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                 durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.create_subscription(Image, str(g('camera_topic')), self._img_cb, best)
            self.create_subscription(CameraInfo, str(g('camera_info_topic')), self._info_cb, best)
            self.create_subscription(Odometry, str(g('odom_topic')), self._odom_cb, best)
            self._pub = self.create_publisher(Float64MultiArray, str(g('ref_path_topic')), latched)
            self._viz = self.create_publisher(
                Path, str(g('ref_viz_topic')), QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE))
            self._dbg = (self.create_publisher(Image, str(g('debug_image_topic')), best)
                         if self.debug_on else None)
            self.create_timer(1.0 / max(float(g('publish_rate_hz')), 1e-3), self._tick)
            self.get_logger().info(
                f'vit_lane_node up: {ckpt_path} (trained on {ckpt.get("maps")}, '
                f'val {ckpt.get("val_mae", float("nan")):.3f} m, '
                f'test {ckpt.get("test_mae", float("nan")):.3f} m) on {self.device}')

        def _now(self):
            return self.get_clock().now().nanoseconds * 1e-9

        def _info_cb(self, m):
            if self.K is None:
                self.K = (m.k[0], m.k[4], m.k[2], m.k[5])
                self.top = int(max(0, math.ceil(horizon_row(self.K, self.cam[1])) + 2))
                self.get_logger().info(f'intrinsics {self.K}, crop from row {self.top}')

        def _img_cb(self, m):
            a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1)[:, :, :3]
            if m.encoding.lower().startswith('bgr'):
                a = a[:, :, ::-1]
            self.img = np.ascontiguousarray(a)
            self.img_t = self._now()

        def _odom_cb(self, m):
            q = m.pose.pose.orientation
            yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            self.pose = (m.pose.pose.position.x, m.pose.pose.position.y, yaw)
            t = m.twist.twist
            self.twist = (t.linear.x, t.linear.y, t.angular.z)

        def _tick(self):
            if self.K is None or self.img is None or self._now() - self.img_t > self.timeout:
                return
            t0 = time.perf_counter()
            x = preprocess(self.img, self.top)[None].to(self.device)
            with self.torch.no_grad():
                Y = self.model(x)[0].float().cpu().numpy()
            infer_ms = (time.perf_counter() - t0) * 1000
            self.Y = Y if self.Y is None else self.alpha * Y + (1 - self.alpha) * self.Y
            path = path_from_pred(self.Y[:len(self.xs)], self.xs, self.target_speed, self.a_lat_max)
            if self.n_out > len(self.xs):
                # stop head: predicted free distance -> brake to a stop short of
                # an obstacle with no lane to pass it (vx 0 = follower's HOLD)
                self.free = float(self.Y[len(self.xs)]) / FREE_SCALE
                v_stop = math.sqrt(2.0 * self.brake * max(0.0, self.free - self.stop_margin))
                path[:, 5] = np.minimum(path[:, 5], v_stop if v_stop > 0.05 else 0.0)
            msg = Float64MultiArray()
            msg.data = path.ravel().tolist()
            self._pub.publish(msg)
            if self.pose is not None:
                pa = Path()
                pa.header.frame_id = 'map'
                pa.header.stamp = self.get_clock().now().to_msg()
                for row in ego_to_world(path, self.pose):
                    ps = PoseStamped()
                    ps.header = pa.header
                    ps.pose.position.x = float(row[1])
                    ps.pose.position.y = float(row[2])
                    ps.pose.position.z = 0.15
                    ps.pose.orientation.z = math.sin(0.5 * float(row[3]))
                    ps.pose.orientation.w = math.cos(0.5 * float(row[3]))
                    pa.poses.append(ps)
                self._viz.publish(pa)
            if self._dbg is not None:
                self._publish_debug()
            self._record(Y, infer_ms)

        # ── run log ──
        def _init_log(self, d, frames_hz):
            self._csv = None
            if not d:
                return
            import csv
            d = os.path.expanduser(d)
            try:
                os.makedirs(os.path.join(d, 'frames'), exist_ok=True)
                self._csv_file = open(os.path.join(d, 'vit.csv'), 'w', newline='')
            except OSError as e:
                self.get_logger().warn(f'run log disabled: {e}')
                return
            self._csv = csv.writer(self._csv_file)
            self._csv.writerow(['wall_t', 'sim_t', 'x', 'y', 'yaw', 'vx', 'vy', 'omega',
                                'frame_age_s', 'infer_ms', 'pred', 'pred_raw'])
            with open(os.path.join(d, 'run.txt'), 'w') as fh:
                fh.write(f'checkpoint: {self.ckpt_path}\nxs: {" ".join(f"{v:.3f}" for v in self.xs)}\n'
                         f'cam: {self.cam}\nema_alpha: {self.alpha}\ntarget_speed: {self.target_speed}\n')
            self._log_dir = d
            self._frame_dt = 1.0 / frames_hz if frames_hz > 0 else float('inf')
            self._last_frame_t = -1e9
            self.get_logger().info(f'run log -> {d}')

        def _record(self, raw, infer_ms):
            """Never let logging take the driving down: warn once and stop."""
            if self._csv is None or self.pose is None:
                return
            try:
                now = self._now()
                ser = lambda a: ' '.join(f'{v:.4f}' for v in a)            # noqa: E731
                self._csv.writerow([f'{time.time():.3f}', f'{now:.3f}',
                                    *(f'{v:.5f}' for v in self.pose), *(f'{v:.4f}' for v in self.twist),
                                    f'{now - self.img_t:.3f}', f'{infer_ms:.1f}', ser(self.Y), ser(raw)])
                self._csv_file.flush()
                if now - self._last_frame_t >= self._frame_dt:
                    import cv2
                    self._last_frame_t = now
                    # the RAW frame, exactly what the model saw (the overlay can be
                    # redrawn from vit.csv; a drawn-on frame cannot be re-run through the model)
                    cv2.imwrite(os.path.join(self._log_dir, 'frames', f'{now:010.3f}.jpg'),
                                self.img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
            except Exception as e:                       # noqa: BLE001
                self.get_logger().error(f'run log disabled after {type(e).__name__}: {e}')
                self._csv = None

        def _debug_frame(self):
            import cv2
            out = self.img.copy()
            Xd = np.linspace(self.xs[0], self.xs[-1], 60)
            u, v = pixel_from_ground(Xd, np.interp(Xd, self.xs, self.Y[:len(self.xs)]), self.K, *self.cam)
            ok = np.isfinite(u) & np.isfinite(v) & (Xd > self.cam[2] + 0.05)
            cv2.polylines(out, [np.stack([u[ok], v[ok]], 1).astype(np.int32)], False,
                          (255, 0, 0), 3, cv2.LINE_AA)
            cv2.line(out, (0, self.top), (out.shape[1], self.top), (0, 0, 255), 1)
            cv2.putText(out, f'ViT  y(car) {self.Y[0]:+.2f}  y({self.xs[-1]:.0f}m) {self.Y[len(self.xs) - 1]:+.2f} m'
                             f'  free {self.free:.1f} m',
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            return out

        def _publish_debug(self):
            out = self._debug_frame()
            m = Image()
            m.height, m.width = out.shape[:2]
            m.encoding = 'rgb8'
            m.step = 3 * m.width
            m.data = out.tobytes()
            self._dbg.publish(m)

    rclpy.init()
    node = VitLaneNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def selfcheck():
    """path_from_pred: straight offset lane and a constant-curvature arc."""
    xs = np.linspace(0.0, 3.0, 16)
    p = path_from_pred(np.full(16, 0.2), xs, 1.5, 3.0)
    assert p.shape == (16, 6) and np.allclose(p[:, 2], 0.2) and np.allclose(p[:, 3:5], 0)
    assert np.allclose(p[:, 5], 1.5) and abs(p[-1, 0] - 3.0) < 1e-9
    k = 0.2                                     # Y = k x^2 / 2 -> kappa ~ k near x=0
    p = path_from_pred(0.5 * k * xs ** 2, xs, 1.5, 3.0)
    assert abs(p[0, 4] - k) < 0.01 and abs(p[2, 4] - k) < 0.01 and p[-1, 4] > 0
    assert np.all(p[:, 5] <= 1.5) and np.all(p[:, 5] > 0)
    print('selfcheck OK')


def main(args=None):
    import sys
    if '--selfcheck' in sys.argv:
        selfcheck()
        return
    _main_ros()


if __name__ == '__main__':
    main()
