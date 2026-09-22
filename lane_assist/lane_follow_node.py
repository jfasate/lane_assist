#!/usr/bin/env python3
"""Pure-pursuit lane follower: /planning/ref_path -> /drive.

The diagnostic counterpart to lpv_mpc_node for the camera pipeline. It answers
one question the MPC stack cannot answer cleanly, because too much sits between
the camera and the wheels: does lane_detector produce a reference good enough to
drive on?

  car keeps the lane  -> perception is sound, the MPC layer is what is fighting
  car does not        -> the fault is in the detector

Why pure pursuit is enough here: the reference is ~2.3 m long (the camera cannot
see further) and the speed is whatever the detector profiled, so at 1.5 m/s
there is ~1.5 s of preview and the lane is a quadratic. There is nothing to
optimise, so the whole apparatus the MPC brings -- 60-step QP, slack variables,
recovery/wall/obstacle guards, speed blending -- buys nothing and has its own
failure modes.

The reference arrives in the EGO frame (base_link), so there is no transform
here at all, and the whole class of frame bugs cannot occur.

Self-check (no ROS needed):  python3 lane_follow_node.py --selfcheck
"""

import math
import os
import sys
import time

import numpy as np


def pursuit_steer(pts, lookahead, wheelbase):
    """(N,2) ego-frame path + lookahead -> (steer_rad, goal_x, goal_y, alpha).

    Picks the first point at least `lookahead` away and aims the front axle at
    it. Points behind the car are skipped; if the path is shorter than the
    lookahead the far end is used, which is the right behaviour for a camera
    horizon that is often shorter than we would like.
    """
    fwd = pts[pts[:, 0] > 0.0]
    if fwd.shape[0] == 0:
        return None
    d = np.hypot(fwd[:, 0], fwd[:, 1])
    i = int(np.argmax(d >= lookahead)) if (d >= lookahead).any() else int(d.argmax())
    gx, gy = float(fwd[i, 0]), float(fwd[i, 1])
    ld = math.hypot(gx, gy)
    if ld < 1e-3:
        return None
    alpha = math.atan2(gy, gx)
    return math.atan(2.0 * wheelbase * math.sin(alpha) / ld), gx, gy, alpha


def _main_ros():
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from std_msgs.msg import Float64MultiArray
    from nav_msgs.msg import Odometry
    from ackermann_msgs.msg import AckermannDriveStamped

    class LaneFollower(Node):
        def __init__(self):
            super().__init__('lane_follow_node')
            self.declare_parameter('ref_path_topic', '/planning/ref_path')
            self.declare_parameter('drive_topic', '/drive')
            self.declare_parameter('odom_topic', '/ego_racecar/odom')
            self.declare_parameter('lookahead', 1.2)
            self.declare_parameter('wheelbase', 0.3302)
            self.declare_parameter('max_steer', 0.4189)
            self.declare_parameter('max_speed', 2.0)
            self.declare_parameter('stop_speed_thresh', 0.05)
            self.declare_parameter('ref_timeout', 0.5)
            self.declare_parameter('control_rate_hz', 50.0)
            self.declare_parameter('enable_csv_log', True)
            self.declare_parameter('log_dir', '')

            g = lambda k: self.get_parameter(k).value            # noqa: E731
            self.lookahead = float(g('lookahead'))
            self.wheelbase = float(g('wheelbase'))
            self.max_steer = float(g('max_steer'))
            self.max_speed = float(g('max_speed'))
            self.stop_thresh = float(g('stop_speed_thresh'))
            self.ref_timeout = float(g('ref_timeout'))

            self.path = None
            self.path_t = None
            self.vx = 0.0

            best = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
            latched = QoSProfile(
                depth=1, reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.create_subscription(Float64MultiArray, str(g('ref_path_topic')),
                                     self._ref_cb, latched)
            self.create_subscription(Odometry, str(g('odom_topic')),
                                     self._odom_cb, best)
            self._drive = self.create_publisher(
                AckermannDriveStamped, str(g('drive_topic')), 1)

            rate = float(g('control_rate_hz'))
            self.create_timer(1.0 / max(rate, 1e-3), self._tick)
            self._init_csv_log()
            self.get_logger().info(
                f'lane_follow up @ {rate:.0f} Hz  lookahead={self.lookahead:.2f} m  '
                f'max_speed={self.max_speed:.2f} m/s')

        def _ref_cb(self, msg):
            raw = np.asarray(msg.data, dtype=float)
            if raw.size < 12 or raw.size % 6:
                return
            self.path = raw.reshape(-1, 6)
            self.path_t = self.get_clock().now().nanoseconds * 1e-9

        def _odom_cb(self, msg):
            self.vx = msg.twist.twist.linear.x

        def _tick(self):
            now = self.get_clock().now().nanoseconds * 1e-9
            # A reference that stopped arriving is the dangerous case: without
            # this the car would coast on the last command indefinitely.
            if self.path is None:
                self._publish(0.0, 0.0)
                self._log('no_ref', now, None)
                return
            age = now - self.path_t
            if age > self.ref_timeout:
                self._publish(0.0, 0.0)
                self._log('ref_stale', now, None)
                return

            path_vx = float(np.nanmax(np.abs(self.path[:, 5])))
            if path_vx <= self.stop_thresh:      # detector's explicit HOLD
                self._publish(0.0, 0.0)
                self._log('hold', now, None)
                return

            got = pursuit_steer(self.path[:, 1:3], self.lookahead, self.wheelbase)
            if got is None:
                self._publish(0.0, 0.0)
                self._log('bad_path', now, None)
                return
            steer, gx, gy, alpha = got
            steer = float(np.clip(steer, -self.max_steer, self.max_steer))
            speed = min(path_vx, self.max_speed)
            self._publish(steer, speed)
            self._log('ok', now, (steer, speed, gx, gy, alpha, age))

        def _publish(self, steer, speed):
            m = AckermannDriveStamped()
            m.header.stamp = self.get_clock().now().to_msg()
            m.drive.steering_angle = float(steer)
            m.drive.speed = float(speed)
            self._drive.publish(m)

        LOG_COLUMNS = [
            'wall_t', 'sim_t', 'status', 'ref_age_s', 'n_pts',
            'lane_y_at_car', 'goal_x', 'goal_y', 'alpha_deg',
            'steer_deg', 'speed_cmd', 'vx_meas',
        ]

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
                p = os.path.join(d, f'follow_{stamp}.csv')
                self._csv_file = open(p, 'w', newline='')
                self._csv_writer = csv.writer(self._csv_file)
                self._csv_writer.writerow(self.LOG_COLUMNS)
                self.get_logger().info(f'follow log -> {p}')
            except OSError as e:
                self.get_logger().warn(f'follow log disabled: {e}')

        def _log(self, status, now, got):
            if self._csv_writer is None:
                return
            nan = float('nan')
            n = 0 if self.path is None else int(self.path.shape[0])
            # y of the reference at the car: the lane offset the follower is
            # reacting to, and the number to compare against the detector's
            # lane_offset_m.
            y0 = float(self.path[0, 2]) if n else nan
            if got is None:
                steer = speed = gx = gy = alpha = nan
                age = nan if self.path_t is None else now - self.path_t
            else:
                steer, speed, gx, gy, alpha, age = got
            self._csv_writer.writerow([
                f'{time.time():.4f}', f'{now:.4f}', status, f'{age:.4f}', n,
                f'{y0:.4f}', f'{gx:.4f}', f'{gy:.4f}',
                f'{math.degrees(alpha):.3f}', f'{math.degrees(steer):.3f}',
                f'{speed:.4f}', f'{self.vx:.4f}'])
            self._csv_file.flush()

        def _close_log(self):
            if self._csv_file is not None:
                self._csv_file.close()
                self._csv_file = None
                self._csv_writer = None

    rclpy.init()
    node = LaneFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._close_log()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def selfcheck():
    L = 0.3302
    # straight lane dead ahead -> no steering
    pts = np.column_stack([np.linspace(0, 2.3, 20), np.zeros(20)])
    steer, gx, gy, a = pursuit_steer(pts, 1.2, L)
    assert abs(steer) < 1e-9, steer
    assert gx >= 1.2, gx

    # lane centre 0.2 m to the LEFT -> steer left (positive)
    pts = np.column_stack([np.linspace(0, 2.3, 20), np.full(20, 0.2)])
    steer, *_ = pursuit_steer(pts, 1.2, L)
    assert steer > 0.0, steer
    # mirrored
    pts = np.column_stack([np.linspace(0, 2.3, 20), np.full(20, -0.2)])
    steer, *_ = pursuit_steer(pts, 1.2, L)
    assert steer < 0.0, steer

    # bigger offset -> more steering
    def s(off):
        p = np.column_stack([np.linspace(0, 2.3, 20), np.full(20, off)])
        return pursuit_steer(p, 1.2, L)[0]
    assert s(0.4) > s(0.2) > s(0.05) > 0.0

    # shorter lookahead -> more aggressive for the same offset
    p = np.column_stack([np.linspace(0, 2.3, 20), np.full(20, 0.2)])
    assert pursuit_steer(p, 0.8, L)[0] > pursuit_steer(p, 1.6, L)[0]

    # path shorter than the lookahead: use the far end, do not give up
    short = np.column_stack([np.linspace(0, 0.6, 6), np.full(6, 0.1)])
    got = pursuit_steer(short, 1.2, L)
    assert got is not None and got[1] <= 0.6

    # a curving lane produces a steer of the same sign as the curve
    x = np.linspace(0, 2.3, 20)
    pts = np.column_stack([x, 0.15 * x ** 2])
    assert pursuit_steer(pts, 1.2, L)[0] > 0.0

    # nothing ahead -> no command
    assert pursuit_steer(np.array([[-1.0, 0.0], [-2.0, 0.0]]), 1.2, L) is None

    print('selfcheck OK')


def main(args=None):
    if '--selfcheck' in sys.argv:
        selfcheck()
        return
    _main_ros()


if __name__ == '__main__':
    main()
