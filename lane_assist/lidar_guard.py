#!/usr/bin/env python3
"""Lidar emergency brake: free distance ahead along the reference path.

Independent of the camera. Every 3D lidar cloud (/lidar/points) is moved into
base_link, points between z_min and z_max above the road are kept (the road and
the sky are not), and the nearest one inside the corridor of the current
reference path (/planning/ref_path, ego frame, +/- half_width) gives

  free = its x - x_front      (front bumper to the obstacle), max_range if none

published on free_topic (std_msgs/Float64). lane_follow_node caps its speed at
sqrt(2 * brake_decel * (free - stop_margin)), so the car brakes to a stop in
front of anything physically in its path -- whatever the ViT predicted.

The corridor follows the path, so a correct lane change around an obstacle
leaves it outside the corridor and does not brake.

A second, SHORT corridor runs straight ahead along the car's heading, out to
straight_range (~ stopping distance + margin at 1.5 m/s). It catches a path
that swerves away at the last moment -- the 105-map model swerved around full
road blocks through the oncoming lane, and the path corridor alone let it go.
A correct lane change around a passable obstacle has turned the car and moved
it sideways long before the obstacle is that close, so it stays clear.
A hit in that straight corridor is a HARD stop (free = 0), latched for latch_s
after the last hit: braking on the sqrt profile alone still let the car roll,
steer along the swerve and slip around the block (road_test_stop, 2026-10-08).

Self-check (no ROS):  python3 lidar_guard.py --selfcheck
"""

import numpy as np


def free_along_path(pts, path_xy, half_width, z_min, z_max, x_front, max_range, min_hits):
    """pts (N,3) in base_link with z above the road; path_xy (M,2) ego path,
    x increasing. -> metres from the front bumper to the nearest obstacle
    point inside the path corridor, max_range if fewer than min_hits."""
    if pts.size == 0 or len(path_xy) < 2:
        return max_range
    p = pts[(pts[:, 2] >= z_min) & (pts[:, 2] <= z_max)
            & (pts[:, 0] > x_front) & (pts[:, 0] < x_front + max_range)]
    if len(p) < min_hits:
        return max_range
    yc = np.interp(p[:, 0], path_xy[:, 0], path_xy[:, 1])     # beyond the ends: end value
    d = np.sort(p[np.abs(p[:, 1] - yc) < half_width, 0])
    if len(d) < min_hits:
        return max_range
    return float(d[min_hits - 1] - x_front)           # min_hits-th nearest: one noisy return is not a wall


STRAIGHT = np.array([[0.0, 0.0], [1.0, 0.0]])          # ego heading, as a 'path'


def guarded_free(pts, path_xy, half_width, z_min, z_max, x_front, max_range, min_hits, straight_range,
                 straight_hw):
    """-> (free along the path corridor, blocked): blocked = something inside the
    short straight corridor (+/- straight_hw, out to straight_range) -> hard stop."""
    free = free_along_path(pts, path_xy, half_width, z_min, z_max, x_front, max_range, min_hits)
    ahead = free_along_path(pts, STRAIGHT, straight_hw, z_min, z_max, x_front, straight_range, min_hits)
    return free, ahead < straight_range


def _main_ros():
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, qos_profile_sensor_data
    from sensor_msgs.msg import PointCloud2
    from sensor_msgs_py import point_cloud2
    from std_msgs.msg import Float64, Float64MultiArray

    class LidarGuard(Node):
        def __init__(self):
            super().__init__('lidar_guard')
            self.declare_parameter('cloud_topic', '/lidar/points')
            self.declare_parameter('ref_path_topic', '/planning/ref_path')
            self.declare_parameter('free_topic', '/lane_assist/lidar_free')
            self.declare_parameter('lidar_x', 0.165)
            self.declare_parameter('lidar_z', 0.20)
            self.declare_parameter('half_width', 0.20)
            self.declare_parameter('z_min', 0.04)
            self.declare_parameter('z_max', 0.60)
            self.declare_parameter('x_front', 0.39)
            self.declare_parameter('max_range', 8.0)
            self.declare_parameter('min_hits', 3)
            self.declare_parameter('straight_range', 1.8)
            self.declare_parameter('straight_half_width', 0.20)
            self.declare_parameter('latch_s', 1.0)

            g = lambda k: self.get_parameter(k).value            # noqa: E731
            self.off = np.array([float(g('lidar_x')), 0.0, float(g('lidar_z'))])
            self.half_width = float(g('half_width'))
            self.z = (float(g('z_min')), float(g('z_max')))
            self.x_front = float(g('x_front'))
            self.max_range = float(g('max_range'))
            self.min_hits = int(g('min_hits'))
            self.straight_range = float(g('straight_range'))
            self.straight_hw = float(g('straight_half_width'))
            self.latch = float(g('latch_s'))
            self.hit_t = -1e9
            self.path = None
            self.checked = False

            latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                 durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.create_subscription(Float64MultiArray, str(g('ref_path_topic')), self._ref_cb, latched)
            self.create_subscription(PointCloud2, str(g('cloud_topic')), self._cloud_cb, qos_profile_sensor_data)
            self._pub = self.create_publisher(Float64, str(g('free_topic')), 10)
            self.get_logger().info(f'lidar_guard up: corridor +/-{self.half_width:.2f} m, '
                                   f'obstacles {self.z[0]:.2f}..{self.z[1]:.2f} m above the road')

        def _ref_cb(self, msg):
            raw = np.asarray(msg.data, dtype=float)
            if raw.size >= 12 and raw.size % 6 == 0:
                self.path = raw.reshape(-1, 6)[:, 1:3]

        def _cloud_cb(self, msg):
            if self.path is None:
                return
            pts = point_cloud2.read_points_numpy(msg, field_names=('x', 'y', 'z'), skip_nans=True)
            pts = np.asarray(pts, dtype=float).reshape(-1, 3) + self.off
            if not self.checked:
                # mount sanity check, once: the road right ahead must read ~0 m
                near = pts[(pts[:, 0] > 0.6) & (pts[:, 0] < 1.5) & (np.abs(pts[:, 1]) < 0.5)]
                if len(near):
                    self.get_logger().info(f'road height ahead (should be ~0): median z '
                                           f'{np.median(near[:, 2]):+.3f} m over {len(near)} points')
                    self.checked = True
            free, blocked = guarded_free(pts, self.path, self.half_width, self.z[0], self.z[1], self.x_front,
                                         self.max_range, self.min_hits, self.straight_range, self.straight_hw)
            now = self.get_clock().now().nanoseconds * 1e-9
            if blocked:
                self.hit_t = now
            if now - self.hit_t < self.latch:
                free = 0.0                       # hard stop, held: no creeping around the block
            self._pub.publish(Float64(data=free))

    rclpy.init()
    node = LidarGuard()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def selfcheck():
    rng = np.random.default_rng(0)
    path = np.column_stack([np.linspace(0, 5, 16), np.zeros(16)])
    road = np.column_stack([rng.uniform(0.6, 8, 3000), rng.uniform(-2, 2, 3000), rng.normal(0, 0.003, 3000)])
    kw = dict(half_width=0.20, z_min=0.04, z_max=0.6, x_front=0.39, max_range=8.0, min_hits=3)
    assert free_along_path(road, path, **kw) == 8.0, 'bare road must be free'
    # debris 0.25 x 0.25 x 0.08 m centred 2.5 m ahead, in the path
    deb = np.column_stack([rng.uniform(2.375, 2.625, 40), rng.uniform(-0.125, 0.125, 40), rng.uniform(0.0, 0.08, 40)])
    f = free_along_path(np.vstack([road, deb]), path, **kw)
    assert 1.95 < f < 2.15, f
    # the same debris one lane to the left: outside the corridor
    side = deb + [0.0, 0.9, 0.0]
    assert free_along_path(np.vstack([road, side]), path, **kw) == 8.0
    # path changing lanes around in-lane debris: the corridor bends away from it
    lc = np.column_stack([np.linspace(0, 5, 16), 0.9 * np.clip(np.linspace(0, 5, 16) / 2.0, 0, 1)])
    assert free_along_path(np.vstack([road, deb]), lc, **kw) == 8.0
    # a single noisy return is not an obstacle
    assert free_along_path(np.vstack([road, [[1.5, 0.0, 0.1]]]), path, **kw) == 8.0
    # full road block 1.5 m ahead, path swerving away at the last moment: the
    # path corridor misses it, the short straight corridor catches it
    block = np.column_stack([rng.uniform(1.85, 1.95, 300), rng.uniform(-1.0, 0.3, 300), rng.uniform(0.0, 0.08, 300)])
    swerve = np.column_stack([np.linspace(0, 5, 16), 1.2 * np.clip(np.linspace(0, 5, 16) / 1.2, 0, 1)])
    assert free_along_path(np.vstack([road, block]), swerve, **kw) == 8.0
    free, blocked = guarded_free(np.vstack([road, block]), swerve, straight_range=1.8, straight_hw=0.20, **kw)
    assert blocked and free == 8.0, (free, blocked)
    # passable obstacle in the old lane 4 m ahead while changing lanes: beyond the short range -> no brake
    far = deb + [1.5, 0.0, 0.0]
    assert guarded_free(np.vstack([road, far]), lc, straight_range=1.8, straight_hw=0.20, **kw) == (8.0, False)
    # clear pass: a parked car whose body edge is 0.24 m from the car's centre
    # line, 1.5 m ahead -- outside the straight corridor (a pass within ~6 cm of
    # it, like road_test car#60 on 2026-10-08, does trigger the hard stop)
    side_car = np.column_stack([rng.uniform(1.4, 1.85, 200), rng.uniform(-0.44, -0.24, 200), rng.uniform(0.0, 0.14, 200)])
    assert not guarded_free(np.vstack([road, side_car]), lc, straight_range=1.8, straight_hw=0.20, **kw)[1]
    print('selfcheck OK')


def main(args=None):
    _main_ros()


if __name__ == '__main__':
    import sys
    if '--selfcheck' in sys.argv:
        selfcheck()
    else:
        main()
