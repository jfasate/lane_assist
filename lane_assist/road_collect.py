#!/usr/bin/env python3
"""Episodic driving data collector for the ViT lane model.

For one road_NN map (lane_assist/tools/maps), episode after episode:
  1. place the car. obstacle_start_frac of episodes start obstacle_start_min..
     max metres behind an obstacle that blocks its lane (an avoidance episode,
     which ends obstacle_end_m past it); the rest start at a random point in a
     random lane (a lane-keeping episode, up to episode_s long). late_frac of
     the avoidance episodes are LATE: hold the lane until late_trigger_min..max
     m before the obstacle, then drive eval_tools.Truth.late_label() -- the
     states a hesitating model reaches, which a perfect expert never visits.
     stop_frac of episodes start behind a FULL road block (no lane to pass in):
     brake to a stop short of it (eval_tools.free_distance), hold, end; any
     episode that reaches such a block does the same. wrong_frac of lane-keeping
     episodes start across the centre line in the oncoming lane and steer back
     (episode ends at wrong_end_s). right_pass_frac of avoidance episodes pick an
     obstacle whose only legal pass is on the right.
  2. drive the EXPERT PLAN, eval_tools.Truth.plan_label(): the true lane
     centre, or the smooth lane change around an obstacle in the lane (into a
     free lane of the same direction; never a bike / bus / parking lane, never
     across the centre line). Pure pursuit, rate-limited steering, speed ramp,
     plus a slow sinusoidal weave so the car drifts and recovers; the weave
     fades out near obstacles and during lane changes
  3. record camera frames at record_hz with the odom pose nearest each frame
  4. end on timeout, on drifting off the plan, or where the plan has no free
     lane (an obstacle that cannot be passed); repeat until n_samples frames

The label is NOT the weave: make_dataset.py rebuilds it from the pose with the
same plan_label(), so collector, labels and tests always agree.

Output, per map, in out_dir/<map>/ (or out_dir/<map>__<run_tag>/ for an
extra run on the same map, e.g. an avoidance top-up):
  NNNNN.jpg      raw camera frame
  poses.csv      file, x, y, yaw, lane, changing, sim_t, episode, kind, speed, steer
  camera.txt     fx fy cx cy width height
"""

import collections
import csv
import math
import os

import numpy as np

from lane_assist.eval_tools import Truth, CAR_HALF_W, OBJ_HALF_W, IN_PATH_MARGIN, FREE_MAX
from lane_assist.lane_follow_node import pursuit_steer

LATE_XS = np.linspace(0.0, 5.0, 26)        # late-change path samples the car drives [m ahead]


def _main_ros():
    import cv2
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image, CameraInfo
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import PoseWithCovarianceStamped
    from ackermann_msgs.msg import AckermannDriveStamped

    class RoadCollect(Node):
        def __init__(self):
            super().__init__('road_collect')
            self.declare_parameter('map_name', '')
            self.declare_parameter('truth_dir', '')
            self.declare_parameter('out_dir', '')
            self.declare_parameter('n_samples', 300)
            self.declare_parameter('record_hz', 5.0)
            self.declare_parameter('episode_s', 60.0)
            self.declare_parameter('warmup_s', 0.5)
            self.declare_parameter('speed_min', 0.8)
            self.declare_parameter('speed_max', 2.0)
            self.declare_parameter('accel', 1.0)
            self.declare_parameter('lat_max_frac', 0.35)
            self.declare_parameter('yaw_max', 0.30)
            self.declare_parameter('weave_max_frac', 0.30)
            self.declare_parameter('weave_period_min', 6.0)
            self.declare_parameter('weave_period_max', 14.0)
            self.declare_parameter('record_max_frac', 0.45)
            self.declare_parameter('abort_frac', 0.5)
            self.declare_parameter('lookahead_min', 0.8)
            self.declare_parameter('lookahead_gain', 0.5)
            self.declare_parameter('lookahead_max', 1.6)
            self.declare_parameter('wheelbase', 0.3302)
            self.declare_parameter('max_steer', 0.4189)
            self.declare_parameter('steer_rate', 1.0)
            self.declare_parameter('obstacle_calm_m', 4.5)
            self.declare_parameter('object_clearance', 0.6)
            self.declare_parameter('obstacle_start_frac', 0.4)
            self.declare_parameter('obstacle_start_min', 7.0)
            self.declare_parameter('obstacle_start_max', 11.0)
            self.declare_parameter('obstacle_end_m', 4.0)
            self.declare_parameter('late_frac', 0.5)
            self.declare_parameter('late_trigger_min', 2.2)
            self.declare_parameter('late_trigger_max', 4.5)
            self.declare_parameter('stop_frac', 0.15)
            self.declare_parameter('stop_hold_s', 1.0)
            self.declare_parameter('brake_decel', 1.0)
            self.declare_parameter('stop_margin', 0.4)
            self.declare_parameter('wrong_frac', 0.1)
            self.declare_parameter('wrong_s', 4.0)
            self.declare_parameter('wrong_end_s', 6.0)
            self.declare_parameter('right_pass_frac', 0.3)
            self.declare_parameter('settle_s', 0.25)
            self.declare_parameter('seed', 0)
            self.declare_parameter('run_tag', '')
            self.declare_parameter('camera_topic', '/camera/color/image_raw')
            self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
            self.declare_parameter('odom_topic', '/ego_racecar/odom')
            self.declare_parameter('drive_topic', '/drive')

            g = lambda k: self.get_parameter(k).value            # noqa: E731
            self.map = str(g('map_name'))
            self.T = Truth(self.map, os.path.expanduser(str(g('truth_dir'))))
            T = self.T
            obs = [p[:2] for p, k in zip(T.obj, T.kinds) if k not in ('barrier', 'bus')]
            self.obs = np.array(obs).reshape(-1, 2)
            self.lane_ok = T.exists.copy()
            for k in range(len(T.paths)):
                for o in self.obs:
                    self.lane_ok[k] &= np.hypot(*(T.paths[k] - o).T) > float(g('object_clearance'))
            # avoidance starts: (lane, index of the obstacle on that lane) for
            # every obstacle the plan passes in a free lane
            self.approach, self.approach_right = [], []
            for k, start, j, tgt in T.maneuvers():
                if tgt >= 0:
                    a = int(np.argmin(np.hypot(*(T.paths[k] - T.obj[j][:2]).T)))
                    self.approach.append((k, a))
                    if T.passes_right(k, tgt):                  # legal pass is on the RIGHT
                        self.approach_right.append((k, a))
            self.approach = list(dict.fromkeys(self.approach))
            self.approach_right = list(dict.fromkeys(self.approach_right))
            # stop starts: obstacles with NO free lane beside them (full road blocks)
            self.blocked_at = list(dict.fromkeys(
                (k, int(np.argmin(np.hypot(*(T.paths[k] - T.obj[j][:2]).T))))
                for k, start, j, tgt in T.maneuvers() if tgt < 0))

            self.n = int(g('n_samples'))
            self.rec_dt = 1.0 / float(g('record_hz'))
            self.episode_s = float(g('episode_s'))
            self.warmup = float(g('warmup_s'))
            self.v_range = (float(g('speed_min')), float(g('speed_max')))
            self.accel = float(g('accel'))
            self.lat_frac = float(g('lat_max_frac'))
            self.yaw_max = float(g('yaw_max'))
            self.weave_frac = float(g('weave_max_frac'))
            self.weave_T = (float(g('weave_period_min')), float(g('weave_period_max')))
            self.rec_max = float(g('record_max_frac'))
            self.abort = float(g('abort_frac'))
            self.ld = (float(g('lookahead_min')), float(g('lookahead_gain')),
                       float(g('lookahead_max')))
            self.wheelbase = float(g('wheelbase'))
            self.max_steer = float(g('max_steer'))
            self.steer_rate = float(g('steer_rate'))
            self.calm_m = float(g('obstacle_calm_m'))
            self.obs_start_frac = float(g('obstacle_start_frac'))
            self.obs_start = (float(g('obstacle_start_min')), float(g('obstacle_start_max')))
            self.obs_end = float(g('obstacle_end_m'))
            self.late_frac = float(g('late_frac'))
            self.late_trigger = (float(g('late_trigger_min')), float(g('late_trigger_max')))
            self.stop_frac = float(g('stop_frac'))
            self.stop_hold = float(g('stop_hold_s'))
            self.brake = float(g('brake_decel'))
            self.stop_margin = float(g('stop_margin'))
            self.wrong_frac = float(g('wrong_frac'))
            self.wrong_s = float(g('wrong_s'))
            self.wrong_end = float(g('wrong_end_s'))
            self.right_frac = float(g('right_pass_frac'))
            self.settle = float(g('settle_s'))
            self.rng = np.random.default_rng(int(g('seed')))
            tag = str(g('run_tag'))
            self.dir = os.path.join(os.path.expanduser(str(g('out_dir'))),
                                    self.map + ('__' + tag if tag else ''))
            os.makedirs(self.dir, exist_ok=True)

            self.img = None
            self.K = None
            self.odom = collections.deque(maxlen=200)
            self.state = 'place'
            self.ep = -1
            self.count = 0
            self.n_lc = 0
            self.last_img_t = -1.0
            self.last_rec_t = -1e9

            best = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
            self.create_subscription(Image, str(g('camera_topic')), self._img_cb, best)
            self.create_subscription(CameraInfo, str(g('camera_info_topic')), self._info_cb, best)
            self.create_subscription(Odometry, str(g('odom_topic')), self._odom_cb,
                                     QoSProfile(depth=50, reliability=ReliabilityPolicy.BEST_EFFORT))
            self._pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
            self._drive = self.create_publisher(AckermannDriveStamped, str(g('drive_topic')), 1)
            self._csv_file = open(os.path.join(self.dir, 'poses.csv'), 'w', newline='')
            self._csv = csv.writer(self._csv_file)
            self._csv.writerow(['file', 'x', 'y', 'yaw', 'lane', 'changing', 'sim_t', 'episode',
                                'kind', 'speed', 'steer'])
            self.create_timer(0.02, self._tick)
            self.get_logger().info(
                f'road_collect {self.map}: {self.n} frames @ {1 / self.rec_dt:.0f} Hz, '
                f'{len(T.paths)} lanes, {len(self.obs)} obstacles, '
                f'{len(self.approach)} passable blockers -> {self.dir}')

        # ── callbacks ──
        def _now(self):
            return self.get_clock().now().nanoseconds * 1e-9

        def _img_cb(self, m):
            self.img = m

        def _info_cb(self, m):
            if self.K is None:
                self.K = m
                with open(os.path.join(self.dir, 'camera.txt'), 'w') as fh:
                    fh.write(f'{m.k[0]} {m.k[4]} {m.k[2]} {m.k[5]} {m.width} {m.height}\n')

        def _odom_cb(self, m):
            q = m.pose.pose.orientation
            yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
            self.odom.append((t, m.pose.pose.position.x, m.pose.pose.position.y, yaw))

        # ── episode ──
        def _drive_cmd(self, v, steer):
            m = AckermannDriveStamped()
            m.header.stamp = self.get_clock().now().to_msg()
            m.drive.speed = float(v)
            m.drive.steering_angle = float(steer)
            self._drive.publish(m)

        def _place(self):
            T = self.T
            n = T.paths.shape[1]
            r = self.rng.random()
            wrong = False
            if self.blocked_at and r < self.stop_frac:
                k, a = self.blocked_at[int(self.rng.integers(len(self.blocked_at)))]
                i = (a - int(self.rng.uniform(*self.obs_start) / 0.05)) % n
                if not self.lane_ok[k][i]:
                    return
                # nothing else in the lane between the start and the block: a passable
                # obstacle just ahead of the start leaves no room to pass it (re-place)
                d_block = ((a - i) % n) * 0.05             # (the block's own taper spans ~3 m before it)
                if any(lat < CAR_HALF_W + OBJ_HALF_W[T.kinds[j]] + IN_PATH_MARGIN and aa * 0.05 < d_block - 3.5
                       for aa, lat, j in T.obstacles_along(k, i, (a - i) % n)):
                    return
                self.kind, self.stop_i = 'stop', None
            elif self.approach and r < (self.stop_frac if self.blocked_at else 0.0) + self.obs_start_frac:
                # right_pass_frac: an obstacle whose only legal pass is to the RIGHT
                # (inner lane next to the centre line) -- the model passed those on
                # the left, through the oncoming lane (road_test, 2026-10-07)
                pool = (self.approach_right if self.approach_right and self.rng.random() < self.right_frac
                        else self.approach)
                k, a = pool[int(self.rng.integers(len(pool)))]
                i = (a - int(self.rng.uniform(*self.obs_start) / 0.05)) % n
                if not self.lane_ok[k][i]:
                    return
                self.kind = 'late' if self.rng.random() < self.late_frac else 'avoid'
                self.trigger = self.rng.uniform(*self.late_trigger)
                self.stop_i = (a + int(self.obs_end / 0.05)) % n
            else:
                k = int(self.rng.integers(0, len(T.paths)))
                idx = np.flatnonzero(self.lane_ok[k])
                if idx.size == 0:
                    return
                i = int(self.rng.choice(idx))
                self.kind, self.stop_i = 'keep', None
                wrong = self.rng.random() < self.wrong_frac
            w = T.widths[k][i]
            lat = self.rng.uniform(-self.lat_frac, self.lat_frac) * w
            self.target = T.pose_on_lane(k, i, lat, self.rng.uniform(-self.yaw_max, self.yaw_max))
            if wrong:
                # wrong side: across the centre line into the oncoming lane, still
                # facing our way; the expert steers back (the state the model
                # drifted into on road_test, which the expert never visits)
                pose = T.pose_on_lane(k, i, self.rng.uniform(0.8, 1.3) * w, self.rng.uniform(-0.1, 0.1))
                if T.oncoming_at(*pose):
                    self.target, self.kind = pose, 'wrong'
            self.stopped_s = 0.0
            self.v_set = self.rng.uniform(*self.v_range)
            self.v = 0.0
            self.steer = 0.0
            self.weave_gain = 1.0
            amp = self.rng.uniform(0.0, self.weave_frac, 2)
            amp *= self.weave_frac / max(self.weave_frac, amp.sum())
            self.weave = [(amp[j], self.rng.uniform(*self.weave_T), self.rng.uniform(0, 2 * math.pi))
                          for j in range(2)]
            self.was_changing = False
            self.released = False
            self.ep += 1
            self._teleport()
            self.state = 'wait'

        def _teleport(self):
            x, y, yaw = self.target
            m = PoseWithCovarianceStamped()
            m.header.frame_id = 'map'
            m.header.stamp = self.get_clock().now().to_msg()
            m.pose.pose.position.x = x
            m.pose.pose.position.y = y
            m.pose.pose.orientation.z = math.sin(0.5 * yaw)
            m.pose.pose.orientation.w = math.cos(0.5 * yaw)
            self._pose_pub.publish(m)
            self.t_cmd = self._now()

        def _end(self):
            self._drive_cmd(0.0, 0.0)
            self.state = 'place'

        # ── main loop ──
        def _tick(self):
            if self.img is None or self.K is None or not self.odom:
                return
            now = self._now()
            _, x, y, yaw = self.odom[-1]
            if self.state == 'place':
                self._place()
                return
            if self.state == 'wait':
                tx, ty, tyaw = self.target
                dyaw = math.atan2(math.sin(yaw - tyaw), math.cos(yaw - tyaw))
                if math.hypot(x - tx, y - ty) < 0.01 and abs(dyaw) < 0.01:
                    self.state, self.t_start = 'settle', now
                elif now - self.t_cmd > 2.0:
                    self._teleport()
                return
            if self.state == 'settle':
                if now - self.t_start >= self.settle:
                    self.state, self.t_start = 'drive', now
                return

            T = self.T
            dt = 0.02
            t = now - self.t_start
            got = T.plan_label(x, y, yaw)
            if got is None:
                self._end()
                return
            pts, blocked, changing, lane = got
            c, s = math.cos(yaw), math.sin(yaw)
            rel = pts - (x, y)
            ego = np.column_stack([c * rel[:, 0] + s * rel[:, 1], -s * rel[:, 0] + c * rel[:, 1]])
            j = int(np.argmin(np.hypot(*rel.T)))
            seg = pts[min(j + 1, len(pts) - 1)] - pts[max(j - 1, 0)]
            seg = seg / max(float(np.linalg.norm(seg)), 1e-9)
            cte = float(-rel[j] @ np.array([-seg[1], seg[0]]))       # car vs plan, + = left
            here = T.lane_at(x, y, yaw)
            if here is None:                                   # facing the wrong way
                self._end()
                return
            k_lane, i_lane, _, w = here
            if changing and not self.was_changing:
                self.n_lc += 1
            self.was_changing = changing
            n = T.paths.shape[1]
            done = self.stop_i is not None and (i_lane - self.stop_i) % n < n // 2
            # late episode: hold this lane while the obstacle is more than `trigger`
            # ahead, then drive the late lane change -- the same late_label the
            # frames are labelled with. Off the plan on purpose, so no abort.
            late = None
            if self.kind == 'late':
                fwd = ego[:, 0] >= -0.2
                late = T.late_label(x, y, yaw, LATE_XS, np.interp(LATE_XS, ego[fwd, 0], ego[fwd, 1]))
            # wrong-side start: far off the plan on purpose while the expert steers back
            returning = self.kind == 'wrong' and t < self.wrong_s
            if (late is None and not returning and abs(cte) > self.abort * w) or t > self.episode_s or done \
                    or (self.kind == 'wrong' and t > self.wrong_end):     # the way back, not more lane keeping
                self._end()
                return
            # no lane to pass in: brake to a stop short of the obstacle (the model's
            # free-distance label is the same eval_tools.free_distance), hold, end
            free = T.free_distance(x, y, yaw) if blocked else FREE_MAX
            v_target = min(self.v_set, math.sqrt(2.0 * self.brake * max(0.0, free - self.stop_margin)))
            if blocked and self.v < 0.02:
                self.stopped_s += dt
                if self.stopped_s >= self.stop_hold:
                    self._end()
                    return
            if late is not None:
                d = [a * 0.05 for a, lat, jj in T.obstacles_along(k_lane, i_lane, int(8.0 / 0.05))
                     if lat < CAR_HALF_W + OBJ_HALF_W[T.kinds[jj]] + IN_PATH_MARGIN]
                self.released |= not d or d[0] <= self.trigger
                p = T.paths[k_lane][(i_lane + np.arange(int(6.0 / 0.05))) % n] - (x, y)
                own = np.column_stack([c * p[:, 0] + s * p[:, 1], -s * p[:, 0] + c * p[:, 1]])
                tgt = np.column_stack([LATE_XS, late]) if self.released else own
            else:
                # weave, faded out over ~1 s next to obstacles and during lane changes
                near = len(self.obs) > 0 and float(np.hypot(*(self.obs - (x, y)).T).min()) < self.calm_m
                calm = changing or near or blocked or returning
                self.weave_gain += float(np.clip((0.0 if calm else 1.0) - self.weave_gain, -dt, dt))
                off = self.weave_gain * sum(a * w * math.sin(2 * math.pi * t / Tp + ph)
                                            for a, Tp, ph in self.weave)
                tgt = ego[j:] + np.array([0.0, off])
            ld = float(np.clip(self.ld[0] + self.ld[1] * self.v, self.ld[0], self.ld[2]))
            st = pursuit_steer(tgt, ld, self.wheelbase)
            if st is None:
                self._end()
                return
            want = float(np.clip(st[0], -self.max_steer, self.max_steer))
            self.steer += float(np.clip(want - self.steer, -self.steer_rate * dt, self.steer_rate * dt))
            self.v += float(np.clip(v_target - self.v, -self.brake * dt, self.accel * dt))
            self._drive_cmd(self.v, self.steer)

            m = self.img
            t_img = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
            if (t_img == self.last_img_t or t_img - self.last_rec_t < self.rec_dt
                    or t < self.warmup):
                return
            self.last_img_t = t_img
            t_o, ox, oy, oyaw = min(self.odom, key=lambda o: abs(o[0] - t_img))
            if abs(t_o - t_img) > 0.015 or (late is None and not returning and abs(cte) > self.rec_max * w):
                return
            a_ = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1)[:, :, :3]
            if not m.encoding.lower().startswith('bgr'):
                a_ = a_[:, :, ::-1]
            name = f'{self.count:05d}.jpg'
            cv2.imwrite(os.path.join(self.dir, name), a_, [cv2.IMWRITE_JPEG_QUALITY, 95])
            self._csv.writerow([name, f'{ox:.5f}', f'{oy:.5f}', f'{oyaw:.6f}', lane, int(changing),
                                f'{t_img:.3f}', self.ep, self.kind, f'{self.v:.2f}', f'{self.steer:.4f}'])
            self._csv_file.flush()
            self.last_rec_t = t_img
            self.count += 1
            if self.count % 50 == 0:
                self.get_logger().info(f'{self.map}: {self.count}/{self.n} '
                                       f'(episode {self.ep}, lane changes {self.n_lc})')
            if self.count >= self.n:
                self._drive_cmd(0.0, 0.0)
                self._csv_file.close()
                self.get_logger().info(f'{self.map}: done, {self.ep + 1} episodes, '
                                       f'{self.n_lc} lane changes')
                raise SystemExit

    rclpy.init()
    node = RoadCollect()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main(args=None):
    _main_ros()


if __name__ == '__main__':
    main()
