"""Levels 2-4 -- closed-loop scenario tests in the running sim (~4 min).

  cd ~/sim_gazebo && source /opt/ros/humble/setup.bash && source install/setup.bash
  launch_test src/lane_assist/test/test_closed_loop.py
  LANE_TEST_MAP=road_05 launch_test src/lane_assist/test/test_closed_loop.py   # default map: road_test

launch_testing starts the sim (Gazebo + RViz windows, from sim.yaml) and
vit_follow_launch.py (ViT -> pure pursuit),
then each scenario teleports the car with /initialpose, lets it drive, and
scores the recording against ground truth (lane_assist.eval_tools):

  Level 2  reference line  rate, gaps, error vs the true lane centre at the
                           car and at 3 m, frame-to-frame jumps
  Level 3  tracking        car offset from its OWN reference vs from the TRUE
                           centre (separates control errors from perception),
                           lane departure
  Level 4  scenarios       lane keeping, recovery from an offset start, a
                           work zone with edge cones, a blocking obstacle that
                           must be passed in the lane the plan chose, and one
                           next to a bike / bus lane that must NOT be used,
                           and random starts where every obstacle the car meets
                           is scored (any collision fails); collision is the
                           car footprint touching an obstacle footprint

Only some tests:  LANE_TEST_ONLY=test_6 launch_test ...   (substring of the name)
test_6 options:   LANE_TEST_EPISODES=6 LANE_TEST_EPISODE_S=45 LANE_TEST_SEED=0 LANE_TEST_GUARD=true
                  test_6 fails on ANY footprint contact or > 0.5 s in an oncoming lane;
                  on a full road block (no lane to pass) stopping short of it is the pass.
Stop test:        LANE_TEST_MAP=road_test_stop (held out, full road blocks + shadowed debris)

Everything from one test session goes into ONE folder,
log/closed_loop_<map>_<stamp>/: <scenario>.csv per run (pose, true offset,
predicted path per sample) and report.txt with every scenario's metrics.
"""

import csv
import datetime
import math
import os
import sys
import time
import unittest

import numpy as np
import pytest

PKG = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, PKG)
from lane_assist.eval_tools import (Truth, encounters, run_metrics, CAR_HALF_W, OBJ_HALF_W,   # noqa: E402
                                   IN_PATH_MARGIN, LC_LEN, LC_CLEAR, STEP)
from lane_assist.vit_lane import XS                        # noqa: E402

MAP = os.environ.get('LANE_TEST_MAP', 'road_test')
LOG = os.path.join(PKG, 'log')

# Level 2 -- reference line
MIN_REF_RATE_HZ = 15.0
MAX_REF_GAP_S = 0.3
MAX_REF_ERR_CAR = 0.10        # mean |ref - true lane centre| at the car [m]
MAX_REF_ERR_3M = 0.20         # ... at 3 m
MAX_REF_JUMP_P95 = 0.05       # frame-to-frame jump of the path at 1.5 m [m]
# Level 3 -- tracking
MAX_TRUE_OFFSET_RMS = 0.10    # car vs true lane centre [m]
MAX_DEPARTURE_FRAC = 0.02     # share of time a wheel is over the lane line
# Level 4 -- scenarios
MIN_CLEARANCE = 0.05          # car footprint to any obstacle [m]
MAX_STOPPED_FRAC = 0.05
RECOVERY_OFFSET = 0.06        # mean |offset| over the last 3 s of recovery [m]
MAX_PROTECTED_M = 0.05        # deepest the car may go into a bike / bus / parking lane [m]
# test_6: random starts, the car drives on its own; every obstacle it meets
# in its lane is scored, any collision fails. Set from the environment.
RANDOM_EPISODES = int(os.environ.get('LANE_TEST_EPISODES', '6'))
RANDOM_EPISODE_S = float(os.environ.get('LANE_TEST_EPISODE_S', '45'))
RANDOM_SEED = int(os.environ.get('LANE_TEST_SEED', '0'))
# lidar emergency brake on (default) or off -- off measures the camera model alone
GUARD = os.environ.get('LANE_TEST_GUARD', 'true')
MAX_ONCOMING_S = 0.5          # test_6 fails above this much time on the wrong side of the road


@pytest.mark.launch_test
def generate_test_description():
    from ament_index_python.packages import get_package_share_directory
    from launch import LaunchDescription
    from launch.actions import IncludeLaunchDescription, TimerAction
    from launch.launch_description_sources import PythonLaunchDescriptionSource
    import launch_testing.actions

    sim = os.path.join(get_package_share_directory('f1tenth_gym_gazebo'), 'launch', 'sim_launch.py')
    vit = os.path.join(get_package_share_directory('lane_assist'), 'launch', 'vit_follow_launch.py')
    return LaunchDescription([
        IncludeLaunchDescription(PythonLaunchDescriptionSource(sim),
                                 launch_arguments={'map_name': MAP}.items()),
        TimerAction(period=20.0, actions=[IncludeLaunchDescription(
            PythonLaunchDescriptionSource(vit), launch_arguments={'lidar_guard': GUARD}.items())]),
        launch_testing.actions.ReadyToTest(),
    ])


class TestClosedLoop(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
        from nav_msgs.msg import Odometry
        from std_msgs.msg import Float64MultiArray
        from ackermann_msgs.msg import AckermannDriveStamped
        from geometry_msgs.msg import PoseWithCovarianceStamped

        rclpy.init()
        cls.rclpy = rclpy
        cls.node = Node('closed_loop_test', parameter_overrides=[])
        cls.truth = Truth(MAP)
        cls.st = dict(odom=None, ref=None, ref_t=None, v_cmd=0.0)

        def odom(m):
            q = m.pose.pose.orientation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
            cls.st['odom'] = (t, m.pose.pose.position.x, m.pose.pose.position.y, yaw)

        def ref(m):
            a = np.asarray(m.data, float).reshape(-1, 6)
            cls.st['ref'] = np.interp(XS, a[:, 1], a[:, 2])
            cls.st['ref_t'] = cls.st['odom'][0] if cls.st['odom'] else 0.0

        def drive(m):
            cls.st['v_cmd'] = float(m.drive.speed)

        cls.node.create_subscription(Odometry, '/ego_racecar/odom', odom,
                                     QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        cls.node.create_subscription(
            Float64MultiArray, '/planning/ref_path', ref,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                       durability=DurabilityPolicy.TRANSIENT_LOCAL))
        cls.node.create_subscription(AckermannDriveStamped, '/drive', drive, 10)
        cls.pose_pub = cls.node.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
        cls.Pose = PoseWithCovarianceStamped
        t0 = time.time()
        while (cls.st['odom'] is None or cls.st['ref'] is None) and time.time() - t0 < 180:
            rclpy.spin_once(cls.node, timeout_sec=0.1)
        if cls.st['ref'] is None:
            raise RuntimeError('no /planning/ref_path after 180 s: is vit_lane_node running?')
        cls.stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        cls.dir = os.path.join(LOG, f'closed_loop_{MAP}_{cls.stamp}')
        os.makedirs(cls.dir, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        cls.node.destroy_node()
        cls.rclpy.shutdown()

    def setUp(self):
        only = os.environ.get('LANE_TEST_ONLY')
        if only and only not in self._testMethodName:
            self.skipTest(f'LANE_TEST_ONLY={only}')

    # ── helpers ──
    def _teleport(self, x, y, yaw):
        m = self.Pose()
        m.header.frame_id = 'map'
        m.pose.pose.position.x, m.pose.pose.position.y = x, y
        m.pose.pose.orientation.z, m.pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        t0 = time.time()
        while time.time() - t0 < 10:
            self.pose_pub.publish(m)
            for _ in range(5):
                self.rclpy.spin_once(self.node, timeout_sec=0.02)
            _, ox, oy, _ = self.st['odom']
            if math.hypot(ox - x, oy - y) < 0.05:
                return
        self.fail('teleport not confirmed by odom')

    def _drive(self, name, pose, seconds, settle=1.0):
        """Teleport, let it drive for `seconds` of sim time, return metrics."""
        self._teleport(*pose)
        rec = {k: [] for k in ('t', 'x', 'y', 'yaw', 'v_cmd', 'ref_t', 'ref')}
        t_start = self.st['odom'][0]
        last = -1.0
        while True:
            self.rclpy.spin_once(self.node, timeout_sec=0.02)
            t, x, y, yaw = self.st['odom']
            if t - t_start > settle + seconds:
                break
            if t - t_start < settle or t - last < 0.05:
                continue
            last = t
            for k, v in zip(('t', 'x', 'y', 'yaw', 'v_cmd', 'ref_t', 'ref'),
                            (t, x, y, yaw, self.st['v_cmd'], self.st['ref_t'], self.st['ref'].copy())):
                rec[k].append(v)
        m = run_metrics(self.truth, rec, XS)
        self._last_rec = rec
        self._save(name, rec, m)
        return m

    def _save(self, name, rec, m):
        path = os.path.join(self.dir, f'{name}.csv')
        with open(path, 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['t', 'x', 'y', 'yaw', 'v_cmd', 'true_offset'] + [f'ref_y{j:02d}' for j in range(len(XS))])
            for i in range(len(rec['t'])):
                w.writerow([f'{rec[k][i]:.4f}' for k in ('t', 'x', 'y', 'yaw', 'v_cmd')]
                           + [f'{m["cte"][i]:.4f}'] + [f'{v:.4f}' for v in rec['ref'][i]])
        lines = [f'{MAP} {name} ({self.stamp})'] + [
            f'  {k:20s} {v:.3f}' if isinstance(v, float) else f'  {k:20s} {v}'
            for k, v in m.items() if not isinstance(v, np.ndarray)]
        with open(os.path.join(self.dir, 'report.txt'), 'a') as fh:
            fh.write('\n'.join(lines) + '\n\n')
        print('\n' + '\n'.join(lines) + f'\n  run log: {path}', file=sys.stderr, flush=True)

    def _check_ref(self, m):
        self.assertGreaterEqual(m['ref_rate_hz'], MIN_REF_RATE_HZ, 'reference published too slowly')
        self.assertLess(m['ref_max_gap_s'], MAX_REF_GAP_S, 'reference went stale')
        self.assertLess(m['ref_err_car_mean'], MAX_REF_ERR_CAR, 'reference off the lane centre at the car')
        self.assertLess(m['ref_err_3m_mean'], MAX_REF_ERR_3M, 'reference off the lane centre at 3 m')
        self.assertLess(m['ref_jump_p95'], MAX_REF_JUMP_P95, 'reference jumps between frames')

    # ── scenarios (run in name order) ──
    def test_1_lane_keeping(self):
        k, i = self.truth.find_straight(clear_m=15.0)
        m = self._drive('lane_keeping', self.truth.pose_on_lane(k, i), seconds=30.0)
        self._check_ref(m)
        self.assertFalse(m['collision'], 'collided with an obstacle')
        self.assertLess(m['true_offset_rms'], MAX_TRUE_OFFSET_RMS, 'not centred in the lane')
        self.assertLess(m['departure_frac'], MAX_DEPARTURE_FRAC, 'left the lane')
        self.assertLess(m['stopped_frac'], MAX_STOPPED_FRAC, 'car stopped')
        self.assertGreater(m['distance_m'], 20.0, 'too little progress')

    def test_2_recovery_from_offset(self):
        k, i = self.truth.find_straight(clear_m=15.0)
        w = float(self.truth.widths[k][i])
        m = self._drive('recovery', self.truth.pose_on_lane(k, i, lat=0.30 * w, dyaw=0.26), seconds=8.0,
                        settle=0.2)
        tail = np.abs(m['cte'][-int(3.0 / 0.05):])
        print(f'  last-3s mean |offset| {tail.mean():.3f} m', file=sys.stderr)
        self.assertFalse(m['collision'], 'collided with an obstacle')
        self.assertLess(m['departure_frac'], MAX_DEPARTURE_FRAC, 'left the lane while recovering')
        self.assertLess(tail.mean(), RECOVERY_OFFSET, 'did not return to the lane centre')

    def test_3_work_zone_edge_cones(self):
        found = self.truth.find_edge_cones()
        if found is None:
            self.skipTest(f'{MAP} has no edge-cone stretch')
        k, i0, i1 = found
        length = ((i1 - i0) % self.truth.paths.shape[1]) * 0.05
        m = self._drive('edge_cones', self.truth.pose_on_lane(k, i0), seconds=length / 1.2 + 3.0)
        self.assertFalse(m['collision'], 'hit a cone/barrel')
        self.assertGreater(m['min_clearance_m'], MIN_CLEARANCE, 'passed too close to a cone')
        self.assertLess(m['departure_frac'], MAX_DEPARTURE_FRAC, 'left the lane in the work zone')
        self.assertEqual(m['lane_end'], m['lane_start'], 'changed lane for edge cones')

    def _pass_obstacle(self, name, protected_side):
        found = self.truth.find_maneuver(protected_side=protected_side)
        if found is None:
            self.skipTest(f'{MAP} has no {"bike/bus-lane " if protected_side else ""}passable obstacle')
        k, i, j = found
        target = next(t for kk, _, jj, t in self.truth.maneuvers() if kk == k and jj == j)
        m = self._drive(name, self.truth.pose_on_lane(k, i), seconds=10.0)
        self.assertFalse(m['collision'],
                         f'collided with the {self.truth.kinds[j]} (min clearance {m["min_clearance_m"]:.2f} m)')
        self.assertEqual(m['lane_end'], target,
                         f'ended in lane {m["lane_end"]}, the plan passes in lane {target}')
        self.assertGreater(m['distance_m'], 10.0, 'did not get past the obstacle')
        self.assertLess(m['protected_max_m'], MAX_PROTECTED_M,
                        f'went {m["protected_max_m"]:.2f} m into a bike/bus/parking lane')
        return m

    def test_4_blocking_obstacle(self):
        self._pass_obstacle('blocking_obstacle', protected_side=False)

    def test_6_random_starts_avoid_everything(self):
        """No placement behind obstacles: start anywhere, drive, and score
        every obstacle the car actually meets in its lane. Any collision
        fails. Never-met passable obstacles are reported (coverage), not
        failed -- the next random seed may reach them."""
        T = self.truth
        rng = np.random.default_rng(RANDOM_SEED)
        obs = np.array([p[:2] for p, k in zip(T.obj, T.kinds) if k not in ('barrier', 'bus')]).reshape(-1, 2)
        all_ev, lines, oncoming_s = [], [], 0.0
        impassable = {j for *_, j, tgt in T.maneuvers() if tgt < 0}
        for ep in range(RANDOM_EPISODES):
            for _ in range(200):                       # a random start clear of obstacles
                k = int(rng.integers(len(T.paths)))
                ok = np.flatnonzero(T.exists[k])
                i = int(rng.choice(ok))
                p = T.paths[k][i]
                # room to react: nothing in this lane's path for a full lane change
                # + clearance + 1 m (a start 1 m behind a stopped car is a crash
                # no driver could avoid, not a test of the model)
                ahead = T.obstacles_along(k, i, int((LC_LEN + LC_CLEAR + 1.0) / STEP))
                blocked = any(lat < CAR_HALF_W + OBJ_HALF_W[T.kinds[j]] + IN_PATH_MARGIN for _, lat, j in ahead)
                if not blocked and (not len(obs) or np.hypot(*(obs - p).T).min() > 1.5):
                    break
            w = float(T.widths[k][i])
            pose = T.pose_on_lane(k, i, rng.uniform(-0.2, 0.2) * w, rng.uniform(-0.15, 0.15))
            m = self._drive(f'random_{ep}', pose, seconds=RANDOM_EPISODE_S)
            rec = {k_: getattr(self, '_last_rec')[k_] for k_ in ('t', 'x', 'y', 'yaw')}
            ev = encounters(T, rec)
            # ANY footprint contact is a collision -- encounters() only looks at
            # objects in the lane's path and missed the first cones of a closure taper
            touched = {}
            for tt, xx, yy, aa in zip(rec['t'], rec['x'], rec['y'], rec['yaw']):
                g, j = T.clearance(xx, yy, aa)
                if g <= 0.0:
                    touched.setdefault(j, tt)
            for e in ev:
                if e['obstacle'] in touched:
                    e['outcome'] = 'collision'
                elif e['obstacle'] in impassable:
                    # no lane to pass in: STOPPING short of it is the pass; getting
                    # past it (around it through the oncoming lane) is not
                    ox, oy = T.obj[e['obstacle']][:2]
                    ahead = [math.cos(aa) * (ox - xx) + math.sin(aa) * (oy - yy)
                             for tt, xx, yy, aa in zip(rec['t'], rec['x'], rec['y'], rec['yaw']) if tt >= e['t']]
                    e['outcome'] = 'evaded' if min(ahead) < -0.5 else 'stopped'
            ev += [dict(obstacle=j, kind=T.kinds[j], t=float(tt), seen_m=0.0, min_gap_m=0.0, outcome='collision')
                   for j, tt in touched.items() if j not in {e['obstacle'] for e in ev}]
            onc = float(sum(T.oncoming_at(xx, yy, aa) for xx, yy, aa in zip(rec['x'], rec['y'], rec['yaw']))
                        * np.median(np.diff(rec['t'])))
            oncoming_s += onc
            all_ev += ev
            lines.append(f'  episode {ep}: start lane {k}, {m["distance_m"]:.0f} m, '
                         + (f'ONCOMING LANE {onc:.1f} s, ' if onc > 0 else '')
                         + (', '.join(f'{e["kind"]}#{e["obstacle"]} {e["outcome"]} (gap {e["min_gap_m"]:.2f})'
                                      for e in ev) or 'no obstacle met'))
        passable = {j for *_, j, tgt in T.maneuvers() if tgt >= 0}
        met = {e['obstacle'] for e in all_ev}
        crashes = [e for e in all_ev if e['outcome'] in ('collision', 'evaded')]
        by = {}
        for e in all_ev:
            by.setdefault(e['kind'], [0, 0, 0, 0, 0])[('avoided', 'collision', 'incomplete', 'stopped', 'evaded')
                                                      .index(e['outcome'])] += 1
        report = (['', f'{MAP} random starts: {RANDOM_EPISODES} episodes x {RANDOM_EPISODE_S:.0f} s (seed {RANDOM_SEED})']
                  + lines
                  + [f'  {kind:7s} avoided {a}  COLLISION {c}  incomplete {n}  stopped {st}  EVADED {ev}'
                     for kind, (a, c, n, st, ev) in sorted(by.items())]
                  + [f'  time in an oncoming lane: {oncoming_s:.1f} s (fail above {MAX_ONCOMING_S} s)',
                     f'  lidar guard: {GUARD}']
                  + [f'  coverage: met {len(met & passable)} of {len(passable)} passable obstacles; never met '
                     f'{sorted(passable - met)}'])
        print('\n'.join(report), file=sys.stderr, flush=True)
        with open(os.path.join(self.dir, 'report.txt'), 'a') as fh:
            fh.write('\n'.join(report) + '\n\n')
        self.assertLessEqual(oncoming_s, MAX_ONCOMING_S, f'{oncoming_s:.1f} s in an oncoming lane')
        if not all_ev:
            self.skipTest('the car met no obstacle in its lane: nothing to judge (try more/longer episodes)')
        self.assertFalse(crashes, f'{len(crashes)} collision(s): ' + ', '.join(
            f'{e["kind"]}#{e["obstacle"]} at t={e["t"]:.1f}' for e in crashes))

    def test_5_obstacle_next_to_bike_bus_lane(self):
        """The obstacle blocks the lane beside a bike / bus lane: the empty
        space is right there, but the car must pass in the other travel lane."""
        self._pass_obstacle('bike_bus_lane_temptation', protected_side=True)
