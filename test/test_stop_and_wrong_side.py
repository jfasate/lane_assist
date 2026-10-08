"""Stop at a full road block, return from the wrong side, lidar guard -- offline, no ROS (~15 s).

  cd ~/sim_gazebo/src/lane_assist && /usr/bin/python3 -m pytest test/test_stop_and_wrong_side.py -q

A kinematic car runs road_collect's expert logic (pure pursuit on plan_label,
braking to sqrt(2 * brake_decel * (free - stop_margin)) with
eval_tools.free_distance) on the held-out road_test_stop map and on road_test:
  - it stops short of a full road block without touching it,
  - free_distance stays at FREE_MAX next to an obstacle that can be passed,
  - a car started in the oncoming lane is back on its own side within wrong_s,
  - lidar_guard's free distance behaves (its own self-check).
"""

import math
import os
import sys

import numpy as np

PKG = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, PKG)
from lane_assist.eval_tools import FREE_MAX, Truth, car_polygon                  # noqa: E402
from lane_assist.lane_follow_node import pursuit_steer                          # noqa: E402
from lane_assist import lidar_guard                                              # noqa: E402

WB, MAX_STEER, STEER_RATE, DT = 0.3302, 0.4189, 1.0, 0.02
BRAKE, MARGIN, ACCEL, V_SET = 1.0, 0.4, 1.0, 1.5          # road_collect defaults


def ego(p, x, y, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    d = p - (x, y)
    return np.column_stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]])


def drive(T, pose, seconds, v0=V_SET):
    """road_collect's expert: follow plan_label, brake on free_distance.
    -> list of (x, y, yaw, v, free)."""
    x, y, yaw = pose
    v, steer, out = v0, 0.0, []
    for _ in range(int(seconds / DT)):
        pts, blocked, _, _ = T.plan_label(x, y, yaw)
        free = T.free_distance(x, y, yaw) if blocked else FREE_MAX
        v_t = min(V_SET, math.sqrt(2 * BRAKE * max(0.0, free - MARGIN)))
        v += float(np.clip(v_t - v, -BRAKE * DT, ACCEL * DT))
        e = ego(pts, x, y, yaw)
        e = e[np.argmin(np.hypot(*e.T)):]
        st = pursuit_steer(e, 0.8 + 0.5 * v, WB)
        want = float(np.clip(st[0], -MAX_STEER, MAX_STEER)) if st else 0.0
        steer += float(np.clip(want - steer, -STEER_RATE * DT, STEER_RATE * DT))
        x += v * math.cos(yaw) * DT
        y += v * math.sin(yaw) * DT
        yaw += v / WB * math.tan(steer) * DT
        out.append((x, y, yaw, v, free))
    return out


def test_stops_before_full_block():
    T = Truth('road_test_stop')
    blocked = [(k, j) for k, _, j, tgt in T.maneuvers() if tgt < 0]
    assert blocked, 'road_test_stop must contain full road blocks'
    done = set()
    for k, j in blocked:
        a = int(np.argmin(np.hypot(*(T.paths[k] - T.obj[j][:2]).T)))
        if (k, a // 200) in done:
            continue                                   # one check per block
        done.add((k, a // 200))
        start = T.pose_on_lane(k, (a - int(9.0 / 0.05)) % len(T.paths[k]))
        run = drive(T, start, 14.0)
        x, y, yaw, v, free = run[-1]
        touch = min(car_polygon(x_, y_, a_).distance(p) for x_, y_, a_, *_ in run[::5] for p in T.polys)
        assert touch > 0.0, f'lane {k}: touched an obstacle'
        front = car_polygon(x, y, yaw).distance(T.polys[j])
        assert front > 0.1, f'lane {k}: stopped only {front:.2f} m short of the block'
        assert v < 0.02, f'lane {k}: not stopped (v {v:.2f})'
        assert free < 1.0, f'lane {k}: stopped {free:.2f} m short, too early'


def test_passable_obstacle_is_free():
    T = Truth('road_test')
    k, _, j, tgt = [m for m in T.maneuvers() if m[3] >= 0][0]
    a = int(np.argmin(np.hypot(*(T.paths[k] - T.obj[j][:2]).T)))
    for d in (6.0, 3.0):
        x, y, yaw = T.pose_on_lane(k, (a - int(d / 0.05)) % len(T.paths[k]))
        assert T.free_distance(x, y, yaw) == FREE_MAX, 'an obstacle with a free lane beside it is not a stop'


def test_returns_from_wrong_side():
    T = Truth('road_test')
    checked = 0
    for k in range(len(T.paths)):
        i = len(T.paths[k]) // 7 * (k + 1)
        w = float(T.widths[k][i])
        pose = T.pose_on_lane(k, i, 1.0 * w, 0.0)
        if not T.oncoming_at(*pose):
            continue
        run = drive(T, pose, 4.0)                      # road_collect wrong_s
        late = [T.oncoming_at(x, y, yaw) for x, y, yaw, *_ in run[-25:]]
        assert not any(late), f'lane {k}: still in the oncoming lane after 4 s'
        checked += 1
    assert checked >= 1, 'no two-way start found'


def test_right_pass_detection():
    """road_test section 10: the closure in the inner forward lane (next to the
    centre line) can only be passed on the right -- the case the 105-map model
    passed on the left, through the oncoming lane."""
    T = Truth('road_test')
    k, _, _, tgt = [m for m in T.maneuvers() if m[2] == 108][0]
    assert T.passes_right(k, tgt)
    assert not T.passes_right(tgt, k)


def test_lidar_guard():
    lidar_guard.selfcheck()
