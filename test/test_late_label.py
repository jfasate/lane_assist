"""Late lane change (eval_tools.Truth.late_label) -- offline, no ROS (~10 s).

  cd ~/sim_gazebo/src/lane_assist && /usr/bin/python3 -m pytest test/test_late_label.py -q

A kinematic car on road_69's cone closure (the one the 84-map model hit 4/4)
holds its lane like road_collect's LATE episode, then follows late_label with
the collector's pure pursuit and steering limits. Checks the label is defined
while holding, starts at the lane centre (same meaning as every label), ends
in the free lane, and that the car really clears the closure from every
trigger distance road_collect uses.
"""

import math
import os
import sys

import numpy as np

PKG = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, PKG)
from lane_assist.eval_tools import (Truth, car_polygon, CAR_HALF_W, OBJ_HALF_W,   # noqa: E402
                                    IN_PATH_MARGIN, RECOVER_CLEAR, RECOVER_MIN)
from lane_assist.lane_follow_node import pursuit_steer                           # noqa: E402

XS = np.linspace(0.0, 5.0, 26)
WB, MAX_STEER, STEER_RATE, DT, V = 0.3302, 0.4189, 1.0, 0.02, 1.5
T = Truth('road_69')
CLOSURE = [52, 53, 54]                       # the cone closure, lane 1 -> lane 0


def ego(p, x, y, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    d = p - (x, y)
    return np.column_stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]])


def plan_y(x, y, yaw):
    e = ego(T.plan_label(x, y, yaw, back=2.0, ahead=8.0)[0], x, y, yaw)
    f = e[:, 0] >= -0.2
    return np.interp(XS, e[f, 0], e[f, 1])


def obstacle_ahead(x, y, yaw):
    k, i = T.lane_at(x, y, yaw)[:2]
    d = [a * 0.05 for a, lat, j in T.obstacles_along(k, i, int(8.0 / 0.05))
         if lat < CAR_HALF_W + OBJ_HALF_W[T.kinds[j]] + IN_PATH_MARGIN]
    return d[0] if d else None


def drive(trigger, start_m=9.0):
    """-> (min gap to the closure, labels seen while holding, end lane)."""
    k, start, j, tgt = [m for m in T.maneuvers() if m[2] == CLOSURE[0]][0]
    a = int(np.argmin(np.hypot(*(T.paths[k] - T.obj[j][:2]).T)))
    x, y, yaw = T.pose_on_lane(k, (a - int(start_m / 0.05)) % len(T.paths[k]))
    steer, released, gap, held = 0.0, False, 9.9, []
    for _ in range(int(9.0 / DT)):
        late = T.late_label(x, y, yaw, XS, plan_y(x, y, yaw))
        if late is not None:
            d = obstacle_ahead(x, y, yaw)
            released |= d is None or d <= trigger
            if not released:
                held.append(late)
                kk, ii = T.lane_at(x, y, yaw)[:2]
                path = ego(T.paths[kk][(ii + np.arange(120)) % len(T.paths[kk])], x, y, yaw)
            else:
                path = np.column_stack([XS, late])
        else:
            path = np.column_stack([XS, plan_y(x, y, yaw)])
        want = float(np.clip(pursuit_steer(path, 0.8 + 0.5 * V, WB)[0], -MAX_STEER, MAX_STEER))
        steer += float(np.clip(want - steer, -STEER_RATE * DT, STEER_RATE * DT))
        x += V * math.cos(yaw) * DT
        y += V * math.sin(yaw) * DT
        yaw += V / WB * math.tan(steer) * DT
        gap = min(gap, min(car_polygon(x, y, yaw).distance(T.polys[c]) for c in CLOSURE))
    return gap, held, T.lane_at(x, y, yaw)[0], tgt


def test_late_label_shape():
    gap, held, end_lane, tgt = drive(trigger=3.0)
    assert len(held) > 20, 'late_label must be defined while the car holds its lane'
    first = np.array(held)
    assert np.all(np.abs(first[:, 0]) < 0.1), 'starts at the lane centre, like every label'
    lane_w = float(np.median(T.widths[1]))
    assert np.all(np.abs(first[:, -1]) > 0.5 * lane_w), 'ends in the free lane'
    assert end_lane == tgt


def test_too_close_has_no_label():
    k, start, j, tgt = [m for m in T.maneuvers() if m[2] == CLOSURE[0]][0]
    a = int(np.argmin(np.hypot(*(T.paths[k] - T.obj[j][:2]).T)))
    x, y, yaw = T.pose_on_lane(k, a - int((RECOVER_MIN + RECOVER_CLEAR - 0.3) / 0.05))
    assert T.late_label(x, y, yaw, XS, plan_y(x, y, yaw)) is None


def test_car_clears_closure_from_every_trigger():
    # road_collect late_trigger_min..max; the minimum must leave late_label
    # defined at release (it ends at RECOVER_MIN + RECOVER_CLEAR = 1.8 m)
    assert 2.2 > RECOVER_MIN + RECOVER_CLEAR
    for trigger in (2.2, 3.0, 4.5):
        gap, _, end_lane, tgt = drive(trigger)
        assert gap > 0.3, f'too close to the closure releasing at {trigger} m (gap {gap:.2f})'
        assert end_lane == tgt, f'released at {trigger} m but ended in lane {end_lane}, not {tgt}'
