#!/usr/bin/env python3
"""Ground-truth scoring for the ViT lane stack (ROS-free).

Used by test/test_closed_loop.py to turn a recorded drive into numbers:

  Truth(map)          lanes, widths, directions and obstacle footprints from
                      tools/maps/truth/<map>.npz
  .lane_at(x,y,yaw)   which lane the car is in (same travel direction) and its
                      signed offset from that lane's true centre
  .clearance(x,y,yaw) gap between the car footprint and the nearest obstacle
                      footprint [m], 0 = collision
  .plan_label(x,y,yaw) THE expert path from the car's pose: lane centre, or a
                      smooth lane change around an obstacle in the lane. Used
                      by the collector (drives it), make_dataset (labels) and
                      the analyser / tests (what the car should have done)
  .protected_at(...)  is the car footprint in a bike / bus / parking lane or
                      on the shoulder
  scenario finders    blocking obstacle / edge-cone stretch / straight
  run_metrics(...)    reference-line quality, tracking, lane departure,
                      collision, progress for one recorded run

Footprints: the car is the URDF chassis plus wheels, base_link at the rear
axle: x in [-0.06, 0.39], |y| <= 0.14. Obstacles use the sizes the generator
gives them in Gazebo (cars/barriers boxes, cones/barrels circles).
"""

import math
import os

import numpy as np
from shapely.geometry import LineString, Point, Polygon, box
from shapely import affinity

HERE = os.path.dirname(os.path.abspath(__file__))
TRUTH_DIR = os.path.normpath(os.path.join(HERE, '..', 'tools', 'maps', 'truth'))
CAR_X = (-0.06, 0.39)
CAR_HALF_W = 0.14
OBJ_BOX = {'car': (0.45, 0.20), 'barrier': (0.98, 0.07), 'debris': (0.25, 0.25),
           'bus': (1.0, 0.25)}
OBJ_RADIUS = {'cone': 0.03, 'barrel': 0.045}
OBJ_HALF_W = {'car': 0.10, 'barrier': 0.035, 'debris': 0.125, 'bus': 0.125,
              'cone': 0.03, 'barrel': 0.045}
STEP = 0.05                         # truth path spacing [m]
# The expert manoeuvre around an obstacle in the car's path: a smooth lane
# change LC_LEN long that is complete LC_CLEAR before the obstacle, i.e. it
# starts ~7 m before it. Same rule for the collector, the labels and the tests.
LC_LEN = 5.5
LC_CLEAR = 1.5
IN_PATH_MARGIN = 0.05               # obstacle counts as in the path if it would pass closer
ENDED_LANE_W = 0.10                # a lane narrower than this has ended (plan_label ignores it)
# Stopping: free distance = how far the car can still drive along its expert
# path before an obstacle it cannot get around (front bumper to the obstacle's
# near edge); FREE_MAX when nothing blocks it within that. The model's 17th
# output and the collector's braking target.
FREE_MAX = 8.0
# A LATE lane change, for a car still in its lane where the plan has already
# left it (the state a model that hesitates drives itself into): from this
# lane's centre into the plan's lane right now, complete RECOVER_CLEAR before
# the obstacle; closer than RECOVER_MIN + RECOVER_CLEAR there is no label.
RECOVER_CLEAR = 0.6
RECOVER_MIN = 1.2
PROTECTED = ('bike', 'bus', 'parking', 'shoulder')


def car_polygon(x, y, yaw):
    p = box(CAR_X[0], -CAR_HALF_W, CAR_X[1], CAR_HALF_W)
    p = affinity.rotate(p, yaw, origin=(0, 0), use_radians=True)
    return affinity.translate(p, x, y)


def _normals(p):
    d = np.roll(p, -1, 0) - np.roll(p, 1, 0)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return np.column_stack([-d[:, 1], d[:, 0]])


class Truth:
    def __init__(self, name, truth_dir=TRUTH_DIR):
        d = np.load(os.path.join(truth_dir, name + '.npz'))
        self.name = name
        self.paths = d['paths']
        self.exists = d['exists']
        self.widths = d['widths']
        self.dirs = d['dirs']
        self.nrm = [_normals(p) for p in self.paths]
        self.kinds = list(d['obj_kind'])
        self.obj = d['obj_pose']
        self.side = None
        if 'ref' in d.files:                         # side lanes (bike/bus/parking/shoulder)
            self.side = dict(ref=d['ref'], nrm=d['ref_nrm'], f=d['side_f'], f_type=d['side_f_type'],
                             b=d['side_b'] if d['side_b'].size else None,
                             b_type=d['side_b_type'] if d['side_b_type'].size else None)
        self.polys = []
        for k, (x, y, yaw) in zip(self.kinds, self.obj):
            if k in OBJ_BOX:
                L, W = OBJ_BOX[k]
                p = affinity.rotate(box(-L / 2, -W / 2, L / 2, W / 2), yaw, origin=(0, 0),
                                    use_radians=True)
                self.polys.append(affinity.translate(p, x, y))
            else:
                self.polys.append(Point(x, y).buffer(OBJ_RADIUS[k]))

    # ── where is the car ──
    def lane_at(self, x, y, yaw):
        """(lane, index, signed offset [m, + = left], lane width) of the
        nearest lane centre travelling the car's way."""
        best = None
        h = np.array([math.cos(yaw), math.sin(yaw)])
        for k, p in enumerate(self.paths):
            i = int(np.argmin(np.hypot(p[:, 0] - x, p[:, 1] - y)))
            t = np.array([self.nrm[k][i][1], -self.nrm[k][i][0]])
            if t @ h < 0:
                continue
            cte = float((np.array([x, y]) - p[i]) @ self.nrm[k][i])
            if best is None or abs(cte) < abs(best[2]):
                best = (k, i, cte, float(self.widths[k][i]))
        return best

    def pose_on_lane(self, k, i, lat=0.0, dyaw=0.0):
        p, n = self.paths[k][i], self.nrm[k][i]
        return (float(p[0] + lat * n[0]), float(p[1] + lat * n[1]),
                math.atan2(-n[0], n[1]) + dyaw)

    def clearance(self, x, y, yaw):
        """(gap [m] to the nearest obstacle, its index); gap 0 = collision."""
        if not self.polys:
            return float('inf'), -1
        c = np.hypot(self.obj[:, 0] - x, self.obj[:, 1] - y)
        near = np.flatnonzero(c < 2.0)               # exact geometry only where it matters
        if near.size == 0:
            j = int(np.argmin(c))
            return float(c[j] - 1.0), j               # lower bound, > 1 m anyway
        car = car_polygon(x, y, yaw)
        d = [car.distance(self.polys[j]) for j in near]
        j = int(np.argmin(d))
        return float(d[j]), int(near[j])

    # ── the expert plan: lane centre, or a lane change around an obstacle ──
    def _plan_tables(self):
        """Per lane: the manoeuvre (if any) active at each index.
        man[k][i] = (start, end, target lane or -1 = no free lane, obstacle,
        obstacle index on the lane)."""
        if hasattr(self, '_man'):
            return self._man
        n = self.paths.shape[1]
        L, C = int(LC_LEN / STEP), int(LC_CLEAR / STEP)
        blockers = []
        for k in range(len(self.paths)):
            bl = [(a, j) for a, lat, j in self.obstacles_along(k, 0, n)
                  if self.exists[k][a] and lat < CAR_HALF_W + OBJ_HALF_W[self.kinds[j]] + IN_PATH_MARGIN]
            blockers.append(sorted(bl))
        self._man = []
        for k in range(len(self.paths)):
            same = [q for q in range(len(self.paths)) if self.dirs[q] == self.dirs[k]]
            pos = same.index(k)
            nbrs = [same[q] for q in (pos - 1, pos + 1) if 0 <= q < len(same)]   # left first
            tab = [None] * n
            for a, j in blockers[k]:
                end, start = a - C, a - C - L
                target = -1
                win = np.arange(start - 40, a + int(6 / STEP)) % n
                for q in nbrs:
                    clear = not np.isin([b for b, _ in blockers[q]], win).any()
                    # directly adjacent only: a gore (the chevron no-drive area
                    # of a lane split) between the two lanes must not be crossed
                    gap = np.hypot(*(self.paths[q][win] - self.paths[k][win]).T)
                    side_by_side = np.all(gap < 0.5 * (self.widths[q][win] + self.widths[k][win]) + 0.15)
                    if self.exists[q][win].all() and clear and side_by_side:
                        target = q
                        break
                for i in range(start, a + 1):            # until the obstacle itself
                    # overlapping windows (two obstacles close in one lane): the
                    # NEAREST obstacle ahead decides; a car that already moved
                    # over for it never consults this lane's next one
                    old = tab[i % n]
                    if old is None or (a - i) % n < (old[4] - i) % n:
                        tab[i % n] = (start % n, end % n, target, j, a % n)
            self._man.append(tab)
        return self._man

    def plan_path(self, k, i0, n_s):
        """World points of the expert plan starting in lane k at index i0, for
        n_s samples. -> (pts, blocked, changing): blocked = runs into an
        obstacle with no free lane; changing = a lane change inside the window."""
        man = self._plan_tables()
        n = self.paths.shape[1]
        c, pts, blocked, changing = k, [], False, False
        for s_ in range(n_s):
            i = (i0 + s_) % n
            m = man[c][i]
            if m is None:
                pts.append(self.paths[c][i])
                continue
            start, end, target = m[:3]
            if target < 0:
                blocked = True
                pts.append(self.paths[c][i])
                continue
            span = (end - start) % n
            u = min(1.0, ((i - start) % n) / max(span, 1))
            u = u * u * (3 - 2 * u)
            pts.append((1 - u) * self.paths[c][i] + u * self.paths[target][i])
            changing = True
            if (i - start) % n >= span:
                c = target                                # change done: now in that lane
        return np.array(pts), blocked, changing

    def plan_label(self, x, y, yaw, back=2.0, ahead=6.0):
        """The expert path for a car at (x, y, yaw): for every lane travelling
        its way, simulate the plan from 2 m behind the car and keep the one
        that passes closest to the car -- mid lane change that is the change,
        not the lane it is leaving or entering.
        -> (pts (n,2) world, blocked, changing, start lane) or None.

        A lane that has ENDED (width ~0) does not compete: its truth path runs
        along the lane it merged into, but its plan sees no obstacles (they
        only count where a lane exists) -- so beside a closure that phantom
        path, sitting exactly on the closed lane's centre, would win as
        "closest" and drive the expert straight through the cones. A lane that
        is still narrowing keeps competing: its blended path is the merge."""
        h = np.array([math.cos(yaw), math.sin(yaw)])
        nb, na = int(back / STEP), int(ahead / STEP)
        cand = []
        for k, p in enumerate(self.paths):
            i = int(np.argmin(np.hypot(p[:, 0] - x, p[:, 1] - y)))
            t = np.array([self.nrm[k][i][1], -self.nrm[k][i][0]])
            if t @ h >= 0:
                cand.append((k, i))
        real = [(k, i) for k, i in cand if self.widths[k][i] > ENDED_LANE_W]
        best = None
        for k, i in real or cand:
            pts, blocked, changing = self.plan_path(k, i - nb, nb + na)
            d = float(np.hypot(*(pts - (x, y)).T).min())
            if best is None or d < best[0]:
                best = (d, pts, blocked, changing, k)
        return None if best is None else best[1:]

    def free_distance(self, x, y, yaw):
        """Metres the car can drive along its expert path before an obstacle
        with no free lane beside it (front bumper to the obstacle's near edge),
        clipped to 0..FREE_MAX; FREE_MAX when nothing like that is ahead."""
        got = self.plan_label(x, y, yaw, back=0.0, ahead=FREE_MAX + 1.0)
        if got is None:
            return FREE_MAX
        k = got[3]
        p = self.paths[k]
        i = int(np.argmin(np.hypot(p[:, 0] - x, p[:, 1] - y)))
        man = self._plan_tables()[k]
        n = len(p)
        for s in range(int((FREE_MAX + 2.0) / STEP)):
            m = man[(i + s) % n]
            if m is not None and m[2] < 0:               # a manoeuvre with no free lane
                j, a = m[3], m[4]
                kind = self.kinds[j]
                half = OBJ_BOX[kind][0] / 2 if kind in OBJ_BOX else OBJ_RADIUS[kind]
                return float(np.clip(((a - i) % n) * STEP - CAR_X[1] - half, 0.0, FREE_MAX))
        return FREE_MAX

    def passes_right(self, k, target):
        """True when a lane change from lane k to `target` goes to the right
        (lanes of one direction are ordered left to right)."""
        same = [q for q in range(len(self.paths)) if self.dirs[q] == self.dirs[k]]
        return same.index(target) > same.index(k)

    def oncoming_at(self, x, y, yaw):
        """True when the car centre is nearest to a lane travelling the other
        way (wrong side of the road)."""
        best = None
        for k, p in enumerate(self.paths):
            i = int(np.argmin(np.hypot(p[:, 0] - x, p[:, 1] - y)))
            if self.widths[k][i] <= ENDED_LANE_W:
                continue
            d = float(np.hypot(p[i, 0] - x, p[i, 1] - y))
            if best is None or d < best[0]:
                best = (d, k, i)
        if best is None:
            return False
        _, k, i = best
        t = np.array([self.nrm[k][i][1], -self.nrm[k][i][0]])
        return bool(t @ np.array([math.cos(yaw), math.sin(yaw)]) < 0)

    def late_label(self, x, y, yaw, xs, plan_y):
        """Path Y at xs (car frame) for a car still in its own lane with an
        obstacle in it close ahead: this lane's centre blending (smoothstep)
        into plan_y -- the plan_label path, already in the free lane -- over
        min(LC_LEN, d - RECOVER_CLEAR), d = distance to the obstacle. Same
        meaning as every label (where the path is, not where the car is).
        None if nothing blocks this lane within 8 m or d is too short."""
        here = self.lane_at(x, y, yaw)
        if here is None:
            return None
        k, i = here[0], here[1]
        d = [a * STEP for a, lat, j in self.obstacles_along(k, i, int(8.0 / STEP))
             if lat < CAR_HALF_W + OBJ_HALF_W[self.kinds[j]] + IN_PATH_MARGIN]
        L = min(LC_LEN, d[0] - RECOVER_CLEAR) if d else 0.0
        if L < RECOVER_MIN:
            return None
        p = self.paths[k][(i + np.arange(-int(1.0 / STEP), int((xs[-1] + 1.0) / STEP))) % len(self.paths[k])]
        c, s = math.cos(yaw), math.sin(yaw)
        X = c * (p[:, 0] - x) + s * (p[:, 1] - y)
        Y = -s * (p[:, 0] - x) + c * (p[:, 1] - y)
        if not np.all(np.diff(X) > 0):
            return None
        own = np.interp(xs, X, Y)
        u = np.clip(np.asarray(xs) / L, 0.0, 1.0)
        return own + (np.asarray(plan_y) - own) * u * u * (3 - 2 * u)

    def maneuvers(self):
        """[(lane, start index, obstacle index, target lane or -1)] one per
        obstacle in a lane's path."""
        out = []
        for k, tab in enumerate(self._plan_tables()):
            for m in dict.fromkeys(m for m in tab if m is not None):
                out.append((k, m[0], m[3], m[2]))       # (lane, start, obstacle, target)
        return out

    def protected_at(self, x, y, yaw):
        """(side-lane type, how far into it [m]) if the car footprint is in a
        bike / bus / parking lane or on the shoulder, else (None, 0)."""
        if self.side is None:
            return None, 0.0
        sd = self.side
        i = int(np.argmin(np.hypot(sd['ref'][:, 0] - x, sd['ref'][:, 1] - y)))
        lat = float((np.array([x, y]) - sd['ref'][i]) @ sd['nrm'][i])
        inner_f = sd['f'][0][i]
        if lat - CAR_HALF_W < inner_f and sd['f_type'][i] in PROTECTED:
            return str(sd['f_type'][i]), float(inner_f - (lat - CAR_HALF_W))
        if sd['b'] is not None:
            inner_b = sd['b'][0][i]
            if lat + CAR_HALF_W > inner_b and sd['b_type'][i] in PROTECTED:
                return str(sd['b_type'][i]), float(lat + CAR_HALF_W - inner_b)
        return None, 0.0

    # ── scenario finders ──
    def obstacles_along(self, k, i0, n):
        """[(samples ahead, |lateral offset|, obstacle index)] for obstacles
        within 1 m of lane k's centre over n samples from i0 (no barriers)."""
        p = self.paths[k]
        idx = (i0 + np.arange(n)) % len(p)
        out = []
        for j, (kind, (x, y, _)) in enumerate(zip(self.kinds, self.obj)):
            if kind == 'barrier':
                continue
            d = np.hypot(p[idx, 0] - x, p[idx, 1] - y)
            a = int(np.argmin(d))
            if d[a] < 1.0:
                lat = abs((np.array([x, y]) - p[idx[a]]) @ self.nrm[k][idx[a]])
                out.append((a, float(lat), j))
        return sorted(out)

    def find_maneuver(self, protected_side=False):
        """(lane, index 7 m before the change starts, obstacle) of a passable
        obstacle; with protected_side=True one in the outermost lane next to a
        bike / bus lane -- the case where the empty-looking lane is forbidden."""
        n = self.paths.shape[1]
        for k, start, j, tgt in self.maneuvers():
            if tgt < 0:
                continue
            if protected_side:
                same = [q for q in range(len(self.paths)) if self.dirs[q] == self.dirs[k]]
                if self.side is None or k != same[-1]:
                    continue
                a = int(np.argmin(np.hypot(*(self.paths[k] - self.obj[j][:2]).T)))
                r = int(np.argmin(np.hypot(*(self.side['ref'] - self.paths[k][a]).T)))
                typ = self.side['f_type'] if self.dirs[k] > 0 else self.side['b_type']
                if typ is None or typ[r] not in ('bike', 'bus'):
                    continue
            i = (start - int(3.0 / STEP)) % n
            if self.exists[k][i]:
                return k, i, j
        return None

    def find_blocking(self, corridor=0.30):
        """(lane, index 5 m before, obstacle) for an obstacle sitting in a lane
        centre with an adjacent same-direction lane free to pass it, or None."""
        for k in range(len(self.paths)):
            same = [q for q in range(len(self.paths)) if self.dirs[q] == self.dirs[k]]
            pos = same.index(k)
            nbrs = [same[q] for q in (pos - 1, pos + 1) if 0 <= q < len(same)]
            for a, lat, j in self.obstacles_along(k, 0, len(self.paths[k])):
                if lat >= corridor or not self.exists[k][a]:
                    continue
                win = (a + np.arange(-int(6 / STEP), int(4 / STEP))) % len(self.paths[k])
                if not self.exists[k][win].all():
                    continue
                for q in nbrs:
                    if self.exists[q][win].all() and not any(
                            l2 < corridor for _, l2, _ in self.obstacles_along(q, win[0], len(win))):
                        return k, int(win[0] + int(1 / STEP)) % len(self.paths[k]), j
        return None

    def find_edge_cones(self, corridor=0.30, margin=0.15):
        """(lane, index 5 m before, index 4 m past) of the longest stretch of
        obstacles inside a lane's edge but clear of its centre corridor."""
        best = None
        for k in range(len(self.paths)):
            hits = [a for a, lat, _ in self.obstacles_along(k, 0, len(self.paths[k]))
                    if corridor <= lat < 0.5 * self.widths[k][a] + margin and self.exists[k][a]]
            blocked = [a for a, lat, _ in self.obstacles_along(k, 0, len(self.paths[k]))
                       if lat < corridor]
            if not hits:
                continue
            a0, a1 = min(hits), max(hits)
            if any(a0 - 100 <= b <= a1 + 80 for b in blocked):
                continue
            if best is None or a1 - a0 > best[2] - best[1]:
                best = (k, (a0 - 100) % len(self.paths[k]), (a1 + 80) % len(self.paths[k]))
        return best

    def find_straight(self, clear_m=8.0):
        """(lane, index) on the straightest clear stretch of the first lane."""
        k = 0
        p = self.paths[k]
        d1 = np.gradient(p, axis=0)
        d2 = np.gradient(d1, axis=0)
        kap = np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / np.linalg.norm(d1, axis=1) ** 3
        kap = np.convolve(np.concatenate([kap[-80:], kap, kap[:80]]), np.ones(161) / 161, 'same')[80:-80]
        n = int(clear_m / STEP)
        for i in np.argsort(kap):
            if self.exists[k][(i + np.arange(n)) % len(p)].all() and not self.obstacles_along(k, i, n):
                return k, int(i)
        return k, 0


# ── metrics for one recorded run ──

def run_metrics(truth, rec, xs):
    """rec: dict of equal-length arrays sampled along the run:
         t, x, y, yaw, v_cmd       pose [m, rad] and commanded speed
         ref_t                     receive time of the ref in use (sim s)
         ref                       (n, len(xs)) predicted Y in the car frame
    Returns a dict of scalars (see keys)."""
    t, x, y, yaw = (np.asarray(rec[k], float) for k in ('t', 'x', 'y', 'yaw'))
    ref = np.asarray(rec['ref'], float)
    lanes = [truth.lane_at(a, b, c) for a, b, c in zip(x, y, yaw)]
    cte = np.array([l[2] for l in lanes])
    w = np.array([l[3] for l in lanes])
    lane_id = np.array([l[0] for l in lanes])
    # a lane that is tapering in or out (merge / drop) has no fixed lines to
    # depart from: the car follows the merge there
    full = np.array([bool(truth.exists[l[0]][l[1]]) for l in lanes])
    gaps = np.array([truth.clearance(a, b, c)[0] for a, b, c in zip(x, y, yaw)])
    prot = [truth.protected_at(a, b, c) for a, b, c in zip(x, y, yaw)]
    prot_in = np.array([p[1] for p in prot])          # how far into a bike/bus/parking lane [m]
    prot_kind = np.array([p[0] or '' for p in prot])

    def world(i, X, Y):
        c, s = math.cos(yaw[i]), math.sin(yaw[i])
        return x[i] + c * X - s * Y, y[i] + s * X + c * Y

    # reference line vs the true centre of the lane it should be in
    ref_err0, ref_err3, jumps = [], [], []
    ref_t = np.asarray(rec['ref_t'], float)
    new = np.flatnonzero(np.diff(ref_t, prepend=-1.0) != 0)
    prev = None
    for i in new:
        for X, out in ((xs[0], ref_err0), (xs[-1], ref_err3)):
            wx, wy = world(i, X, np.interp(X, xs, ref[i]))
            l = truth.lane_at(wx, wy, yaw[i])
            out.append(abs(l[2]))
        line = LineString([world(i, X, Y) for X, Y in zip(xs, ref[i])])
        if prev is not None:
            mid = Point(world(i, 1.5, np.interp(1.5, xs, ref[i])))
            jumps.append(prev.distance(mid))
        prev = line
    d1 = np.gradient(ref, xs, axis=1, edge_order=2)
    d2 = np.gradient(d1, xs, axis=1, edge_order=2)
    kap = np.abs(d2) / (1 + d1 * d1) ** 1.5
    dur = t[-1] - t[0] if len(t) > 1 else 0.0
    gaps_t = np.diff(ref_t[new]) if len(new) > 1 else np.array([np.inf])
    return dict(
        duration_s=float(dur),
        distance_m=float(np.hypot(np.diff(x), np.diff(y)).sum()),
        # reference line (Level 2)
        ref_rate_hz=float(len(new) / dur) if dur > 0 else 0.0,
        ref_max_gap_s=float(gaps_t.max()),
        ref_err_car_mean=float(np.mean(ref_err0)) if ref_err0 else float('nan'),
        ref_err_3m_mean=float(np.mean(ref_err3)) if ref_err3 else float('nan'),
        ref_jump_p95=float(np.percentile(jumps, 95)) if jumps else 0.0,
        ref_kappa_max=float(kap.max()) if kap.size else 0.0,
        # tracking (Level 3)
        own_ref_offset_rms=float(np.sqrt(np.mean(ref[:, 0] ** 2))),   # ref Y at X=0 = car's offset from its own path
        true_offset_rms=float(np.sqrt(np.mean(cte ** 2))),
        true_offset_max=float(np.abs(cte).max()),
        departure_frac=float(np.mean(full & (np.abs(cte) > 0.5 * w - CAR_HALF_W))),
        lane_start=int(lane_id[0]),
        lane_end=int(lane_id[-1]),
        # safety / progress (Level 4)
        min_clearance_m=float(gaps.min()),
        protected_frac=float(np.mean(np.isin(prot_kind, ['bike', 'bus', 'parking']) & (prot_in > 0.02))),
        protected_max_m=float(prot_in[np.isin(prot_kind, ['bike', 'bus', 'parking'])].max())
        if np.isin(prot_kind, ['bike', 'bus', 'parking']).any() else 0.0,
        collision=bool((gaps <= 0.0).any()),
        stopped_frac=float(np.mean(np.asarray(rec['v_cmd'], float) < 0.1)),
        # per-sample series (not scalars) for event analysis
        cte=cte, w=w, lane_id=lane_id, gaps=gaps, full=full, prot_in=prot_in, prot_kind=prot_kind,
        ref_err0=np.array(ref_err0), ref_err3=np.array(ref_err3), ref_new=new,
    )


def encounters(truth, rec, ahead_m=8.0, gap_s=1.0):
    """Every time the car drove at an obstacle sitting in its lane's path.

    An encounter starts when an obstacle that would hit a car on the lane
    centre is within ahead_m in the car's current lane, and lasts while it
    stays there (gaps under gap_s merge). Its outcome uses the car footprint
    against the obstacle footprint from the start of the encounter until 2 s
    after it: 'collision' (they touched), 'avoided', or 'incomplete' (the run
    ended before the car got past). -> [dict(obstacle, kind, t, seen_m,
    min_gap_m, outcome)]"""
    t, x, y, yaw = (np.asarray(rec[k], float) for k in ('t', 'x', 'y', 'yaw'))
    hits = {}
    for i in range(len(t)):
        here = truth.lane_at(x[i], y[i], yaw[i])
        if here is None:
            continue
        for a, lat, j in truth.obstacles_along(here[0], here[1], int(ahead_m / STEP)):
            if lat < CAR_HALF_W + OBJ_HALF_W[truth.kinds[j]] + IN_PATH_MARGIN:
                hits.setdefault(j, []).append((i, a * STEP))
    out = []
    for j, hs in hits.items():
        spans, cur = [], [hs[0]]
        for h in hs[1:]:
            if t[h[0]] - t[cur[-1][0]] > gap_s:
                spans.append(cur)
                cur = []
            cur.append(h)
        spans.append(cur)
        for sp in spans:
            a, b = sp[0][0], sp[-1][0]
            end = int(np.searchsorted(t, t[b] + 2.0))
            gap = min(car_polygon(x[i], y[i], yaw[i]).distance(truth.polys[j])
                      for i in range(a, min(end, len(t))))
            outcome = ('collision' if gap <= 0.0 else
                       'incomplete' if end >= len(t) else 'avoided')
            out.append(dict(obstacle=j, kind=truth.kinds[j], t=float(t[a]),
                            seen_m=float(max(d for _, d in sp)), min_gap_m=float(gap), outcome=outcome))
    return sorted(out, key=lambda e: e['t'])
