#!/usr/bin/env python3
"""Labels + train/val/test split + visual check for collected road_NN data.

  /usr/bin/python3 make_dataset.py road_00 [road_01 ...] [--val 2] [--test 6]
  /usr/bin/python3 make_dataset.py $(ls ../dataset/raw | grep road_) --test   # many maps:
        one val section per map, test = whole maps held out in train_vit.py --test-maps

For every frame in dataset/raw/<map>/ and dataset/raw/<map>__<tag>/ (extra
collection runs on the same map, e.g. avoidance top-ups):
  1. the expert plan from the recorded pose, eval_tools.Truth.plan_label():
     lane centre, or the smooth lane change around an obstacle in the lane --
     the SAME function the collector drove and the tests check against. It
     needs only the pose, so frames from any collector version get the same,
     current labels
  2. move it into the car's frame
  3. read off the label: path Y [m, +left] at vit_lane.XS (0..5 m ahead)
Frames whose plan runs into an obstacle with no free lane are KEPT: the path
stays in the lane and `free` says how far it is to the stop.

Split by TRACK SECTION, not by frame: neighbouring frames are near copies, so
a random split would leak. The loop is cut into 10 equal sections by
reference arc length; --val / --test sections are held out, and train frames
within GAP_M of a held-out section are dropped (they would see its road).

Writes:
  dataset/<map>_labels.csv     file, map, split, lane_change, top, y00..y15, free, wrong
                               (free = eval_tools.free_distance: metres to an obstacle with
                               no lane to pass it, FREE_MAX when none -- the stop label;
                               wrong = 1 when the car is on the wrong side of the road)
  dataset/check/<map>/*.jpg    N_CHECK frames with the label drawn in green
                               (blue line = top of the crop the model sees)
"""

import argparse
import csv
import glob
import math
import os
import sys

import cv2
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.normpath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, PKG)
from lane_assist.camera_geometry import horizon_row, pixel_from_ground  # noqa: E402
from lane_assist.eval_tools import Truth, FREE_MAX                     # noqa: E402
from lane_assist.vit_lane import XS                                    # noqa: E402

DATA = os.path.join(PKG, 'dataset')
TRUTH = os.path.join(PKG, 'tools', 'maps', 'truth')
N_BLOCKS = 10
GAP_M = 4.0
N_CHECK = 30
OFF_PLAN_FRAC = 0.45            # = road_collect record_max_frac


def lane_label(pts, x, y, yaw):
    """Plan points (world, in driving order) -> Y at XS in the car frame, or
    None if they do not run monotonically forward over 0..5 m."""
    d = pts - (x, y)
    c, s = math.cos(yaw), math.sin(yaw)
    X = c * d[:, 0] + s * d[:, 1]
    Y = -s * d[:, 0] + c * d[:, 1]
    # Only the stretch the label reads must run forward; mm-scale kinks in the
    # lane path well behind the car (taper/shift seams) are irrelevant.
    use = (X >= XS[0] - 0.2) & (X <= XS[-1] + 0.2)
    j = np.flatnonzero(use)
    if (j.size < 2 or not np.all(np.diff(j) == 1) or not np.all(np.diff(X[j]) > 0)
            or X[j[0]] > XS[0] or X[j[-1]] < XS[-1]):
        return None
    return np.interp(XS, X[j], Y[j])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('maps', nargs='+')
    ap.add_argument('--val', type=int, nargs='+', default=[2], help='held-out sections 0..9')
    ap.add_argument('--test', type=int, nargs='*', default=[6],
                    help='held-out sections 0..9; none (bare --test) when whole maps are held out')
    args = ap.parse_args()
    cam = yaml.safe_load(open(os.path.join(PKG, 'config', 'lane_assist_params.yaml')))
    cam = cam['vit_lane_node']['ros__parameters']
    h, pitch, cx_ = float(cam['cam_height']), float(cam['cam_pitch']), float(cam['cam_x'])
    rng = np.random.default_rng(0)
    for name in args.maps:
        T = Truth(name, TRUTH)
        N = T.paths.shape[1]
        held = {b: 'val' for b in args.val} | {b: 'test' for b in args.test}
        gap = int(GAP_M / 0.05)
        rows, dropped, blocked, offplan, n_late, n_wrong = [], 0, 0, 0, 0, 0
        runs = [os.path.join(DATA, 'raw', name)] + sorted(glob.glob(os.path.join(DATA, 'raw', name + '__*')))
        for raw in [r_ for r_ in runs if os.path.exists(os.path.join(r_, 'poses.csv'))]:
            fx, fy, cx, cy, W, H = map(float, open(os.path.join(raw, 'camera.txt')).read().split())
            K = (fx, fy, cx, cy)
            top = int(max(0, math.ceil(horizon_row(K, pitch)) + 2))
            for r in csv.DictReader(open(os.path.join(raw, 'poses.csv'))):
                x, y, yaw = float(r['x']), float(r['y']), float(r['yaw'])
                got = T.plan_label(x, y, yaw, back=2.0, ahead=8.0)
                here = T.lane_at(x, y, yaw)
                if got is None or here is None:
                    dropped += 1
                    continue
                pts, blk, _, k = got
                # no lane to pass in: still a label -- the path stays in the lane and
                # the free distance says where to stop (dropped before 2026-10-06)
                blocked += int(blk)
                lab = lane_label(pts, x, y, yaw)
                if lab is None:
                    dropped += 1
                    continue
                # lane change visible in the label: the plan leaves its lane within 0..5 m
                changing = T.plan_label(x, y, yaw, back=0.0, ahead=XS[-1])[2]
                # far off the plan: either still in its lane with an obstacle close
                # ahead (a late lane change -- exactly the state a hesitating model
                # reaches, so it gets the late-change label), or not a state the
                # expert is ever in -> no label
                if abs(lab[0]) > OFF_PLAN_FRAC * here[3]:
                    late = T.late_label(x, y, yaw, XS, lab)
                    if late is not None:
                        lab, changing, n_late = late, True, n_late + 1
                    elif T.oncoming_at(x, y, yaw):
                        n_wrong += 1                     # wrong side: the plan IS the way back
                    else:
                        offplan += 1
                        continue
                free = T.free_distance(x, y, yaw) if blk else FREE_MAX
                rows.append((raw, r, K, top, lab, int(changing), here, free, int(T.oncoming_at(x, y, yaw))))
        if not rows:
            print(f'{name}: no collected frames in dataset/raw/{name}[__*] -- skipped')
            continue
        out_rows = []
        for raw, r, K, top, lab, changing, here, free, wrong in rows:
            k, i = here[0], here[1]
            ref_i = i if T.dirs[k] > 0 else N - 1 - i      # same section for both directions
            block = ref_i * N_BLOCKS // N
            split = held.get(block, 'train')
            if split == 'train':
                near = [abs((ref_i - b * N // N_BLOCKS + N // 2) % N - N // 2) for b in held] + \
                       [abs((ref_i - (b + 1) * N // N_BLOCKS + N // 2) % N - N // 2) for b in held]
                if min(near) < gap:
                    split = 'gap'
            out_rows.append([os.path.relpath(os.path.join(raw, r['file']), DATA), name, split,
                             changing, top] + [f'{v:.4f}' for v in lab] + [f'{free:.3f}', wrong])
        cams = {os.path.relpath(raw, DATA): K for raw, _, K, *_ in rows}
        rows = out_rows

        out = os.path.join(DATA, name + '_labels.csv')
        with open(out, 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['file', 'map', 'split', 'lane_change', 'top'] + [f'y{j:02d}' for j in range(len(XS))]
                       + ['free', 'wrong'])
            w.writerows(rows)

        # visual check: label projected into the camera image
        chk = os.path.join(DATA, 'check', name)
        os.makedirs(chk, exist_ok=True)
        for f in os.listdir(chk):
            os.remove(os.path.join(chk, f))
        lc = [r for r in rows if r[3]]
        plain = [r for r in rows if not r[3]]
        pick = (list(rng.choice(len(lc), min(len(lc), N_CHECK // 3), replace=False)) if lc else [])
        sample = [lc[j] for j in pick] + [plain[j] for j in
                                          rng.choice(len(plain), min(len(plain), N_CHECK - len(pick)),
                                                     replace=False)]
        for r in sample:
            img = cv2.imread(os.path.join(DATA, r[0]))
            K = cams[os.path.dirname(r[0])]
            Y = np.array(r[5:5 + len(XS)], float)
            Xd = np.linspace(XS[0], XS[-1], 60)
            u, v = pixel_from_ground(Xd, np.interp(Xd, XS, Y), K, h, pitch, cx_)
            # points behind the camera (X < cam_x) project to the sky, flipped
            ok = np.isfinite(u) & np.isfinite(v) & (Xd > cx_ + 0.05)
            pts = np.stack([u[ok], v[ok]], 1).astype(np.int32)
            cv2.polylines(img, [pts], False, (0, 255, 0), 3, cv2.LINE_AA)
            u, v = pixel_from_ground(XS, Y, K, h, pitch, cx_)
            for xx, uu, vv in zip(XS, u, v):
                if np.isfinite(uu) and np.isfinite(vv) and xx > cx_ + 0.05:
                    cv2.circle(img, (int(uu), int(vv)), 4, (0, 255, 0), -1)
            cv2.line(img, (0, top), (int(W), top), (255, 0, 0), 1)
            cv2.putText(img, f'{r[2]}{"  LANE CHANGE" if r[3] else ""}  y0={Y[0]:+.2f} y{XS[-1]:.0f}={Y[-1]:+.2f}'
                             f'{"  STOP in " + r[-2] + " m" if float(r[-2]) < FREE_MAX else ""}'
                             f'{"  WRONG SIDE" if r[-1] else ""}',
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imwrite(os.path.join(chk, r[0].replace(os.sep, '_')), img)

        sp = np.array([r[2] for r in rows])
        Y = np.array([r[5:5 + len(XS)] for r in rows], float)
        print(f'{name}: {len(rows)} frames from {len(cams)} run(s) ({dropped} dropped, '
              f'{blocked} stopping (no free lane), {offplan} off the plan; {n_late} late lane changes, '
              f'{n_wrong} wrong side) -> {out}')
        for s in ('train', 'val', 'test', 'gap'):
            m = sp == s
            print(f'  {s:5s} {m.sum():5d}  lane-change {int(np.array([r[3] for r in rows])[m].sum()):3d}')
        print(f'  label y at car  : mean {Y[:, 0].mean():+.3f}  std {Y[:, 0].std():.3f}  '
              f'range [{Y[:, 0].min():+.2f}, {Y[:, 0].max():+.2f}] m')
        print(f'  label y at {XS[-1]:.0f} m  : mean {Y[:, -1].mean():+.3f}  std {Y[:, -1].std():.3f}  '
              f'range [{Y[:, -1].min():+.2f}, {Y[:, -1].max():+.2f}] m')
        print(f'  check images    : {chk} ({len(sample)})')


if __name__ == '__main__':
    main()
