#!/usr/bin/env python3
"""Score a ViT driving run against the map's ground truth.

  /usr/bin/python3 analyze_run.py                       # newest log/vit_run_*
  /usr/bin/python3 analyze_run.py ../log/vit_run_20261005_101500 [--map road_05]

Reads the run folder written by vit_follow_launch.py (vit.csv, follow_*.csv,
frames/) and writes everything back INTO that folder:

  report.txt     summary, every collision / near miss / lane departure /
                 obstacle encounter, and per-section numbers
  track.png      the map with the car's track coloured by lane offset and the
                 events marked
  timeline.png   lane offset, reference error, steering, speed, clearance
  events/        the logged camera frames nearest each event

The map is detected from the poses (the truth file whose lanes they sit on)
unless --map is given. Lane departures are blamed on PERCEPTION when the
published path itself was off the lane centre there, else on CONTROL (the car
did not follow a correct path).
"""

import argparse
import csv
import glob
import math
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.normpath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, PKG)
from lane_assist.eval_tools import CAR_HALF_W, TRUTH_DIR, Truth, car_polygon, run_metrics   # noqa: E402

SIM = os.path.normpath(os.path.join(PKG, '..', 'f1tenth_gym_gazebo'))
NEAR_MISS = 0.05            # clearance below this is a near miss [m]
PERCEPTION_ERR = 0.15       # path off the true centre by more than this -> perception's fault [m]
ENCOUNTER_M = 8.0           # an obstacle within this far ahead in the car's lane is an encounter
N_SECTIONS = 10


def load_run(d):
    rows = list(csv.DictReader(open(os.path.join(d, 'vit.csv'))))
    if not rows:
        sys.exit(f'{d}/vit.csv is empty: did the car drive?')
    xs = np.array([float(v) for v in
                   open(os.path.join(d, 'run.txt')).read().split('xs:')[1].split('\n')[0].split()])
    rec = dict(t=[float(r['sim_t']) for r in rows], x=[float(r['x']) for r in rows],
               y=[float(r['y']) for r in rows], yaw=[float(r['yaw']) for r in rows],
               vx=[float(r['vx']) for r in rows],
               ref=[[float(v) for v in r['pred'].split()] for r in rows],
               frame_age=[float(r['frame_age_s']) for r in rows],
               infer_ms=[float(r['infer_ms']) for r in rows])
    rec['ref_t'] = rec['t']
    fol = glob.glob(os.path.join(d, 'follow_*.csv'))
    t = np.array(rec['t'])
    if fol:
        f = list(csv.DictReader(open(fol[0])))
        ft = np.array([float(r['sim_t']) for r in f])
        j = np.clip(np.searchsorted(ft, t), 0, len(f) - 1)
        rec['v_cmd'] = [float(f[i]['speed_cmd']) for i in j]
        rec['steer_deg'] = [float(f[i]['steer_deg']) for i in j]
        rec['status'] = [f[i]['status'] for i in j]
    else:
        rec['v_cmd'] = rec['vx']
        rec['steer_deg'] = [float('nan')] * len(t)
        rec['status'] = ['?'] * len(t)
    return rec, xs


def detect_map(rec):
    pts = list(zip(rec['x'], rec['y'], rec['yaw']))[::max(1, len(rec['x']) // 40)]
    best = None
    for f in sorted(glob.glob(os.path.join(TRUTH_DIR, 'road_*.npz'))):
        T = Truth(os.path.basename(f)[:-4])
        e = np.median([abs(l[2]) if l else 9.0 for l in (T.lane_at(*p) for p in pts)])
        if best is None or e < best[0]:
            best = (e, T)
    if best[0] > 0.6:
        sys.exit(f'poses do not sit on any road_NN map (best median offset {best[0]:.2f} m)')
    return best[1]


def spans(mask):
    """[(start, end)] index ranges where mask is True."""
    i = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(np.int8), [0]])))
    return list(zip(i[::2], i[1::2]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run', nargs='?')
    ap.add_argument('--map')
    args = ap.parse_args()
    d = args.run or max(glob.glob(os.path.join(PKG, 'log', 'vit_run_*')), key=os.path.getmtime)
    d = os.path.abspath(d)
    rec, xs = load_run(d)
    T = Truth(args.map) if args.map else detect_map(rec)
    m = run_metrics(T, rec, xs)
    t = np.array(rec['t']) - rec['t'][0]
    x, y, yaw = (np.array(rec[k]) for k in ('x', 'y', 'yaw'))
    cte, w, lane, gaps = m['cte'], m['w'], m['lane_id'], m['gaps']
    # path error exists per NEW reference; carry it to every sample (ticks can
    # repeat a sim time when the sim runs slower than real time)
    idx = np.maximum.accumulate(np.where(np.isin(np.arange(len(t)), m['ref_new']),
                                         np.arange(len(t)), 0))
    pos = np.searchsorted(m['ref_new'], idx)
    err0, err3 = m['ref_err0'][pos], m['ref_err3'][pos]
    own = np.array(rec['ref'])[:, 0]                   # car's offset from its own path
    frames = sorted(glob.glob(os.path.join(d, 'frames', '*.jpg')))
    frame_t = np.array([float(os.path.basename(f)[:-4]) for f in frames])
    ev_dir = os.path.join(d, 'events')
    shutil.rmtree(ev_dir, ignore_errors=True)
    os.makedirs(ev_dir)
    out = []

    def frame_at(i, tag):
        if not len(frames):
            return ''
        j = int(np.argmin(np.abs(frame_t - rec['t'][i])))
        dst = os.path.join(ev_dir, f'{tag}_t{t[i]:06.1f}.jpg')
        shutil.copy(frames[j], dst)
        return os.path.relpath(dst, d)

    # teleports (/initialpose, RViz 2D Pose Estimate): the car jumps; anything
    # within 1.5 s after one is where it was PUT, not where it drove
    tp = np.flatnonzero(np.hypot(np.diff(x), np.diff(y)) > 1.0) + 1

    def after_tp(i):
        return bool(len(tp)) and bool(np.any((t[i] - t[tp] >= 0) & (t[i] - t[tp] < 1.5)))

    def where(i):
        k, idx, _, _ = T.lane_at(x[i], y[i], yaw[i])
        n = T.paths.shape[1]
        sec = (idx if T.dirs[k] > 0 else n - 1 - idx) * N_SECTIONS // n
        return f't={t[i]:6.1f}s  ({x[i]:+6.2f},{y[i]:+6.2f})  lane {k}  section {sec}'

    out.append(f'RUN       {d}')
    out.append(f'map       {T.name}  ' + ('(given)' if args.map else '(detected from poses)'))
    out.append(open(os.path.join(d, 'run.txt')).read().strip())
    out.append('')
    out.append('SUMMARY')
    out.append(f'  duration {m["duration_s"]:.1f} s, distance {m["distance_m"]:.1f} m, '
               f'mean speed {m["distance_m"] / max(m["duration_s"], 1e-6):.2f} m/s, '
               f'stopped {m["stopped_frac"]:.0%} of the time')
    out.append(f'  reference  : {m["ref_rate_hz"]:.1f} Hz (max gap {m["ref_max_gap_s"]:.2f} s), '
               f'error vs true lane centre at car {m["ref_err_car_mean"]:.3f} m, at {xs[-1]:.0f} m '
               f'{m["ref_err_3m_mean"]:.3f} m, frame-to-frame jump p95 {m["ref_jump_p95"]:.3f} m')
    out.append(f'  tracking   : offset from own path rms {m["own_ref_offset_rms"]:.3f} m, '
               f'from TRUE lane centre rms {m["true_offset_rms"]:.3f} m / max {m["true_offset_max"]:.3f} m')
    out.append(f'  lane       : departure {m["departure_frac"]:.1%} of the time; in a bike/bus/parking '
               f'lane {m["protected_frac"]:.1%} of the time (deepest {m["protected_max_m"]:.2f} m)')
    real = [i for i in np.flatnonzero(gaps <= 0) if not after_tp(i)]
    out.append(f'  safety     : min clearance {m["min_clearance_m"]:.3f} m, '
               f'collision {"YES" if real else "no"}'
               + (f'  ({len(tp)} teleports at t = {", ".join(f"{v:.1f}" for v in t[tp])} s; '
                  'contact right after one is not counted)' if len(tp) else ''))
    out.append(f'  timing     : inference p95 {np.percentile(rec["infer_ms"], 95):.1f} ms, '
               f'camera frame age p95 {np.percentile(rec["frame_age"], 95):.3f} s')

    # collisions and near misses, one line per contact
    out.append('')
    out.append('COLLISIONS / NEAR MISSES')
    hit = gaps < NEAR_MISS
    if not hit.any():
        out.append('  none')
    for a, b in spans(hit):
        i = a + int(np.argmin(gaps[a:b]))
        _, j = T.clearance(x[i], y[i], yaw[i])
        kind = 'COLLISION' if gaps[a:b].min() <= 0 else 'near miss'
        if after_tp(a):
            kind = 'teleported into contact' if kind == 'COLLISION' else 'teleported near'
        out.append(f'  {kind:9s} {where(i)}  with {T.kinds[j]} #{j}  min gap {gaps[i]:.3f} m  '
                   f'{t[b - 1] - t[a]:.1f} s  frame {frame_at(i, kind.split()[0].lower())}')

    # lane departures, blamed on perception or control
    out.append('')
    out.append('LANE DEPARTURES (wheel over the lane line, outside intended lane changes)')
    dep = m['full'] & (np.abs(cte) > 0.5 * w - CAR_HALF_W)
    n_dep = 0
    for a, b in spans(dep):
        changed = lane[max(0, a - 20)] != lane[min(len(lane) - 1, b + 20)]
        i = a + int(np.argmax(np.abs(cte[a:b])))
        if changed:
            continue
        n_dep += 1
        blame = ('TELEPORT' if after_tp(a) else
                 'PERCEPTION' if err0[i] > PERCEPTION_ERR else 'CONTROL')
        out.append(f'  {blame:10s} {where(i)}  {t[b - 1] - t[a]:.1f} s  offset {cte[i]:+.3f} m '
                   f'(lane {w[i]:.2f} m)  path error at car {err0[i]:.3f} m  '
                   f'frame {frame_at(i, "departure")}')
    if not n_dep:
        out.append('  none')

    # obstacles: IN PATH (would hit a car driving the lane centre) vs EDGE
    # (passed by staying centred). One pass of the car = one encounter.
    half = {'cone': 0.03, 'barrel': 0.045, 'car': 0.10, 'barrier': 0.035}
    seen = {}
    for i in range(len(t)):
        k, idx, _, _ = T.lane_at(x[i], y[i], yaw[i])
        for a, lat, j in T.obstacles_along(k, idx, int(ENCOUNTER_M / 0.05)):
            if lat < 0.5 * T.widths[k][(idx + a) % T.paths.shape[1]] + 0.15:
                inpath = lat < CAR_HALF_W + half[T.kinds[j]] + 0.02
                seen.setdefault(j, []).append((i, a * 0.05, k, inpath))
    passes = []
    for j, hits in seen.items():
        ii = np.array([h[0] for h in hits])
        for a, b in spans(np.isin(np.arange(len(t)), ii)):
            seg = [h for h in hits if a <= h[0] < b]
            end = min(len(t) - 1, b + 40)
            g = min(car_polygon(x[i], y[i], yaw[i]).distance(T.polys[j]) for i in range(a, end))
            passes.append((j, a, b, seg, end, g, any(h[3] for h in seg)))

    out.append('')
    out.append('OBSTACLES IN THE CAR\'S PATH (would hit a car driving the lane centre)')
    n_in = 0
    for j, a, b, seg, end, g, inpath in sorted(passes, key=lambda p: p[1]):
        if not inpath:
            continue
        n_in += 1
        k0 = seg[0][2]
        react = None
        for i, dist, k, _ in seg:
            if after_tp(i):                          # still recovering from where it was put
                continue
            c, s_ = math.cos(yaw[i]), math.sin(yaw[i])
            px = x[i] + c * xs[-1] - s_ * rec['ref'][i][-1]
            py = y[i] + s_ * xs[-1] + c * rec['ref'][i][-1]
            p = T.paths[k0]
            q = int(np.argmin(np.hypot(p[:, 0] - px, p[:, 1] - py)))
            if abs((np.array([px, py]) - p[q]) @ T.nrm[k0][q]) > 0.35 * T.widths[k0][q]:
                react = dist
                break
        outcome = ('COLLISION' if g <= 0 else 'passed (lane change)' if lane[end] != k0
                   else 'stopped / run ended' if rec['v_cmd'][end] < 0.1 or end == len(t) - 1
                   else 'passed in lane' if g > NEAR_MISS else 'squeezed past in lane')
        out.append(f'  {T.kinds[j]} #{j}: seen {max(h[1] for h in seg):.1f} m ahead, path turned away at '
                   f'{"%.1f m" % react if react is not None else "NEVER"}, min gap {g:.3f} m -> {outcome}'
                   f'  [{where(a)}]  frame {frame_at(a + (b - a) // 2, "inpath")}')
    if not n_in:
        out.append('  none met -- this run did not test avoidance (use test_closed_loop.py '
                   'test_4_blocking_obstacle, which starts behind one)')

    out.append('')
    out.append('EDGE OBSTACLES (inside the lane edge, passed by staying centred)')
    edge = [p for p in passes if not p[6]]
    if not edge:
        out.append('  none')
    else:
        gs = np.array([p[5] for p in edge])
        out.append(f'  {len(edge)} passes of {len({p[0] for p in edge})} obstacles, min gap '
                   f'{gs.min():.3f} m, median {np.median(gs):.3f} m, collisions {int((gs <= 0).sum())}')
        for j, a, b, seg, end, g, _ in sorted(edge, key=lambda p: p[5]):
            if g < 0.15:
                out.append(f'  tight: {T.kinds[j]} #{j} gap {g:.3f} m  [{where(a)}]  '
                           f'frame {frame_at(a + (b - a) // 2, "tight")}')

    # protected lanes entered (bike / bus / parking), outside teleport landings
    out.append('')
    out.append('BIKE / BUS / PARKING LANE ENTRIES')
    pin = np.isin(m['prot_kind'], ['bike', 'bus', 'parking']) & (m['prot_in'] > 0.02)
    n_p = 0
    for a, b in spans(pin):
        if after_tp(a):
            continue
        n_p += 1
        i = a + int(np.argmax(m['prot_in'][a:b]))
        out.append(f'  {m["prot_kind"][i]:8s} {where(i)}  {t[b - 1] - t[a]:.1f} s  {m["prot_in"][i]:.2f} m in  '
                   f'frame {frame_at(i, "protected")}')
    if not n_p:
        out.append('  none')

    # per section of the loop
    out.append('')
    out.append('PER SECTION (10 equal stretches of the loop)')
    out.append('  sec  samples  |offset| mean  path err car  path err 3m  speed  departures')
    n = T.paths.shape[1]
    sec = np.array([(T.lane_at(*p)[1] if T.dirs[T.lane_at(*p)[0]] > 0 else n - 1 - T.lane_at(*p)[1])
                    * N_SECTIONS // n for p in zip(x, y, yaw)])
    v = np.array(rec['vx'])
    for s_ in range(N_SECTIONS):
        k = sec == s_
        if k.any():
            out.append(f'  {s_:3d}  {k.sum():7d}  {np.abs(cte[k]).mean():12.3f}  {err0[k].mean():12.3f}  '
                       f'{err3[k].mean():11.3f}  {v[k].mean():5.2f}  {dep[k].mean():9.0%}')

    report = '\n'.join(out)
    open(os.path.join(d, 'report.txt'), 'w').write(report + '\n')
    print(report)
    plots(d, T, rec, m, t, x, y, cte, err0, err3, own, gaps, dep)
    print(f'\nwritten to {d}: report.txt, track.png, timeline.png, events/')


def plots(d, T, rec, m, t, x, y, cte, err0, err3, own, gaps, dep):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import cv2

    yml = open(os.path.join(SIM, 'maps', T.name + '.yaml')).read()
    ox = float(yml.split('origin: [')[1].split(',')[0])
    ext = -2 * ox
    fig, ax = plt.subplots(figsize=(11, 11))
    prev = os.path.join(SIM, 'maps', T.name + '_preview.png')
    if os.path.exists(prev):
        ax.imshow(cv2.imread(prev)[:, :, ::-1], extent=[ox, ox + ext, ox, ox + ext], alpha=0.6)
    for p, ex in zip(T.paths, T.exists):
        ax.plot(p[ex, 0], p[ex, 1], ',', color='white', alpha=0.6)
    sc = ax.scatter(x, y, c=np.abs(cte), cmap='turbo', vmin=0, vmax=0.3, s=6)
    plt.colorbar(sc, ax=ax, fraction=0.03, label='|offset from true lane centre| [m]')
    obs = [i for i, k in enumerate(T.kinds) if k != 'barrier']
    ax.scatter(T.obj[obs, 0], T.obj[obs, 1], marker='s', s=25, c='black', label='obstacles')
    hit = gaps <= 0
    ax.scatter(x[hit], y[hit], marker='x', s=120, c='red', label='collision')
    ax.scatter(x[dep], y[dep], marker='o', s=30, facecolors='none', edgecolors='orange',
               label='lane departure')
    ax.plot(x[0], y[0], 'g^', ms=12, label='start')
    pad = 1.5
    ax.set_xlim(x.min() - pad, x.max() + pad)
    ax.set_ylim(y.min() - pad, y.max() + pad)
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    ax.set_title(f'{T.name}  {os.path.basename(d)}  collision={m["collision"]}  '
                 f'offset rms {m["true_offset_rms"]:.3f} m')
    fig.savefig(os.path.join(d, 'track.png'), dpi=110, bbox_inches='tight')
    plt.close(fig)

    fig, axs = plt.subplots(5, 1, figsize=(14, 13), sharex=True)
    axs[0].plot(t, cte, label='car vs TRUE lane centre')
    axs[0].plot(t, -own, label='car vs its OWN path', alpha=0.7)
    axs[0].plot(t, 0.5 * m['w'] - CAR_HALF_W, 'r--', lw=0.8, label='departure limit')
    axs[0].plot(t, -(0.5 * m['w'] - CAR_HALF_W), 'r--', lw=0.8)
    axs[0].set_ylabel('offset [m]')
    axs[1].plot(t, err0, label='at car')
    axs[1].plot(t, err3, label='at 3 m', alpha=0.7)
    axs[1].set_ylabel('path error [m]')
    axs[2].plot(t, rec['steer_deg'])
    axs[2].set_ylabel('steer [deg]')
    axs[3].plot(t, rec['v_cmd'], label='command')
    axs[3].plot(t, rec['vx'], label='measured', alpha=0.7)
    axs[3].set_ylabel('speed [m/s]')
    axs[4].plot(t, np.minimum(gaps, 2.0))
    axs[4].axhline(0, color='r', lw=0.8)
    axs[4].set_ylabel('clearance [m]')
    axs[4].set_xlabel('time [s]')
    for a in axs:
        a.grid(alpha=0.3)
        if a.get_legend_handles_labels()[0]:
            a.legend(loc='upper right', fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(d, 'timeline.png'), dpi=100)
    plt.close(fig)


if __name__ == '__main__':
    main()
