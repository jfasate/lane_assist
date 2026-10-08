"""Level 1 -- offline model tests (no sim, ~1 min).

  cd ~/sim_gazebo/src/lane_assist
  /usr/bin/python3 -m pytest test/test_model_offline.py -v -s
  LANE_TEST_MAP=road_00 VIT_CHECKPOINT=models/vit_lane_road_00.pt /usr/bin/python3 -m pytest ...

Scores the trained checkpoint on frames it never trained on: the map's
held-out track sections (dataset/<map>_labels.csv), or EVERY frame when the
checkpoint held the whole map out (train_vit.py --test-maps):
  accuracy      mean |error| at the car / 1.5 m / 3 m on the test split
  baseline      must clearly beat predict-the-train-mean (no mean collapse)
  slices        straight, curve, lane change, obstacle ahead, edge obstacle,
                next to a bike / bus lane (held-out = val + test, so rare
                cases have enough frames)
  protected     next to a bike / bus lane, the predicted path must not
                enter it
  symmetry      mirrored image -> mirrored prediction
  robustness    darker / brighter / noisier frame -> nearly the same path
  stability     consecutive frames -> the error does not jump
  latency       crop + inference fast enough for the 20 Hz node
"""

import csv
import os
import time

import numpy as np
import pytest
import torch

from lane_assist.eval_tools import CAR_HALF_W, IN_PATH_MARGIN, OBJ_HALF_W, Truth
from lane_assist.vit_lane import XS, build_lane_model, crop_resize

PKG = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
MAP = os.environ.get('LANE_TEST_MAP', 'road_00')
CKPT = os.environ.get('VIT_CHECKPOINT', os.path.join(PKG, 'models', f'vit_lane_{MAP}.pt'))
FAR = f'{XS[-1]:.0f}m'
STATION = {'car': 0, '1.5m': int(np.argmin(np.abs(XS - 1.5))), FAR: len(XS) - 1}

MAX_ERR = {'car': 0.10, '1.5m': 0.10, FAR: 0.30}    # test split, mean |error| [m]
BASELINE_RATIO = 0.6       # model error must be < 60% of the mean predictor's
MAX_SLICE_ERR = 0.20       # mean |error| over 0..3 m within each scenario slice [m]
MIN_SLICE = 10             # fewer frames than this -> slice skipped, not passed
FLIP_TOL = 0.05            # mean |pred(flip) + pred| [m]
PHOTO_TOL = 0.05           # mean |pred(perturbed) - pred| [m]
JITTER_TOL = 0.05          # mean |change of error| between consecutive frames [m]
LATENCY_MS = 20.0          # p95 crop + inference, one frame
MAX_PROTECTED_FRAC = 0.02  # share of next-to-bike/bus frames whose path enters that lane


def _cuda():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


@pytest.fixture(scope='module')
def model():
    if not os.path.isfile(CKPT):
        pytest.skip(f'no checkpoint {CKPT} (train with tools/train/train_vit.py)')
    c = torch.load(CKPT, map_location='cpu', weights_only=False)
    m = build_lane_model(npz=None, n_out=int(c['state_dict']['head.weight'].shape[0]))
    m.load_state_dict(c['state_dict'])
    return m.to(_cuda()).eval()


@pytest.fixture(scope='module')
def data():
    import cv2
    lab = os.path.join(PKG, 'dataset', f'{MAP}_labels.csv')
    if not os.path.isfile(lab):
        pytest.skip(f'no {lab} (run tools/data/make_dataset.py {MAP})')
    poses = {r['file']: r for r in csv.DictReader(
        open(os.path.join(PKG, 'dataset', 'raw', MAP, 'poses.csv')))}
    truth = Truth(MAP)
    held_out_map = MAP in torch.load(CKPT, map_location='cpu', weights_only=False).get('test_maps', []) \
        if os.path.isfile(CKPT) else False
    rows = list(csv.DictReader(open(lab)))
    for r in rows:
        if held_out_map:
            r['split'] = 'test'                        # the whole road is unseen
    rows = [r for r in rows if r['split'] in ('train', 'val', 'test')]
    d = dict(img=[], y=[], split=[], tags=[], frame=[], episode=[], raw=[])
    for r in rows:
        if r['split'] == 'train':
            d['y'].append([float(r[f'y{j:02d}']) for j in range(len(XS))])
            d['split'].append('train')
            d['img'].append(None)
            d['tags'].append(set())
            d['frame'].append(-1)
            d['episode'].append(-1)
            d['raw'].append(None)
            continue
        img = cv2.imread(os.path.join(PKG, 'dataset', r['file']))[:, :, ::-1]
        p = poses[os.path.basename(r['file'])]
        Y = np.array([float(r[f'y{j:02d}']) for j in range(len(XS))])
        tags = set()
        if int(r['lane_change']):
            tags.add('lane_change')
        k, i, _, w = truth.lane_at(float(p['x']), float(p['y']), float(p['yaw']))
        for a, lat, j in truth.obstacles_along(k, i, int(7.0 / 0.05)):
            if lat < CAR_HALF_W + OBJ_HALF_W[truth.kinds[j]] + IN_PATH_MARGIN:
                tags.add('obstacle_ahead')
            elif lat < 0.5 * w + 0.15 and a * 0.05 < 4.0:
                tags.add('edge_obstacle')
        same = [q for q in range(len(truth.paths)) if truth.dirs[q] == truth.dirs[k]]
        if truth.side is not None and k == same[-1]:
            r_ = int(np.argmin(np.hypot(*(truth.side['ref'] - (float(p['x']), float(p['y']))).T)))
            typ = truth.side['f_type'] if truth.dirs[k] > 0 else truth.side['b_type']
            if typ is not None and typ[r_] in ('bike', 'bus'):
                tags.add('next_to_bike_bus')
        # Road shape only for plain lane keeping: a lane-change path is curved
        # too, and would otherwise be counted as a "curve" a second time.
        if not tags & {'lane_change', 'obstacle_ahead'}:
            c2 = 2 * np.polyfit(XS, Y, 2)[0]
            tags.add('curve' if abs(c2) > 0.08 else 'straight' if abs(c2) < 0.03 else 'gentle')
        d['img'].append(crop_resize(img, int(r['top'])))
        d['y'].append(Y)
        d['split'].append(r['split'])
        d['tags'].append(tags)
        d['frame'].append(int(os.path.basename(r['file'])[:-4]))
        d['episode'].append(int(p['episode']))
        d['raw'].append(img)
        d.setdefault('pose', []).append((float(p['x']), float(p['y']), float(p['yaw'])))
    d['y'] = np.array(d['y'])
    d['split'] = np.array(d['split'])
    held = np.flatnonzero(d['split'] != 'train')
    d['held'] = held
    d['truth'] = truth
    d['x'] = torch.from_numpy(np.stack([d['img'][i] for i in held])).permute(0, 3, 1, 2)
    return d


def _predict(model, u8):
    out = []
    with torch.no_grad():
        for i in range(0, len(u8), 256):
            x = u8[i:i + 256].to(_cuda()).float() / 127.5 - 1.0
            out.append(model(x)[:, :len(XS)].float().cpu().numpy())      # path only (17th = stop head)
    return np.concatenate(out)


@pytest.fixture(scope='module')
def pred(model, data):
    return _predict(model, data['x'])                      # aligned with data['held']


def _held_mask(data, split):
    return data['split'][data['held']] == split


def test_has_test_frames(data):
    n = int(_held_mask(data, 'test').sum())
    assert n >= 50, f'only {n} test frames -- pick --test sections the car actually drove'


def test_beats_mean_baseline(data, pred):
    m = _held_mask(data, 'test')
    y = data['y'][data['held']][m]
    ref = data['split'] == 'train'
    base = np.abs(data['y'][ref if ref.any() else data['held']].mean(0) - y).mean()
    err = np.abs(pred[m] - y).mean()
    print(f'\n  test mean |error| {err:.3f} m vs mean-predictor {base:.3f} m '
          f'({err / base:.0%})')
    assert err < BASELINE_RATIO * base, \
        f'model {err:.3f} m is not clearly better than predicting the mean ({base:.3f} m)'


@pytest.mark.parametrize('where', list(STATION))
def test_accuracy(data, pred, where):
    m = _held_mask(data, 'test')
    j = STATION[where]
    err = np.abs(pred[m, j] - data['y'][data['held']][m, j])
    mean = float(err.mean())
    print(f'\n  {where:5s} mean {mean:.3f}  p95 {np.percentile(err, 95):.3f}  '
          f'max {err.max():.3f} m  (limit {MAX_ERR[where]:.2f})')
    assert mean < MAX_ERR[where], f'{where}: mean error {mean:.3f} m > {MAX_ERR[where]:.2f} m'


@pytest.mark.parametrize('tag', ['straight', 'curve', 'lane_change', 'obstacle_ahead', 'edge_obstacle',
                                 'next_to_bike_bus'])
def test_slice(data, pred, tag):
    m = np.array([tag in data['tags'][i] for i in data['held']])
    if m.sum() < MIN_SLICE:
        pytest.skip(f'only {int(m.sum())} held-out "{tag}" frames: not enough data to judge')
    err = np.abs(pred[m] - data['y'][data['held']][m]).mean(1)
    mean = float(err.mean())
    print(f'\n  {tag:15s} {int(m.sum()):4d} frames  mean {mean:.3f}  '
          f'p95 {np.percentile(err, 95):.3f} m  (limit {MAX_SLICE_ERR:.2f})')
    assert mean < MAX_SLICE_ERR, f'{tag}: mean error {mean:.3f} m > {MAX_SLICE_ERR:.2f} m'


def test_never_into_bike_bus_lane(data, pred):
    """Next to a bike / bus lane, the predicted path (0..5 m) must stay out
    of it: the empty-looking lane is not a lane to drive or swerve into."""
    T = data['truth']
    sel = [n for n, i in enumerate(data['held']) if 'next_to_bike_bus' in data['tags'][i]]
    if len(sel) < MIN_SLICE:
        pytest.skip(f'only {len(sel)} held-out frames next to a bike / bus lane')
    bad = 0
    for n in sel:
        x, y, yaw = data['pose'][n]
        c, s = np.cos(yaw), np.sin(yaw)
        inside = False
        for X, Y in zip(XS, pred[n]):
            wx, wy = x + c * X - s * Y, y + s * X + c * Y
            kind, depth = T.protected_at(wx, wy, yaw)
            if kind in ('bike', 'bus') and depth - CAR_HALF_W > 0.05:   # path centre 5 cm inside
                inside = True
                break
        bad += inside
    frac = bad / len(sel)
    print(f'\n  {bad}/{len(sel)} next-to-bike/bus frames with the path entering that lane ({frac:.1%})')
    assert frac < MAX_PROTECTED_FRAC, f'path enters the bike/bus lane on {frac:.1%} of frames'


def test_flip_symmetry(model, data, pred):
    flipped = _predict(model, data['x'].flip(-1))
    gap = float(np.abs(flipped + pred).mean())
    print(f'\n  mean |pred(flip) + pred| {gap:.3f} m  (limit {FLIP_TOL:.2f})')
    assert gap < FLIP_TOL, f'mirrored image not mirrored prediction: {gap:.3f} m > {FLIP_TOL:.2f} m'


@pytest.mark.parametrize('kind', ['dark', 'bright', 'noise'])
def test_photometric_robustness(model, data, pred, kind):
    x = data['x'].float()
    if kind == 'dark':
        x = x * 0.6
    elif kind == 'bright':
        x = x * 1.4
    else:
        x = x + torch.randn_like(x) * 12.0
    out = _predict(model, x.clamp(0, 255).to(torch.uint8))
    gap = float(np.abs(out - pred).mean())
    print(f'\n  {kind:6s} mean |change| {gap:.3f} m  (limit {PHOTO_TOL:.2f})')
    assert gap < PHOTO_TOL, f'{kind} image moves the path by {gap:.3f} m > {PHOTO_TOL:.2f} m'


def test_temporal_stability(data, pred):
    held = data['held']
    err = pred - data['y'][held]
    fr = np.array(data['frame'])[held]
    ep = np.array(data['episode'])[held]
    pair = (np.diff(fr) == 1) & (np.diff(ep) == 0)
    if pair.sum() < MIN_SLICE:
        pytest.skip('not enough consecutive held-out frames')
    j = STATION['1.5m']
    jump = np.abs(np.diff(err[:, j]))[pair]
    mean = float(jump.mean())
    print(f'\n  {int(pair.sum())} consecutive pairs: mean |change of error| at 1.5 m '
          f'{mean:.3f}  p95 {np.percentile(jump, 95):.3f} m  (limit {JITTER_TOL:.2f})')
    assert mean < JITTER_TOL, f'error jumps {mean:.3f} m between frames > {JITTER_TOL:.2f} m'


def test_latency(model, data):
    from lane_assist.vit_lane import preprocess
    img = next(r for r in data['raw'] if r is not None)
    times = []
    with torch.no_grad():
        for n in range(60):
            t0 = time.perf_counter()
            x = preprocess(img, 221)[None].to(_cuda())
            model(x)
            if _cuda() == 'cuda':
                torch.cuda.synchronize()
            if n >= 10:
                times.append((time.perf_counter() - t0) * 1000)
    p95 = float(np.percentile(times, 95))
    print(f'\n  crop + inference p95 {p95:.1f} ms on {_cuda()}  (limit {LATENCY_MS:.0f})')
    assert p95 < LATENCY_MS, f'inference p95 {p95:.1f} ms > {LATENCY_MS:.0f} ms'
