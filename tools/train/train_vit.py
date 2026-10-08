#!/usr/bin/env python3
"""Fine-tune the ViT lane model on dataset/<map>_labels.csv.

  /usr/bin/python3 train_vit.py road_00 [road_01 ...] [--epochs 30] [--bs 64]
  /usr/bin/python3 train_vit.py $(ls ../dataset/raw | grep road_) \
      --test-maps road_02 road_05 road_11 road_23 road_33 road_45 --epochs 15

Splits: --test-maps are held out WHOLE (roads never seen in training). For the
other maps make_dataset.py's held-out track sections become validation. With
no --test-maps (one map), its test sections are the test set.
Lane-change frames are drawn --lc-weight times as often as the rest, because
avoidance is a few % of the driving but the case that matters.

Images are cropped below the horizon (vit_lane.crop_resize) once and cached
per map in dataset/cache/<map>.npy, read batch by batch (50k frames do not fit
in RAM or GPU memory); augmentation runs on the GPU per batch:
  left/right flip (labels negated), brightness / contrast / saturation,
  sensor noise, blur, random erasing (glare, occlusion).
No geometric shifts or rotations: the labels are metric positions in the
car frame, and moving the image would make them wrong.

Reports mean |error| [m] at the car (X=0), at 1.5 m and at 3 m, next to the
predict-the-train-mean baseline -- a model that does not clearly beat it has
collapsed to the mean. Keeps the best checkpoint by val error, then scores it
once on test and draws truth (green) vs prediction (red) on test frames.

Writes:
  models/vit_lane_<tag>.pt            best checkpoint + metrics (tag: map name,
                                      or <N>maps)
  dataset/check/<tag>_pred/*.jpg      test frames, truth vs prediction
"""

import argparse
import csv
import hashlib
import math
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.normpath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, PKG)
from lane_assist.camera_geometry import pixel_from_ground                # noqa: E402
from lane_assist.eval_tools import FREE_MAX                              # noqa: E402
from lane_assist.vit_lane import FREE_SCALE, XS, IMG_H, IMG_W, build_lane_model, crop_resize   # noqa: E402

DATA = os.path.join(PKG, 'dataset')
NOUT = len(XS) + 1                  # 16 path offsets + scaled free distance (stop head)
STATIONS = {'car': 0, '1.5m': int(np.argmin(np.abs(XS - 1.5))), f'{XS[-1]:.0f}m': len(XS) - 1}


def crops(name, rows):
    """(n, H, W, 3) uint8 memmap of the cropped frames of one map's label rows,
    built once and rebuilt only when the frames themselves change (image list
    and crop row). Relabelling alone -- make_dataset rewrites every labels
    file -- does not change a crop, so it keeps the cache."""
    path = os.path.join(DATA, 'cache', name + '.npy')
    key = hashlib.md5(''.join(f"{r['file']},{r['top']}\n" for r in rows).encode()).hexdigest()
    if os.path.exists(path) and os.path.exists(path + '.key') and open(path + '.key').read() == key:
        return np.load(path, mmap_mode='r')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mm = np.lib.format.open_memmap(path + '.tmp.npy', 'w+', np.uint8, (len(rows), IMG_H, IMG_W, 3))
    for j, r in enumerate(rows):
        mm[j] = crop_resize(cv2.imread(os.path.join(DATA, r['file']))[:, :, ::-1], int(r['top']))
    mm.flush()
    del mm
    os.replace(path + '.tmp.npy', path)
    with open(path + '.key', 'w') as fh:
        fh.write(key)
    return np.load(path, mmap_mode='r')


class Split:
    """Frames of one split, spread over per-map memmaps."""

    def __init__(self):
        self.src, self.y, self.lc, self.wrong, self.rows = [], [], [], [], []

    def finish(self, mms):
        self.mms = mms
        self.src = np.array(self.src, dtype=np.int64).reshape(-1, 2)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32).reshape(-1, NOUT).cuda()
        self.lc = torch.tensor(self.lc, dtype=torch.bool).cuda()
        self.wrong = torch.tensor(self.wrong, dtype=torch.bool).cuda()
        self.stop = self.y[:, len(XS)] < (FREE_MAX - 0.01) * FREE_SCALE
        return self

    def __len__(self):
        return len(self.src)

    def images(self, idx):
        """uint8 (B, 3, H, W) on the GPU for indices idx."""
        idx = np.asarray(idx)
        out = np.empty((len(idx), IMG_H, IMG_W, 3), np.uint8)
        for n, (m, j) in enumerate(self.src[idx]):
            out[n] = self.mms[m][j]
        return torch.from_numpy(out).cuda(non_blocking=True).permute(0, 3, 1, 2)


def load(maps, test_maps, all_data=False):
    out = {s: Split() for s in ('train', 'val', 'test')}
    mms = []
    for name in maps:
        rows = list(csv.DictReader(open(os.path.join(DATA, name + '_labels.csv'))))
        t0 = time.time()
        mms.append(crops(name, rows))
        if time.time() - t0 > 2:
            print(f'  cached {name}: {len(rows)} frames in {time.time() - t0:.0f}s', flush=True)
        for j, r in enumerate(rows):
            if all_data:
                split = 'train'                         # final model: every frame, incl. gap
            elif name in test_maps:
                split = 'test'                          # the whole road is unseen
            elif r['split'] == 'test' and test_maps:
                split = 'val'                           # sections only validate
            else:
                split = r['split']
            if split not in out:
                continue
            d = out[split]
            d.src.append((len(mms) - 1, j))
            d.y.append([float(r[f'y{k:02d}']) for k in range(len(XS))]
                       + [float(r.get('free') or FREE_MAX) * FREE_SCALE])
            d.lc.append(int(r['lane_change']))
            d.wrong.append(int(r.get('wrong') or 0))
            d.rows.append(r)
    return {s: d.finish(mms) for s, d in out.items()}


def to_input(u8):
    return u8.float() / 127.5 - 1.0


def augment(x, y, flip=True):
    """x: (B,3,H,W) float in [-1,1]; y: (B,17) = 16 path offsets + scaled free
    distance (a mirror flips the path, not the distance). GPU, per sample."""
    B = x.shape[0]
    dev = x.device
    if flip:
        f = torch.rand(B, device=dev) < 0.5
        x = torch.where(f[:, None, None, None], x.flip(-1), x)
        sign = torch.ones(y.shape[1], device=dev)
        sign[:len(XS)] = -1.0
        y = torch.where(f[:, None], y * sign, y)
    x = (x + 1) / 2                                            # [0,1]
    x = x * torch.empty(B, 1, 1, 1, device=dev).uniform_(0.45, 1.6)           # brightness
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - m) * torch.empty(B, 1, 1, 1, device=dev).uniform_(0.6, 1.4) + m  # contrast
    g = x.mean(dim=1, keepdim=True)
    x = (x - g) * torch.empty(B, 1, 1, 1, device=dev).uniform_(0.5, 1.5) + g  # saturation
    x = x + torch.randn_like(x) * torch.empty(B, 1, 1, 1, device=dev).uniform_(0, 0.04)
    blur = torch.rand(B, device=dev) < 0.25
    if blur.any():
        xb = F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode='replicate'), 3, 1)
        x = torch.where(blur[:, None, None, None], xb, x)
    for _ in range(2):                                         # random erasing
        hit = torch.rand(B, device=dev) < 0.3
        for i in torch.nonzero(hit).flatten().tolist():
            h = int(np.random.uniform(0.1, 0.4) * IMG_H)
            w = int(np.random.uniform(0.05, 0.25) * IMG_W)
            r0 = np.random.randint(0, IMG_H - h)
            c0 = np.random.randint(0, IMG_W - w)
            x[i, :, r0:r0 + h, c0:c0 + w] = torch.rand(3, 1, 1, device=dev)
    return x.clamp(0, 1) * 2 - 1, y


@torch.no_grad()
def predict(model, split, bs=256):
    model.eval()
    out = []
    for i in range(0, len(split), bs):
        with torch.autocast('cuda', dtype=torch.float16):
            out.append(model(to_input(split.images(np.arange(i, min(i + bs, len(split)))))).float())
    return torch.cat(out) if out else torch.empty(0, NOUT, device='cuda')


def report(name, pred, y, lc):
    e = (pred[:, :len(XS)] - y[:, :len(XS)]).abs()
    s = '  '.join(f'{k} {e[:, j].mean():.3f}' for k, j in STATIONS.items())
    s = f'{name:9s} mean {e.mean():.3f}  {s}'
    if lc.any():
        s += f'  | lane-change mean {e[lc].mean():.3f} ({int(lc.sum())})'
    stop = y[:, len(XS)] < (FREE_MAX - 0.01) * FREE_SCALE
    if stop.any():
        ef = (pred[stop, len(XS)] - y[stop, len(XS)]).abs() / FREE_SCALE
        s += f'  | stop frames free-dist err {ef.mean():.2f} m ({int(stop.sum())})'
    return s, float(e.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('maps', nargs='+')
    ap.add_argument('--test-maps', nargs='*', default=[], help='maps held out whole as the test set')
    ap.add_argument('--lc-weight', type=float, default=5.0, help='sampling weight of lane-change frames')
    ap.add_argument('--stop-weight', type=float, default=20.0,
                    help='sampling weight of stop frames (no lane to pass): without it the 17th output '
                         'collapsed to "always clear" (1.2%% of frames, 2026-10-07)')
    ap.add_argument('--wrong-weight', type=float, default=10.0,
                    help='sampling weight of frames on the wrong side of the road (the way back)')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--wd', type=float, default=0.05)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--no-flip', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--all', action='store_true',
                    help='final model: train on EVERY frame of the given maps (no val/test); keeps '
                         'the last epoch. Measure first with a held-out run, then train this.')
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    maps = list(dict.fromkeys(args.maps + args.test_maps))
    tag = maps[0] if len(maps) == 1 else '_'.join(maps) if len(maps) <= 3 else f'{len(maps)}maps'
    if args.all:
        tag += '_all'
        args.test_maps = []

    t0 = time.time()
    data = load(maps, set(args.test_maps), args.all)
    tr, va, te = data['train'], data['val'], data['test']
    print(f'loaded train {len(tr)} ({int(tr.lc.sum())} lane-change, {int(tr.stop.sum())} stop, '
          f'{int(tr.wrong.sum())} wrong-side)  val {len(va)}  test {len(te)} '
          f'frames in {time.time() - t0:.0f}s  test maps: {args.test_maps or "-"}')
    weight = torch.where(tr.lc, args.lc_weight, 1.0)
    weight = torch.where(tr.wrong, torch.maximum(weight, torch.tensor(args.wrong_weight, device='cuda')), weight)
    weight = torch.where(tr.stop, torch.maximum(weight, torch.tensor(args.stop_weight, device='cuda')), weight)

    model = build_lane_model(dropout=args.dropout, n_out=NOUT).cuda()
    mean = tr.y.mean(0)
    with torch.no_grad():
        model.head.bias.copy_(mean)        # start as the mean predictor, learn from there
    head = [p for n, p in model.named_parameters() if n.startswith('head.')]
    no_wd = [p for n, p in model.named_parameters()
             if not n.startswith('head.') and (p.ndim == 1 or 'posembed' in n or n == 'cls')]
    rest = [p for n, p in model.named_parameters()
            if not n.startswith('head.') and not (p.ndim == 1 or 'posembed' in n or n == 'cls')]
    opt = torch.optim.AdamW([dict(params=rest, weight_decay=args.wd),
                             dict(params=no_wd, weight_decay=0.0),
                             dict(params=head, lr=args.lr * 10, weight_decay=0.0)], lr=args.lr)
    steps = args.epochs * math.ceil(len(tr) / args.bs)
    warm = math.ceil(len(tr) / args.bs)
    base = [g['lr'] for g in opt.param_groups]
    scaler = torch.amp.GradScaler('cuda')

    for name, d in (('val', va), ('test', te)):
        if len(d):
            print(report(f'{name} base', mean.expand_as(d.y), d.y, d.lc)[0],
                  '<- predict-the-train-mean')
    os.makedirs(os.path.join(PKG, 'models'), exist_ok=True)
    ckpt = os.path.join(PKG, 'models', f'vit_lane_{tag}.pt')
    best, step = 1e9, 0
    for ep in range(args.epochs):
        model.train()
        perm = torch.multinomial(weight, len(tr), replacement=True)
        tot, n = 0.0, 0
        t0 = time.time()
        for b in range(0, len(perm), args.bs):
            idx = perm[b:b + args.bs]
            x, y = augment(to_input(tr.images(idx.cpu().numpy())), tr.y[idx], flip=not args.no_flip)
            k = min(1.0, (step + 1) / warm) * 0.5 * (1 + math.cos(math.pi * step / steps))
            for g, lr0 in zip(opt.param_groups, base):
                g['lr'] = lr0 * k
            with torch.autocast('cuda', dtype=torch.float16):
                loss = F.smooth_l1_loss(model(x).float(), y, beta=0.05)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            tot += loss.item() * len(idx)
            n += len(idx)
            step += 1
        if len(va):
            line, err = report('val', predict(model, va), va.y, va.lc)
        else:                                       # --all: nothing held out, keep the last epoch
            line, err = 'no validation (--all)', -float(ep)
        mark = ''
        if err < best:
            best = err
            torch.save(dict(state_dict=model.state_dict(), xs=XS, img=(IMG_H, IMG_W), n_out=NOUT,
                            free_scale=FREE_SCALE, free_max=FREE_MAX,
                            model='ViT-Ti/16 augreg i21k->in1k', maps=maps,
                            test_maps=args.test_maps, epoch=ep,
                            val_mae=err if len(va) else float('nan'), args=vars(args)), ckpt)
            mark = '  *saved'
        print(f'ep {ep + 1:2d}/{args.epochs} loss {tot / n:.4f} {time.time() - t0:4.1f}s | {line}{mark}')

    model.load_state_dict(torch.load(ckpt, weights_only=False)['state_dict'])
    print(f'\n{"final epoch" if args.all else f"best val mean {best:.3f} m"} -> {ckpt}')
    if not len(te):
        return
    pred = predict(model, te)
    line, err = report('test', pred, te.y, te.lc)
    print(line)
    te_map = np.array([r['map'] for r in te.rows])
    for name in dict.fromkeys(te_map):
        k = torch.from_numpy(np.flatnonzero(te_map == name)).cuda()
        print('  ' + report(name, pred[k], te.y[k], te.lc[k])[0])
    c = torch.load(ckpt, weights_only=False)
    c['test_mae'] = err
    torch.save(c, ckpt)

    # truth (green) vs prediction (red) drawn on test frames
    cam = yaml.safe_load(open(os.path.join(PKG, 'config', 'lane_assist_params.yaml')))
    cam = cam['vit_lane_node']['ros__parameters']
    geo = (float(cam['cam_height']), float(cam['cam_pitch']), float(cam['cam_x']))
    chk = os.path.join(DATA, 'check', f'{tag}_pred')
    os.makedirs(chk, exist_ok=True)
    for f in os.listdir(chk):
        os.remove(os.path.join(chk, f))
    rng = np.random.default_rng(0)
    pick = rng.choice(len(te), min(30, len(te)), replace=False)
    lcs = torch.nonzero(te.lc).flatten().cpu().numpy()
    lcs = rng.choice(lcs, min(15, len(lcs)), replace=False) if len(lcs) else []
    for i in sorted(set(pick.tolist()) | set(int(v) for v in lcs)):
        r = te.rows[i]
        img = cv2.imread(os.path.join(DATA, r['file']))
        fx, fy, cx, cy = map(float, open(os.path.join(DATA, 'raw', r['map'], 'camera.txt')).read().split()[:4])
        for Y, col in ((te.y[i, :len(XS)], (0, 255, 0)), (pred[i, :len(XS)], (0, 0, 255))):
            Xd = np.linspace(XS[0], XS[-1], 60)
            u, v = pixel_from_ground(Xd, np.interp(Xd, XS, Y.cpu().numpy()), (fx, fy, cx, cy), *geo)
            ok = np.isfinite(u) & np.isfinite(v) & (Xd > geo[2] + 0.05)   # in front of the camera
            cv2.polylines(img, [np.stack([u[ok], v[ok]], 1).astype(np.int32)], False, col, 3, cv2.LINE_AA)
        e = (pred[i, :len(XS)] - te.y[i, :len(XS)]).abs()
        cv2.putText(img, f'{r["map"]}  err car {e[0]:.2f}  {XS[-1]:.0f}m {e[-1]:.2f} m{"  LANE CHANGE" if te.lc[i] else ""}',
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imwrite(os.path.join(chk, f"{r['map']}_{os.path.basename(r['file'])}"), img)
    print(f'test overlays (green truth, red prediction): {chk}')


if __name__ == '__main__':
    main()
