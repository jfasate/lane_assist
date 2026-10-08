#!/usr/bin/env python3
"""US-style painted road maps for the ViT lane model.

  /usr/bin/python3 make_road_maps.py [n_maps] [first_seed]     (default 48 0)
  /usr/bin/python3 make_road_maps.py 48 0 --maps-only    # only the /map png+yaml, seconds
  /usr/bin/python3 make_road_maps.py 1 23 --relight-noon # re-light one world to noon in place
  /usr/bin/python3 make_road_maps.py --test              # road_test, the evaluation map
  /usr/bin/python3 make_road_maps.py --test-stop         # road_test_stop: held-out stop / debris map
  then: colcon build --packages-select f1tenth_gym_gazebo

Each map `road_NN` is a closed loop whose cross-section changes section by
section, marked per the MUTCD (scaled 1/10 for F1Tenth, right-hand traffic).

Road types (per map)
  two-way   yellow centre: double solid / solid+dashed / dashed / none, or a
            two-way left-turn lane (solid outside + dashed inside, both sides)
  divided   two one-way carriageways split by a grass or concrete median
            (jersey barriers), each with a solid yellow left edge
  one-way   solid yellow left edge, 1-3 lanes
Shapes: smooth blobs, wiggly loops with S-bends, stadiums with long straights.

Per section
  lane lines   dashed / solid / double white, or none
  marking kind painted, raised pavement markers (Botts' dots + reflectors),
               or temporary tabs on fresh pavement
  edges        solid white / none / curb and gutter (no paint)
  side lane    none, paved shoulder (diagonal hatching, rumble strips), bike
               lane (solid white, optional green paint, bike symbols) or
               parking lane (T marks, parked cars)
  lane width   varies section to section (narrow work zones, wide arterials)
  lane add/drop  outer lane tapers in/out behind a wide dotted white line
  lane split   outer lane peels away behind a gore (two solid lines + chevrons)
  lane shift   construction: whole road shifts sideways past cones/barrels,
               old markings blacked out

Things that make lines unclear (random per map)
  worn / faded paint, stretches with paint gone, sun-bleached yellow that reads
  as white, ghost lines of an old layout, paving seams parallel to lane lines,
  tar snakes, cracks, patches, concrete joints, oil, skid marks, puddles, wet
  road, snow with only the wheel tracks clear, autumn leaves, sand drift,
  tree / building / overpass shadows, other cars covering the lines
Distractors that are NOT lane lines
  crosswalks, arrows, STOP / SLOW / ONLY / SCHOOL / BUS / XING / RXR text,
  yield shark teeth, HOV diamonds, bike symbols
Lighting (per world): sun direction and strength, ambient, sky -> noon, dusk,
overcast.

The road is painted into ONE ground texture on a single mesh; Gazebo renders
the very pixels the labels come from.

Writes, per map:
  f1tenth_gym_gazebo/models/road_NN/           model.config/.sdf, meshes/road.{obj,mtl,jpg}
  f1tenth_gym_gazebo/worlds/road_NN.world
  f1tenth_gym_gazebo/maps/road_NN.{yaml,png}   /map: road border black, markings gray
  f1tenth_gym_gazebo/maps/road_NN_preview.png  the texture, 1024 px
  f1tenth_gym_gazebo/maps/generated_safe_maps_spawns.yaml   road_NN line
  lane_assist/tools/maps/truth/road_NN.npz       lane-centre paths = labels

Truth: one path per drivable lane (never bike/parking/shoulder/TWLTL), in its
TRAVEL direction. Where a lane does not exist (dropped / not yet added) its
path runs along the inner neighbour it merges into, so a label is defined
everywhere; `exists` marks where a car may be placed in it, `widths` is that
lane's width there.
"""

import math
import os
import re
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.normpath(os.path.join(HERE, '..', '..', '..', 'f1tenth_gym_gazebo'))
TRUTH = os.path.join(HERE, 'truth')
DS = 0.05                       # reference-line spacing [m]
TEX = 4096                      # ground texture [px]
R_MIN = 7.5                     # tightest reference radius [m]

WHITE = (238.0, 238.0, 232.0)
YELLOW = (245.0, 185.0, 25.0)


# ── Geometry ──────────────────────────────────────────────────────────────────

def ref_loop(rng, force_shape=None):
    """Closed loop resampled at DS with |kappa| <= 1/R_MIN.
    -> (xy, left normals, arc length, shape name)."""
    shape = force_shape or str(rng.choice(['blob', 'wiggly', 'stadium']))
    while True:
        th = np.linspace(0.0, 2 * math.pi, 8000, endpoint=False)
        if shape == 'stadium':                 # superellipse: straights + corners
            a, b = rng.uniform(11.0, 18.0), rng.uniform(8.0, 12.0)
            p = rng.uniform(2.5, 5.0)
            c, s = np.cos(th), np.sin(th)
            xy = np.column_stack([a * np.sign(c) * np.abs(c) ** (2 / p),
                                  b * np.sign(s) * np.abs(s) ** (2 / p)])
        else:
            r = np.full_like(th, rng.uniform(10.0, 15.0))
            ks = (2, 3, 4) if shape == 'blob' else (2, 3, 4, 5, 6)
            amp = 0.14 if shape == 'blob' else 0.30
            for k in ks:
                r *= 1.0 + rng.uniform(0.0, amp / k) * np.cos(k * th + rng.uniform(0, 6.3))
            sx = rng.uniform(1.0, 1.4)
            xy = np.column_stack([sx * r * np.cos(th), r * np.sin(th) / sx])
        xy, nrm, sn, kappa = resample(xy)
        if np.abs(kappa).max() <= 1.0 / R_MIN:
            return xy, nrm, sn, shape


def resample(xy):
    closed = np.vstack([xy, xy[:1]])
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(closed, axis=0).T))])
    sn = np.arange(0.0, s[-1], DS)
    xy = np.column_stack([np.interp(sn, s, closed[:, 0]), np.interp(sn, s, closed[:, 1])])
    d1 = (np.roll(xy, -1, 0) - np.roll(xy, 1, 0)) / (2 * DS)
    d2 = (np.roll(xy, -1, 0) - 2 * xy + np.roll(xy, 1, 0)) / DS ** 2
    sp = np.linalg.norm(d1, axis=1)
    kappa = (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / sp ** 3
    t = d1 / sp[:, None]
    return xy, np.column_stack([-t[:, 1], t[:, 0]]), sn, kappa


def smooth_wrap(a, n):
    """Circular moving average: a step change becomes a linear taper."""
    pad = np.concatenate([a[-n:], a, a[:n]])
    return np.convolve(pad, np.ones(n) / n, mode='same')[n:-n]


def runs(mask):
    """Index arrays of the True runs of a circular boolean array."""
    if mask.all():
        return [np.append(np.arange(mask.size), 0)]
    i = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(np.int8), [0]])))
    out = [np.arange(a, b) for a, b in zip(i[::2], i[1::2])]
    if len(out) > 1 and mask[0] and mask[-1]:
        out = [np.concatenate([out[-1], out[0]])] + out[1:-1]
    return out


# ── Road layout ───────────────────────────────────────────────────────────────

def pick(rng, opts):
    """opts: {value: weight}."""
    k = list(opts)
    p = np.array([opts[x] for x in k], float)
    return k[int(rng.choice(len(k), p=p / p.sum()))]


def plan(rng, N, avoid=False, hard=False, force_road=None):
    """Per-sample widths / offsets / styles along the loop (dict of arrays).
    avoid=True: an obstacle-avoidance map -- always 2+ lanes per direction,
    dashed lane lines, protected bike (green) / bus (red) lanes alongside."""
    road = (pick(rng, {'two_way': 3, 'divided': 3, 'one_way': 4}) if avoid else
            pick(rng, {'two_way': 5, 'divided': 2, 'one_way': 3}))
    road = force_road or road
    w0 = rng.uniform(0.75, 1.05)
    nf_max = 3 if road == 'one_way' else 2
    nb_max = 0 if road == 'one_way' else 2
    m = dict(road=road, w0=w0, nf_max=nf_max, nb_max=nb_max,
             line_w=rng.uniform(0.04, 0.09), dbl_gap=rng.uniform(0.04, 0.08),
             dash_on=rng.uniform(0.3, 0.7),
             median=rng.uniform(0.4, 1.5) if road == 'divided' else 0.0,
             median_kind=str(rng.choice(['grass', 'concrete'])))
    m['dash_off'] = m['dash_on'] * rng.uniform(1.0, 3.0)
    a = dict(wf=np.zeros((nf_max, N)), wb=np.zeros((nb_max, N)), wsec=np.zeros(N),
             tw=np.zeros(N), gore=np.zeros(N), shift=np.zeros(N),
             sf=np.zeros(N), sb=np.zeros(N))
    st = {k: np.empty(N, object) for k in
          ('center', 'lane', 'edge_r', 'edge_l', 'kind', 'side_f', 'side_b', 'green')}
    work = np.zeros(N, bool)
    n_sec, i = 0, 0
    while i < N:
        sl = slice(i, i + int(rng.uniform(10.0, 28.0) / DS))
        ws = w0 * rng.uniform(0.85, 1.15)
        nf = int(rng.integers(1, nf_max + 1))
        nb = int(rng.integers(1, nb_max + 1)) if nb_max else 0
        if avoid:                               # always a lane to pass in
            nf, nb = (nf_max if road != 'one_way' else int(rng.integers(2, 4))), nb_max
        a['wsec'][sl] = ws
        a['wf'][:nf, sl] = ws
        a['wb'][:nb, sl] = ws
        if road == 'two_way' and nf == 1 and nb == 1 and rng.random() < 0.25:
            a['tw'][sl] = ws
        if nf == nf_max and rng.random() < 0.25 and not avoid:
            g = np.zeros(N)
            g[sl] = rng.uniform(0.6, 1.2)
            a['gore'] = np.maximum(a['gore'], g)
        if rng.random() < (0.35 if hard else 0.08 if avoid else 0.15):   # construction lane shift
            a['shift'][sl] = rng.choice([-1, 1]) * rng.uniform(0.3, 0.7)
            work[sl] = True
        unmarked = rng.random() < 0.08 and not avoid
        st['center'][sl] = 'none' if unmarked else pick(rng, {
            'double_yellow': 4, 'solid_dashed': 1.5, 'dashed_solid': 1.5,
            'dashed_yellow': 1.5, 'none': 0.5})
        st['lane'][sl] = 'none' if unmarked else pick(rng, {
            'dashed_white': 6, 'solid_white': 1, 'double_white': 0.5, 'none': 0.5})
        if avoid:                               # changing lanes must be legal
            st['lane'][sl] = 'dashed_white' if rng.random() < 0.9 else 'none'
        st['edge_r'][sl] = 'none' if unmarked else pick(rng, {'solid': 6, 'none': 1, 'curb': 2})
        st['edge_l'][sl] = 'none' if unmarked else pick(rng, {'solid': 6, 'none': 1})
        st['kind'][sl] = pick(rng, {'paint': 7, 'dots': 2, 'tabs': 0.7})
        for side, key in (('side_f', 'sf'), ('side_b', 'sb')):
            sd = (pick(rng, {'bike': 3, 'bus': 2.5, 'parking': 2, 'shoulder': 1, 'none': 1}) if avoid
                  else pick(rng, {'none': 4, 'shoulder': 2, 'bike': 1.5, 'parking': 1}))
            st[side][sl] = sd
            wd = dict(none=0.0, shoulder=rng.uniform(0.3, 0.7),
                      bike=rng.uniform(0.35, 0.5), parking=rng.uniform(0.45, 0.6))
            if avoid:                       # extra draw only here: road_00..47 stay identical
                wd['bus'] = rng.uniform(0.55, 0.75)
            a[key][sl] = wd[sd]
        st['green'][sl] = rng.random() < 0.4 or avoid
        n_sec += 1
        i = sl.stop
    smooth_plan(a, work, int(rng.uniform(3.0, 5.0) / DS))
    m['n_sections'] = n_sec
    return m, a, st


def smooth_plan(a, work, taper):
    """Section steps -> tapers (in place)."""
    for k in ('wf', 'wb'):
        a[k] = np.array([smooth_wrap(smooth_wrap(x, taper), taper) for x in a[k]]).reshape(a[k].shape)
    # Smoothed twice: a single pass makes linear ramps whose corners kink the
    # lane paths (and fold the inner lane of a shifted section back on itself).
    for k in ('wsec', 'tw', 'shift', 'sf', 'sb'):
        a[k] = smooth_wrap(smooth_wrap(a[k], taper), taper)
    # Shrink each gore inward by a taper so it opens and closes inside its
    # section, where the split lane already exists.
    inner = (np.roll(a['gore'], taper) > 0) & (np.roll(a['gore'], -taper) > 0)
    a['gore'] = smooth_wrap(smooth_wrap(np.where(inner, a['gore'], 0.0), taper), taper)
    a['work'] = work


def cross_section(m, a):
    """Lane (inner, outer) edges as lateral offsets, + = left of reference.
    Forward lanes on the reference's right, backward lanes on its left."""
    if m['road'] == 'two_way':
        ef, eb = a['shift'] - a['tw'] / 2, a['shift'] + a['tw'] / 2
    elif m['road'] == 'divided':
        ef, eb = a['shift'] - m['median'] / 2, a['shift'] + m['median'] / 2
    else:
        ef, eb = a['shift'] + m['nf_max'] * m['w0'] / 2, None
    f = []
    for k in range(m['nf_max']):
        if k == m['nf_max'] - 1 and k >= 1:
            ef = ef - a['gore']
        f.append((ef, ef - a['wf'][k]))
        ef = ef - a['wf'][k]
    b = []
    for k in range(m['nb_max']):
        b.append((eb, eb + a['wb'][k]))
        eb = eb + a['wb'][k]
    # side lanes outside the travel lanes: (inner, outer)
    side_f = (ef, ef - a['sf'])
    side_b = (eb, eb + a['sb']) if eb is not None else None
    return f, b, side_f, side_b


def label_paths(m, a, f, b):
    """[(centre offset, exists, width, direction)] per drivable lane. A
    narrowing lane's path blends into its inner neighbour's, so a car in a
    dropped lane is labelled with the lane it has to merge into."""
    out = []
    for widths, edges, d in ((a['wf'], f, 1), (a['wb'], b, -1)):
        prev = None
        for k in range(len(widths)):
            c = 0.5 * (edges[k][0] + edges[k][1])
            al = np.clip(widths[k] / a['wsec'], 0.0, 1.0)
            lab = c if prev is None else al * c + (1 - al) * prev
            out.append((lab, widths[k] >= 0.92 * a['wsec'], widths[k], d))
            prev = lab
    return out


def marking_lines(m, a, st, f, b, side_f, side_b):
    """-> [(offset, on_mask, is_yellow, width_m, pattern)], pattern None
    (solid), ('dash', on, off) or ('dots', spacing)."""
    lw = m['line_w']
    w = a['wsec']
    off = (lw + m['dbl_gap']) / 2
    C, LS, kind = st['center'], st['lane'], st['kind']
    ER, EL = st['edge_r'] == 'solid', st['edge_l'] == 'solid'
    allw = np.vstack([a['wf'], a['wb']])
    taper = ((np.abs(allw - w) > 0.05 * w) & (allw > 0.05 * w)).any(axis=0)
    dots, tabs = kind == 'dots', kind == 'tabs'
    out = []

    def add(o, on, yellow, pat=None, width=lw):
        """Painted line, or the same line as raised markers / temporary tabs
        where the section's marking kind says so."""
        p_on = on & ~dots & ~tabs
        if p_on.any():
            out.append((o, p_on, yellow, width, pat))
        if (on & dots).any():
            out.append((o, on & dots, yellow, 0.03, ('dots', 0.12 if pat is None else 0.25)))
        if (on & tabs).any():
            out.append((o, on & tabs, yellow, 0.03, ('dash', 0.05, 0.95)))

    dash = ('dash', m['dash_on'], m['dash_off'])
    if m['road'] == 'two_way':
        tw = a['tw']
        nt = tw < 0.3 * w
        z = a['shift']
        add(z - off, nt & np.isin(C, ['double_yellow', 'dashed_solid']), True)
        add(z + off, nt & np.isin(C, ['double_yellow', 'solid_dashed']), True)
        add(z - off, nt & (C == 'solid_dashed'), True, dash)
        add(z + off, nt & (C == 'dashed_solid'), True, dash)
        add(z, nt & (C == 'dashed_yellow'), True, dash)
        for sgn in (-1, 1):                      # TWLTL
            add(z + sgn * (tw / 2 + off), ~nt, True)
            add(z + sgn * (tw / 2 - off), ~nt, True, dash)
    else:                                        # left edge of each carriageway
        add(f[0][0] - lw / 2, EL, True)
        if m['road'] == 'divided':
            add(b[0][0] + lw / 2, EL, True)

    for edges, widths in ((f, a['wf']), (b, a['wb'])):
        for k in range(1, len(edges)):
            both = (widths[k] > 0.3 * w) & (widths[k - 1] > 0.5 * w)
            if edges is f and k == m['nf_max'] - 1:
                g = a['gore'] > 0.02
                add(edges[k - 1][1], both & g, False)
                add(edges[k][0], both & g, False)
                both = both & ~g
            o = edges[k][0]
            add(o, both & taper, False, ('dash', 0.15, 0.35), lw * 1.6)
            plain = both & ~taper
            add(o, plain & (LS == 'dashed_white'), False, dash)
            add(o, plain & (LS == 'solid_white'), False)
            add(o - off, plain & (LS == 'double_white'), False)
            add(o + off, plain & (LS == 'double_white'), False)

    # travel-lane outer edge: always a line next to a bike/parking lane
    for side, key, sgn in ((side_f, 'side_f', -1), (side_b, 'side_b', 1)):
        if side is None:
            continue
        sd = st[key]
        has = sd != 'none'
        add(side[0] + sgn * lw / 2, (ER & has) | np.isin(sd, ['bike', 'parking', 'bus']), False)
        add(side[1] - sgn * lw / 2, ER & has & (sd == 'shoulder'), False)
        add(side[0] + sgn * lw / 2, ER & ~has, False)
    return out


# ── Texture ───────────────────────────────────────────────────────────────────

class Canvas:
    def __init__(self, extent):
        self.extent = extent
        self.res = extent / TEX

    def px(self, xy):
        """World (x, y) -> cv2 fixed-point pixel (shift=4), row 0 = +y."""
        xy = np.asarray(xy, float)
        u = (xy[..., 0] + self.extent / 2) / self.res
        v = (self.extent / 2 - xy[..., 1]) / self.res
        return np.round(np.stack([u, v], -1) * 16).astype(np.int32)

    def draw(self, mask, pts_list, width_m=0.0, fill=False, val=255):
        pts = [self.px(p) for p in pts_list if len(p) >= 2]
        if pts:
            if fill:
                cv2.fillPoly(mask, pts, int(val), cv2.LINE_AA, shift=4)
            else:
                cv2.polylines(mask, pts, False, int(val),
                              max(1, int(round(width_m / self.res))), cv2.LINE_AA, shift=4)
        return mask

    def disc(self, mask, c, r_m, val=255):
        u, v = self.px(c)
        cv2.circle(mask, (int(u), int(v)), max(16, int(r_m / self.res * 16)),
                   int(val), -1, cv2.LINE_AA, shift=4)

    def stamp(self, mask, patch, p, t, n, length, width):
        """Lay a glyph patch on the ground at p: patch column 0 on the
        driver's left (+n), row 0 farthest away (+t), so a driver travelling
        along +t reads it the right way round."""
        ph, pw = patch.shape
        src = np.float32([[0, 0], [pw, 0], [0, ph]])
        corners = [p + 0.5 * width * n + 0.5 * length * t,
                   p - 0.5 * width * n + 0.5 * length * t,
                   p + 0.5 * width * n - 0.5 * length * t]
        dst = np.float32([self.px(c) / 16.0 for c in corners])
        warped = cv2.warpAffine(patch, cv2.getAffineTransform(src, dst), (TEX, TEX))
        np.maximum(mask, warped, out=mask)


def noise(rng, cell):
    small = rng.random((TEX // cell + 2, TEX // cell + 2)).astype(np.float32)
    return cv2.resize(small, (TEX, TEX), interpolation=cv2.INTER_CUBIC)[:TEX, :TEX]


def zeros():
    return np.zeros((TEX, TEX), np.uint8)


def glyph(kind, rng):
    """Small binary patch for ground text / symbols."""
    if kind == 'BUS ONLY':                      # MUTCD: read bottom-up, BUS nearest
        a_, b_ = glyph('ONLY', rng), glyph('BUS', rng)
        w_ = max(a_.shape[1], b_.shape[1])
        pad = lambda q: np.pad(q, ((0, 0), (0, w_ - q.shape[1])))     # noqa: E731
        return np.vstack([pad(a_), np.zeros((20, w_), np.uint8), pad(b_)])
    if kind in ('STOP', 'SLOW', 'ONLY', 'SCHOOL', 'BUS', 'XING', 'RXR'):
        txt = 'R X R' if kind == 'RXR' else kind
        (tw_, th_), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 7)
        p = np.zeros((th_ + 20, tw_ + 20), np.uint8)
        cv2.putText(p, txt, (10, th_ + 10), cv2.FONT_HERSHEY_SIMPLEX, 2.0, 255, 7, cv2.LINE_AA)
        if kind == 'RXR':
            cv2.line(p, (0, 0), (p.shape[1], p.shape[0]), 255, 8)
            cv2.line(p, (0, p.shape[0]), (p.shape[1], 0), 255, 8)
        return p
    p = np.zeros((120, 120), np.uint8)
    if kind == 'bike':
        cv2.circle(p, (30, 80), 22, 255, 6)
        cv2.circle(p, (90, 80), 22, 255, 6)
        cv2.polylines(p, [np.array([[30, 80], [55, 40], [90, 80], [60, 80], [55, 40], [75, 35]])],
                      False, 255, 6)
        cv2.circle(p, (60, 18), 9, 255, -1)
    elif kind == 'diamond':
        cv2.polylines(p, [np.array([[60, 2], [118, 60], [60, 118], [2, 60]])], True, 255, 10)
    elif kind == 'teeth':                       # yield line, apex toward driver
        for x in range(0, 120, 30):
            cv2.fillPoly(p, [np.array([[x + 2, 0], [x + 28, 0], [x + 15, 119]])], 255)
    return p


def paint_texture(rng, cv, ref, nrm, sn, m, a, st, f, b, side_f, side_b,
                  lines, left, right, lanes, weather=None):
    img = np.empty((TEX, TEX, 3), np.float32)
    N = len(ref)
    tang = np.column_stack([nrm[:, 1], -nrm[:, 0]])

    def blend(mask, color, alpha=1.0):
        al = mask.astype(np.float32)[..., None] * (alpha / 255.0)
        img[:] = img * (1 - al) + np.asarray(color, np.float32) * al

    def on_road(n=1):
        i = rng.integers(0, N, n)
        return ref[i] + rng.uniform(right[i], left[i])[:, None] * nrm[i], i

    def band(lo, hi, idx=None):
        idx = np.arange(N) if idx is None else np.asarray(idx) % N
        j1 = (idx + 1) % N
        return [np.array([ref[i] + lo[i] * nrm[i], ref[k] + lo[k] * nrm[k],
                          ref[k] + hi[k] * nrm[k], ref[i] + hi[i] * nrm[i]])
                for i, k in zip(idx, j1)]

    drawn = pick(rng, {'clear': 6, 'wet': 1.5, 'snow': 1, 'leaves': 1, 'sand': 0.8})
    weather = weather or drawn                 # forced (road_test) or random; same draws either way

    # ground off the road
    ground = 'snowy' if weather == 'snow' else str(rng.choice(['grass', 'dirt', 'gravel', 'concrete']))
    img[:] = dict(grass=(70, 98, 48), dirt=(122, 100, 74), gravel=(128, 126, 118),
                  concrete=(165, 162, 155), snowy=(215, 218, 225))[ground]
    img += (noise(rng, 64)[..., None] - 0.5) * 40
    img += (rng.random((TEX, TEX, 1), np.float32) - 0.5) * 30

    # pavement
    curb = (st['edge_r'] == 'curb')
    sh = smooth_wrap(np.where(curb, 0.0, rng.uniform(0.1, 0.4)), int(2.0 / DS))
    road = cv.draw(zeros(), band(right - sh, left + sh), fill=True)
    surface = pick(rng, {'new': 1, 'worn': 2, 'old': 1, 'concrete': 1})
    g = dict(new=55, worn=95, old=128, concrete=168)[surface] + rng.uniform(-12, 12)
    tint = np.array([1.0, 1.0, 0.97 if surface == 'concrete' else 1.02], np.float32)
    asph = np.empty_like(img)
    asph[:] = g * tint
    asph += (noise(rng, 400)[..., None] - 0.5) * 30
    asph += (noise(rng, 25)[..., None] - 0.5) * 18
    asph += (rng.random((TEX, TEX, 1), np.float32) - 0.5) * (14 if surface == 'concrete' else 28)
    al = cv2.GaussianBlur(road, (0, 0), 1.5).astype(np.float32)[..., None] / 255
    img[:] = img * (1 - al) + asph * al
    del asph

    # divided-highway median
    if m['road'] == 'divided':
        med = cv.draw(zeros(), band(f[0][0] - 0.0, b[0][0]), fill=True)
        if m['median_kind'] == 'grass':
            blend(med, (70, 98, 48) if weather != 'snow' else (215, 218, 225), 0.95)
        else:
            blend(med, (172, 170, 165), 0.9)

    # curb and gutter: light gutter pan + dark curb face, no paint
    for side, sgn in ((side_f, -1), (side_b, 1)):
        if side is None:
            continue
        idx = np.flatnonzero(curb)
        if idx.size:
            o = side[1]
            pan = cv.draw(zeros(), band(o, o - sgn * 0.08, idx), fill=True)
            blend(pan, (178, 176, 170), 0.9)
            face = cv.draw(zeros(), band(o + sgn * 0.0, o + sgn * 0.03, idx), fill=True)
            blend(face, (60, 60, 58), 0.8)

    # bike lane green paint + symbols, parking T marks, shoulder hatching
    marks = zeros()
    hatch = rng.random() < 0.5
    for side, key in ((side_f, 'side_f'), (side_b, 'side_b')):
        if side is None:
            continue
        sd = st[key]
        green = (sd == 'bike') & st['green'].astype(bool)
        if green.any():
            blend(cv.draw(zeros(), band(side[0], side[1], np.flatnonzero(green)), fill=True),
                  (60, 140, 80), rng.uniform(0.5, 0.85))
        for i in np.flatnonzero(sd == 'bike')[::int(6.0 / DS)]:
            c = ref[i] + 0.5 * (side[0][i] + side[1][i]) * nrm[i]
            wdt = abs(side[0][i] - side[1][i]) * 0.7
            sgn = 1 if key == 'side_f' else -1
            cv.stamp(marks, glyph('bike', rng), c, sgn * tang[i], sgn * nrm[i], wdt * 1.6, wdt)
        bus = sd == 'bus'
        if bus.any():                           # US bus lane: red paint + BUS ONLY
            blend(cv.draw(zeros(), band(side[0], side[1], np.flatnonzero(bus)), fill=True),
                  (165, 40, 35), rng.uniform(0.6, 0.85))
        for i in np.flatnonzero(bus)[::int(8.0 / DS)]:
            c = ref[i] + 0.5 * (side[0][i] + side[1][i]) * nrm[i]
            wdt = abs(side[0][i] - side[1][i]) * 0.75
            sgn = 1 if key == 'side_f' else -1
            cv.stamp(marks, glyph('BUS ONLY', rng), c, sgn * tang[i], sgn * nrm[i], 1.4, wdt)
        for i in np.flatnonzero(sd == 'parking')[::int(0.55 / DS)]:
            o0, o1 = side[0][i], side[1][i]
            cv.draw(marks, [np.array([ref[i] + o0 * nrm[i], ref[i] + o1 * nrm[i]])], 0.03)
        if hatch:
            for i in np.flatnonzero(sd == 'shoulder')[::int(0.4 / DS)]:
                j = (i + int(0.25 / DS)) % N
                cv.draw(marks, [np.array([ref[i] + side[0][i] * nrm[i],
                                          ref[j] + side[1][j] * nrm[j]])], 0.04, val=180)
    k = m['nf_max'] - 1                        # gore chevrons
    for i in np.flatnonzero(a['gore'] > 0.2)[::int(0.5 / DS)]:
        j = (i + int(0.3 / DS)) % N
        cv.draw(marks, [np.array([ref[i] + f[k - 1][1][i] * nrm[i],
                                  ref[j] + f[k][0][j] * nrm[j]])], m['line_w'])
    rumble = zeros()
    if rng.random() < 0.4:
        for side, key in ((side_f, 'side_f'), (side_b, 'side_b')):
            if side is None:
                continue
            for i in np.flatnonzero(st[key] == 'shoulder')[::2]:
                p = ref[i] + (side[0][i] + 0.6 * (side[1][i] - side[0][i])) * nrm[i]
                cv.draw(rumble, [np.array([p - 0.05 * nrm[i], p + 0.05 * nrm[i]])], 0.012)
    blend(rumble, (30, 30, 30), 0.6)

    if surface == 'concrete':                  # transverse joints
        step = int(rng.uniform(0.4, 0.6) / DS)
        segs = [np.array([ref[i] + right[i] * nrm[i], ref[i] + left[i] * nrm[i]])
                for i in range(0, N, step)]
        blend(cv.draw(zeros(), segs, 0.012), (70, 70, 70), 0.6)

    for _ in range(int(rng.integers(0, 12))):  # repair patches
        (c,), (i,) = on_road()
        box = cv2.boxPoints(((0, 0), tuple(rng.uniform(0.3, 1.5, 2)),
                             math.degrees(math.atan2(nrm[i][1], nrm[i][0]))))
        blend(cv.draw(zeros(), [box + c], fill=True), g * tint + rng.uniform(-25, 15), 0.9)

    tar = zeros()                              # tar snakes / cracks / seams
    for _ in range(int(rng.integers(0, 40))):
        (c,), _ = on_road()
        n = int(rng.integers(10, 60))
        walk = c + np.cumsum(rng.normal(0, 0.06, (n, 2)) + rng.normal(0, 0.05, 2), 0)
        cv.draw(tar, [walk], rng.uniform(0.005, 0.03), val=rng.uniform(120, 230))
    if rng.random() < 0.5:                     # paving seam beside a lane line
        o, on = lines[int(rng.integers(0, len(lines)))][:2]
        sft = rng.choice([-1, 1]) * rng.uniform(0.05, 0.2)
        i0 = int(rng.integers(0, N))
        idx = (i0 + np.arange(int(rng.uniform(10, 40) / DS))) % N
        cv.draw(tar, [ref[idx] + (o[idx] + sft)[:, None] * nrm[idx]], 0.012, val=200)
    blend(np.minimum(tar, road), (25, 25, 25))

    oil = zeros()
    for _ in range(int(rng.integers(0, 12))):
        cv.disc(oil, on_road()[0][0], rng.uniform(0.05, 0.25), rng.uniform(80, 180))
    blend(cv2.GaussianBlur(oil, (0, 0), 6), (20, 20, 22))

    skid = zeros()
    for _ in range(int(rng.integers(0, 4))):
        (c,), (i,) = on_road()
        idx = (i + np.arange(int(rng.integers(20, 80)))) % N
        o = (c - ref[i]) @ nrm[i] + np.cumsum(rng.normal(0, 0.004, len(idx)))
        for sd in (-0.08, 0.08):
            cv.draw(skid, [ref[idx] + (o + sd)[:, None] * nrm[idx]], 0.04, val=rng.uniform(80, 150))
    blend(skid, (15, 15, 15))

    # ghosts: an old layout shifted sideways, plus blacked-out lines in work zones
    if rng.random() < 0.35:
        sft = rng.uniform(-0.4, 0.4)
        ghost = [ref[r] + (o[r] + sft)[:, None] * nrm[r] for o, on, *_ in lines for r in runs(on)]
        blend(cv.draw(zeros(), ghost, m['line_w'] * 1.4),
              (200, 200, 195) if rng.random() < 0.5 else (30, 30, 30), rng.uniform(0.08, 0.3))
    if a['work'].any():
        old = [ref[r] + (o[r] - a['shift'][r])[:, None] * nrm[r]
               for o, on, *_ in lines for r in runs(on & a['work'])]
        blend(cv.draw(zeros(), old, m['line_w'] * 1.8), (28, 28, 28), rng.uniform(0.5, 0.85))

    # the markings, with wear, missing stretches and faded yellow
    paint = {False: zeros(), True: zeros()}
    for o, on, yellow, width, pat in lines:
        if pat is not None and pat[0] == 'dash':
            on = on & (np.mod(sn, pat[1] + pat[2]) < pat[1])
        if pat is not None and pat[0] == 'dots':
            step = max(1, int(round(pat[1] / DS)))
            for n_, i in enumerate(np.flatnonzero(on)[::step]):
                c = ref[i] + o[i] * nrm[i]
                if n_ % 4 == 3:                 # retroreflective marker
                    cv.draw(paint[yellow], [np.array([c - 0.02 * nrm[i], c + 0.02 * nrm[i]])], 0.03)
                else:
                    cv.disc(paint[yellow], c, 0.018)
            continue
        cv.draw(paint[yellow], [ref[r] + o[r][:, None] * nrm[r] for r in runs(on)], width)
    amount = np.full((TEX, TEX), rng.uniform(0.6, 1.0), np.float32)
    if rng.random() < 0.6:
        wear = noise(rng, 300) * rng.uniform(0.0, 0.8) + noise(rng, 8) * 0.5
        amount *= np.clip(1 - (wear - rng.uniform(0.2, 0.6)) * 4, 0, 1)
    gone = zeros()
    for _ in range(int(rng.integers(0, 6))):
        i0 = int(rng.integers(0, N))
        cv.draw(gone, band(right - 0.2, left + 0.2, i0 + np.arange(int(rng.uniform(2, 10) / DS))),
                fill=True, val=rng.uniform(150, 255))
    amount *= 1 - gone.astype(np.float32) / 255
    yel = np.array(YELLOW) + rng.uniform(-30, 10, 3)
    if rng.random() < 0.25:                    # sun-bleached: almost white
        yel = 0.5 * yel + 0.5 * np.array(WHITE)
    blend((paint[False] * amount).astype(np.uint8), WHITE)
    blend((paint[True] * amount).astype(np.uint8), yel)
    del paint, amount, gone

    # crosswalks, arrows, text, symbols: road markings that are NOT lane lines
    for _ in range(int(rng.integers(0, 3))):
        i = int(rng.integers(0, N))
        bars = [np.array([ref[i] + o * nrm[i] - 0.2 * tang[i], ref[i] + o * nrm[i] + 0.2 * tang[i]])
                for o in np.arange(right[i] + 0.1, left[i] - 0.1, 0.18)]
        cv.draw(marks, bars, 0.08, val=rng.uniform(130, 255))
    for _ in range(int(rng.integers(0, 14))):
        k = int(rng.integers(0, len(lanes)))
        path, ex = lanes[k][0], lanes[k][1]
        if not ex.any():
            continue
        i = int(rng.choice(np.flatnonzero(ex)))
        p = path[i]
        d = path[(i + 1) % N] - path[i - 1]
        t = d / np.linalg.norm(d)
        n_ = np.array([-t[1], t[0]])
        lw_ = lanes[k][2][i]
        what = pick(rng, {'arrow': 4, 'STOP': 1, 'SLOW': 1, 'ONLY': 1, 'SCHOOL': 0.7,
                          'BUS': 0.5, 'XING': 0.5, 'RXR': 0.5, 'diamond': 0.7, 'teeth': 0.8})
        if what == 'arrow':
            arrow = np.array([p - 0.25 * t + 0.02 * n_, p + 0.05 * t + 0.02 * n_,
                              p + 0.05 * t + 0.07 * n_, p + 0.2 * t, p + 0.05 * t - 0.07 * n_,
                              p + 0.05 * t - 0.02 * n_, p - 0.25 * t - 0.02 * n_])
            cv.draw(marks, [arrow], fill=True, val=rng.uniform(130, 255))
        elif what == 'teeth':
            cv.stamp(marks, glyph(what, rng), p, t, n_, 0.12, 0.8 * lw_)
        elif what == 'diamond':
            cv.stamp(marks, glyph(what, rng), p, t, n_, 0.9, 0.35)
        else:
            cv.stamp(marks, glyph(what, rng), p, t, n_, 0.8, 0.6 * lw_)
    blend(marks, WHITE, rng.uniform(0.6, 1.0))

    # weather
    if weather == 'wet':
        img[:] = img * (1 - al * 0.3)
    wet = zeros()
    for _ in range(int(rng.integers(0, 6 if weather != 'wet' else 25))):
        u, v = cv.px(on_road()[0][0]) // 16
        ax = tuple(int(x / cv.res) for x in rng.uniform(0.1, 0.5, 2))
        cv2.ellipse(wet, (int(u), int(v)), ax, rng.uniform(0, 180), 0, 360,
                    int(rng.uniform(80, 180)), -1)
    blend(cv2.GaussianBlur(wet, (0, 0), 4), (150, 155, 165))
    if weather == 'snow':                      # snow everywhere but the wheel tracks
        tracks = zeros()
        for path, ex, wid, _ in lanes:
            for sgn in (-1, 1):
                d = np.gradient(path, axis=0)
                nn = np.column_stack([-d[:, 1], d[:, 0]]) / np.linalg.norm(d, axis=1)[:, None]
                cv.draw(tracks, [path + sgn * 0.22 * wid[:, None] * nn], 0.12)
        cover = np.clip(noise(rng, 40) * 1.6 - 0.2, 0, 1) * rng.uniform(0.6, 0.95)
        cover *= 1 - cv2.GaussianBlur(tracks, (0, 0), 4).astype(np.float32) / 255 * 0.85
        blend((cover * 255).astype(np.uint8), (222, 226, 232))
    n_deb = dict(leaves=4000, sand=0).get(weather, int(rng.integers(0, 300)))
    for col in ((110, 85, 45), (90, 110, 40), (150, 140, 110), (170, 90, 30)):
        deb = zeros()
        for c in on_road(n_deb // 4)[0]:
            cv.disc(deb, c, rng.uniform(0.005, 0.03), 230)
        blend(deb, col)
    if weather == 'sand':                      # drifts over the road edges
        drift = zeros()
        for side in (left, right):
            cv.draw(drift, band(side - 0.3, side + 0.3), fill=True)
        cover = np.clip(noise(rng, 60) * 1.8 - 0.6, 0, 1) * drift.astype(np.float32) / 255
        blend((cover * 255).astype(np.uint8), (190, 165, 120), 0.9)

    shade = zeros()                            # trees, buildings, overpasses
    for _ in range(int(rng.integers(0, 10))):
        i = int(rng.integers(0, N))
        lat = left[i] + rng.uniform(0, 1) if rng.random() < 0.5 else right[i] - rng.uniform(0, 1)
        for _ in range(int(rng.integers(5, 20))):
            cv.disc(shade, ref[i] + lat * nrm[i] + rng.normal(0, 0.4, 2), rng.uniform(0.1, 0.4))
    for _ in range(int(rng.integers(0, 3))):
        a_ = rng.uniform(1.0, 4.0)
        box = cv2.boxPoints(((0, 0), (a_, a_ * rng.uniform(0.5, 2)), rng.uniform(0, 90)))
        cv.draw(shade, [box + on_road()[0][0]], fill=True)
    for _ in range(int(rng.integers(0, 2))):
        i0 = int(rng.integers(0, N))
        cv.draw(shade, band(right - 2, left + 2, i0 + np.arange(int(rng.uniform(1, 3) / DS))), fill=True)
    s = cv2.GaussianBlur(shade, (0, 0), 6).astype(np.float32) / 255
    img *= (1 - s * rng.uniform(0.3, 0.6))[..., None]

    return np.clip(img, 0, 255).astype(np.uint8), dict(ground=ground, surface=surface,
                                                        weather=weather)


# ── 3D objects ────────────────────────────────────────────────────────────────

def place_objects(rng, m, a, st, ref, nrm, side_f, side_b, lanes, left, right, avoid=False):
    """(kind, x, y, yaw, rgb). Parked cars on shoulders / parking lanes,
    traffic in the lanes, cones and barrels in work zones, median barriers."""
    N = len(ref)
    objs = []

    def heading(i, sgn=1):
        return math.atan2(sgn * -nrm[i][0], sgn * nrm[i][1])

    for side, key, sgn in ((side_f, 'side_f', 1), (side_b, 'side_b', -1)):
        if side is None:
            continue
        cand = np.flatnonzero(np.isin(st[key], ['parking', 'shoulder']))
        for i in rng.choice(cand, min(len(cand), int(rng.integers(0, 6))), replace=False) if len(cand) else []:
            p = ref[i] + 0.5 * (side[0][i] + side[1][i]) * nrm[i]
            objs.append(('car', p[0], p[1], heading(i, sgn), tuple(rng.uniform(0.1, 0.9, 3))))
    for _ in range(0 if avoid else int(rng.integers(0, 6))):     # traffic in the lanes
        path, ex, _, _ = lanes[int(rng.integers(0, len(lanes)))]
        if ex.any():
            i = int(rng.choice(np.flatnonzero(ex)))
            d = path[(i + 1) % N] - path[i - 1]
            objs.append(('car', path[i][0], path[i][1], math.atan2(d[1], d[0]),
                         tuple(rng.uniform(0.1, 0.9, 3))))
    for r in runs(a['work']):                    # work zone: cones along the shift
        o = right + rng.uniform(0.0, 0.2)
        for i in r[::int(0.6 / DS)]:
            p = ref[i] + o[i] * nrm[i]
            objs.append(('cone' if rng.random() < 0.6 else 'barrel', p[0], p[1], 0.0, (1.0, 0.35, 0.0)))
    for _ in range(0 if avoid else int(rng.integers(0, 3))):     # stray cones on an edge
        i0 = int(rng.integers(0, N))
        for j in range(int(rng.integers(3, 8))):
            i = (i0 + j * int(0.6 / DS)) % N
            p = ref[i] + (right[i] + 0.1) * nrm[i]
            objs.append(('cone', p[0], p[1], 0.0, (1.0, 0.35, 0.0)))
    if m['road'] == 'divided' and m['median_kind'] == 'concrete':
        for i in range(0, N, int(1.0 / DS)):
            p = ref[i] + a['shift'][i] * nrm[i]
            objs.append(('barrier', p[0], p[1], heading(i), (0.75, 0.75, 0.72)))
    return objs


GEOM = dict(car=('<box><size>0.45 0.2 0.14</size></box>', 0.07),
            debris=('<box><size>0.25 0.25 0.08</size></box>', 0.04),
            bus=('<box><size>1.0 0.25 0.3</size></box>', 0.15),
            cone=('<cylinder><radius>0.03</radius><length>0.09</length></cylinder>', 0.045),
            barrel=('<cylinder><radius>0.045</radius><length>0.11</length></cylinder>', 0.055),
            barrier=('<box><size>0.98 0.07 0.09</size></box>', 0.045))


def object_sdf(objs, first=0):
    out = []
    for n, (kind, x, y, yaw, col) in enumerate(objs, first):
        geo, z = GEOM[kind]
        c = '%.2f %.2f %.2f 1' % col
        out.append(f"""    <link name='obj_{n}'>
      <pose>{x:.4f} {y:.4f} {z:.3f} 0 0 {yaw:.4f}</pose>
      <visual name='obj_{n}_visual'>
        <geometry>{geo}</geometry>
        <material><ambient>{c}</ambient><diffuse>{c}</diffuse></material>
      </visual>
    </link>""")
    return '\n'.join(out)


def place_blockers(rng, lanes, kinds, spacing=(18.0, 28.0), right_bias=0.4, debris_col=None, keep_away=None):
    """Obstacles that block one lane where another lane of the same direction
    is free to pass in -- one scenario every `spacing` metres per direction.
    kinds: {'car' | 'debris' | 'closure': weight}. right_bias: share placed in
    the rightmost lane, which on avoid maps borders the bike / bus lane, so the
    tempting empty space is a lane the car must not use."""
    out = []
    for grp in ([k for k, l in enumerate(lanes) if l[3] > 0], [k for k, l in enumerate(lanes) if l[3] < 0]):
        if len(grp) < 2:
            continue
        N = len(lanes[grp[0]][0])
        i, done = int(rng.integers(0, N)), 0.0
        while True:
            step = rng.uniform(*spacing)
            done += step
            if done > N * DS - spacing[0]:
                break
            i = (i + int(step / DS)) % N
            win = (i + np.arange(-int(10 / DS), int(8 / DS))) % N
            here = [q for q in grp if lanes[q][1][win].all()]      # lanes present at this spot
            if len(here) < 2:
                continue
            if keep_away is not None and len(keep_away) and \
                    np.hypot(*(keep_away - lanes[grp[0]][0][i]).T).min() < 10.0:
                continue                                         # crash maps: clear of full blocks
            k = here[-1] if rng.random() < right_bias else here[int(rng.integers(len(here)))]
            kind = pick(rng, kinds)
            # sgn: a closure starts from the edge away from the open lane
            out += blocker(rng, lanes[k][0], lanes[k][2], i, kind, 1 if k == here[0] else -1, debris_col)
    return out


def blocker(rng, path, wid, i, kind, sgn, debris_col=None):
    """One obstacle in a lane at path index i: a stopped car, debris, or a
    closure (cone taper across the lane, toward side -sgn, then barrels)."""
    N = len(path)

    def frame(j):
        j %= N
        t = path[(j + 1) % N] - path[j - 1]
        t = t / np.linalg.norm(t)
        return path[j], np.array([-t[1], t[0]]), math.atan2(t[1], t[0])

    p, n_, yaw = frame(i)
    if kind == 'car':
        q = p + n_ * rng.uniform(-0.05, 0.05)
        return [('car', q[0], q[1], yaw + rng.uniform(-0.1, 0.1), tuple(rng.uniform(0.1, 0.9, 3)))]
    if kind == 'debris':
        q = p + n_ * rng.uniform(-0.05, 0.05)
        yaw_d = yaw + rng.uniform(-0.8, 0.8)
        return [('debris', q[0], q[1], yaw_d, debris_col(rng) if debris_col else (0.35, 0.3, 0.25))]
    out = []
    w = float(wid[i % N])
    for c in range(7):
        u = c / 6
        q, nq, _ = frame(i + int((c - 6) * 0.5 / DS))
        q = q + nq * sgn * (0.5 * w - 0.06) * (1 - 2 * u)
        out.append(('cone', q[0], q[1], 0.0, (1.0, 0.35, 0.0)))
    for c in range(1, 4):
        q, nq, _ = frame(i + int(c * 0.6 / DS))
        q = q - nq * sgn * (0.5 * w - 0.06)
        out.append(('barrel', q[0], q[1], 0.0, (1.0, 0.35, 0.0)))
    return out


def side_decoys(rng, ref, nrm, st, side_f, side_b):
    """A stopped bus in each bus lane: obviously not a lane to swerve into."""
    out = []
    for side, key, sgn in ((side_f, 'side_f', 1), (side_b, 'side_b', -1)):
        if side is None:
            continue
        cand = np.flatnonzero(st[key] == 'bus')
        for i in (rng.choice(cand, min(len(cand), 2), replace=False) if len(cand) else []):
            p = ref[i] + 0.5 * (side[0][i] + side[1][i]) * nrm[i]
            yaw = math.atan2(sgn * -nrm[i][0], sgn * nrm[i][1])
            out.append(('bus', p[0], p[1], yaw, (0.8, 0.15, 0.1)))
    return out


# ── Output ────────────────────────────────────────────────────────────────────

# The model is trained for DAYTIME (noon) driving only (decided 2026-10-05):
# every world is lit at noon. pick() still draws once and the per-light values
# are all still drawn, so the random stream -- and therefore every road -- is
# unchanged; only maps that used to be overcast / dusk now get noon light.
DAYLIGHT = {'noon': 1}


def write_world(name, rng, objs, light_w=None):
    tpl = open(os.path.join(SIM, 'worlds', 'superspeedway.world')).read()
    head = tpl[:tpl.index("<model name='superspeedway'>")]
    light = pick(rng, light_w or DAYLIGHT)
    d = dict(noon=rng.uniform(0.7, 1.0), overcast=rng.uniform(0.3, 0.5),
             dusk=rng.uniform(0.35, 0.6))[light]
    amb = dict(noon=rng.uniform(0.3, 0.5), overcast=rng.uniform(0.5, 0.7),
               dusk=rng.uniform(0.15, 0.3))[light]
    el = math.radians(dict(noon=rng.uniform(50, 85), overcast=rng.uniform(40, 80),
                           dusk=rng.uniform(8, 20))[light])
    az = rng.uniform(0, 2 * math.pi)
    sun = np.array([math.cos(az) * math.cos(el), math.sin(az) * math.cos(el), -math.sin(el)])
    warm = rng.uniform(0.75, 0.9) if light == 'dusk' else 1.0
    sky = dict(noon=(0.55, 0.7, 0.9), overcast=(0.7, 0.72, 0.75), dusk=(0.8, 0.55, 0.4))[light]
    head = head.replace('<diffuse>0.8 0.8 0.8 1</diffuse>',
                        f'<diffuse>{d:.2f} {d * (0.5 + warm) / 1.5:.2f} {d * warm:.2f} 1</diffuse>')
    head = head.replace('<direction>-0.5 0.1 -0.9</direction>',
                        '<direction>%.3f %.3f %.3f</direction>' % tuple(sun))
    head = re.sub(r'<ambient>[^<]*</ambient>(\s*<background>)',
                  f'<ambient>{amb:.2f} {amb:.2f} {amb:.2f} 1</ambient>\\1', head, count=1)
    head = re.sub(r'<background>[^<]*</background>',
                  '<background>%.2f %.2f %.2f 1</background>' % sky, head)
    with open(os.path.join(SIM, 'worlds', name + '.world'), 'w') as fh:
        fh.write(head + f"""  <model name='{name}'>
    <static>1</static>
    <pose>0 0 0.001 0 0 0</pose>
    <link name='road'>
      <visual name='road_visual'>
        <cast_shadows>0</cast_shadows>
        <geometry><mesh><uri>model://{name}/meshes/road.obj</uri></mesh></geometry>
      </visual>
    </link>
{object_sdf(objs)}
  </model>
  </world>
</sdf>
""")
    return dict(light=light)


def write_model(name, tex, extent):
    d = os.path.join(SIM, 'models', name, 'meshes')
    os.makedirs(d, exist_ok=True)
    cv2.imwrite(os.path.join(d, 'road.jpg'), tex[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
    h = extent / 2
    with open(os.path.join(d, 'road.mtl'), 'w') as fh:
        fh.write('newmtl road\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nmap_Kd road.jpg\n')
    with open(os.path.join(d, 'road.obj'), 'w') as fh:
        fh.write(f'mtllib road.mtl\nv {-h} {-h} 0\nv {h} {-h} 0\nv {h} {h} 0\nv {-h} {h} 0\n'
                 'vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nvn 0 0 1\n'
                 'usemtl road\nf 1/1/1 2/2/1 3/3/1\nf 1/1/1 3/3/1 4/4/1\n')
    with open(os.path.join(SIM, 'models', name, 'model.config'), 'w') as fh:
        fh.write(f'<?xml version="1.0" ?>\n<model>\n  <name>{name}</name>\n'
                 '  <version>1.0</version>\n  <sdf version="1.7">model.sdf</sdf>\n'
                 '  <description>US-marked road (lane_assist/tools/maps).</description>\n'
                 '</model>\n')
    with open(os.path.join(SIM, 'models', name, 'model.sdf'), 'w') as fh:
        fh.write(f"<?xml version='1.0'?>\n<sdf version='1.7'>\n  <model name='{name}'>\n"
                 "    <static>1</static>\n    <link name='road'>\n      <visual name='road_visual'>\n"
                 f"        <geometry><mesh><uri>model://{name}/meshes/road.obj</uri></mesh></geometry>\n"
                 "      </visual>\n    </link>\n  </model>\n</sdf>\n")


MAP_RES = 0.05
MAP_MARKING = 165       # gray: occupancy ~34 in 'scale' mode, below the plant's
                        # collision threshold (50), so the car drives over it
MAP_EDGE = 0            # black: occupancy 100, the road border acts as a curb


def write_map(name, extent, ref, nrm, sn, lines, left, right, tex=None):
    """/map for RViz and the plant's collision guard: road border black,
    lane markings gray (dashes stay dashed), everything else white/free.
    Written in map_server 'scale' mode -- in the default trinary mode gray
    would be 'unknown', which the plant treats as a wall."""
    n = int(round(extent / MAP_RES))
    img = np.full((n, n), 254, np.uint8)

    def px(xy):
        u = (xy[..., 0] + extent / 2) / MAP_RES
        v = (extent / 2 - xy[..., 1]) / MAP_RES
        return np.round(np.stack([u, v], -1) * 16).astype(np.int32)

    for o, on, yellow, width, pat in lines:
        if pat is not None and pat[0] == 'dash':
            on = on & (np.mod(sn, pat[1] + pat[2]) < pat[1])
        pts = [px(ref[r] + o[r][:, None] * nrm[r]) for r in runs(on) if len(r) >= 2]
        cv2.polylines(img, pts, False, MAP_MARKING, 1, cv2.LINE_8, shift=4)
    # 0.25 m outside the outermost lane edge, where the pavement ends: a car
    # weaving inside its lane never touches the curb, only one leaving the road
    for o in (left + 0.25, right - 0.25):
        cv2.polylines(img, [px(ref + o[:, None] * nrm)], True, MAP_EDGE, 1, cv2.LINE_8, shift=4)
    cv2.imwrite(os.path.join(SIM, 'maps', name + '.png'), img)
    with open(os.path.join(SIM, 'maps', name + '.yaml'), 'w') as fh:
        fh.write(f'image: {name}.png\nmode: scale\nresolution: {MAP_RES:.6f}\n'
                 f'origin: [{-extent / 2:.6f}, {-extent / 2:.6f}, 0.000000]\n'
                 'negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n')
    if tex is not None:
        cv2.imwrite(os.path.join(SIM, 'maps', name + '_preview.png'),
                    cv2.resize(tex, (1024, 1024), interpolation=cv2.INTER_AREA)[:, :, ::-1])


def write_spawn(name, x, y, yaw):
    p = os.path.join(SIM, 'maps', 'generated_safe_maps_spawns.yaml')
    rows = [r for r in open(p).read().splitlines() if r.split(':', 1)[0].strip() != name]
    rows.append(f'{name}: {{x: {x:.4f}, y: {y:.4f}, yaw: {yaw:.5f}}}')
    with open(p, 'w') as fh:
        fh.write('\n'.join(rows) + '\n')


AVOID_FROM = 48          # road_48 and up are obstacle-avoidance maps
# road_27 came out a single-lane one-way loop: no lane to pass an obstacle in,
# so it was regenerated with the avoidance layout (2026-10-05, before any of
# its frames were collected) so every map carries avoidance data.
AVOID_EXTRA = {27}
AVOID_KINDS = {'car': 5, 'debris': 2, 'closure': 3}
# road_72 and up: HARD avoidance maps, made after the 72-map model hit a cone
# closure at dusk that sat next to a row of work-zone edge cones (training had
# 1,805 closure frames, 30 of them at dusk, vs 11,725 stopped-car frames).
# Mostly closures and debris, mostly dusk / overcast, work zones (= rows of
# decoy edge cones) on a third of the sections, closures biased to the lane
# beside them.
HARD_FROM = 72
HARD_KINDS = {'closure': 5, 'debris': 3, 'car': 2}
HARD_LIGHT = DAYLIGHT
# road_84 and up: CONE maps, made after the 84-map model hit the road_69 cone
# closure 4/4 in closed loop. Hard layout (work zones = decoy edge cones),
# mostly closures, one forced weather each, and every ~12 m stretch baked with
# a worst-case condition (faded / missing-ish paint, shadows, glare, tar).
CONE_FROM = 84
CONE_KINDS = {'closure': 6, 'debris': 1, 'car': 1}
CONE_WEATHER = ['clear', 'wet', 'snow', 'leaves', 'sand']
CONE_CONDS = ['faded', 'blur', 'overpass', 'stripes', 'building', 'trees', 'ghost tar', 'glare',
              'faded trees', 'blur building', 'faded stripes', 'tar glare', '']
# road_93 and up: BARE maps, made after the 93-map model on road_test (1) drifted
# across an unpainted centre line into the oncoming lane on a curve, 2/2, and
# (2) never reacted to debris inside an overpass shadow, 3/3. Two-way wiggly
# loops (centre line crossable, curves everywhere), about half of every loop
# with NO paint at all, mostly debris, every debris under its own shadow band.
BARE_FROM = 93
BARE_KINDS = {'debris': 5, 'car': 2, 'closure': 2}
# road_99 and up: CRASH maps, made after the retrained 99-map model hit debris
# 5/5 on road_test and drove into the oncoming lane on faint ghost lines.
# Everything of BARE (two-way wiggly, unpainted stretches -- here ~30% --,
# debris under shadow bands) plus: debris-heavy, debris in many colours (dark
# on dark asphalt, light, tyre-black, tarp blue), ghost lines on most stretches
# with random strength and offset, and FULL ROAD BLOCKS (every lane of one
# direction blocked at one spot) where the only right answer is to stop.
CRASH_FROM = 99
CRASH_KINDS = {'debris': 6, 'closure': 1, 'car': 1}
CRASH_CONDS = ['ghost', 'ghost tar', 'faded ghost', 'ghost stripes', 'blur ghost', 'ghost building',
               'overpass', 'building', 'trees', 'stripes', 'glare', 'faded', '']
CRASH_GHOST = (0.15, 0.5, 0.2, 0.5)      # alpha lo/hi, lateral offset lo/hi [m]
DEBRIS_COLORS = [(0.35, 0.30, 0.25), (0.12, 0.12, 0.12), (0.25, 0.25, 0.27), (0.55, 0.45, 0.30),
                 (0.70, 0.68, 0.62), (0.20, 0.30, 0.55), (0.40, 0.25, 0.15), (0.05, 0.05, 0.05)]
STOP_TEST_SEED = 9100                    # road_test_stop: held-out stop / debris test map
# road_105 and up: BLOCK maps, made after the 105-map model drove AROUND full road
# blocks through the oncoming lane on road_test_stop (only 12 block spots existed
# in training). Crash-map layout with BLOCK_PER_DIR full blocks in ONE direction
# (alternating by map) and passable obstacles in the other: on the same road the
# model sees both answers -- stop (no lane of our direction is free, only the
# oncoming one) and change lanes. More unpainted road.
BLOCK_FROM = 105
BLOCK_PER_DIR = 3


def is_block(seed):
    return BLOCK_FROM <= seed < STOP_TEST_SEED


def debris_color(rng):
    base = np.array(DEBRIS_COLORS[int(rng.integers(len(DEBRIS_COLORS)))])
    return tuple(np.clip(base + rng.uniform(-0.05, 0.05, 3), 0.0, 1.0))


def place_full_blocks(rng, lanes, taken, n_per_dir=2, clear_m=12.0, only_dir=None):
    """Spots where EVERY lane of one direction is blocked at the same station:
    no lane to pass in, the right answer is to stop. Away from other obstacles."""
    out = []
    taken = np.array([o[1:3] for o in taken]).reshape(-1, 2)
    for grp in ([k for k, l in enumerate(lanes) if l[3] > 0], [k for k, l in enumerate(lanes) if l[3] < 0]):
        if not grp or (only_dir is not None and lanes[grp[0]][3] != only_dir):
            continue
        N, done = len(lanes[grp[0]][0]), 0
        for _ in range(300):
            if done >= n_per_dir:
                break
            i = int(rng.integers(N))
            win = (i + np.arange(-int(10 / DS), int(8 / DS))) % N
            p = lanes[grp[0]][0][i]
            if not all(lanes[q][1][win].all() for q in grp) or \
                    (len(taken) and np.hypot(*(taken - p).T).min() < clear_m):
                continue
            for n_, q in enumerate(grp):
                kind = pick(rng, {'debris': 3, 'closure': 2, 'car': 1})
                out += blocker(rng, lanes[q][0], lanes[q][2], i, kind, 1 if n_ == 0 else -1, debris_color)
            taken = np.vstack([taken, p])
            done += 1
    return out


def unmark(rng, st, sn, p=0.5):
    """~Half the loop in 12-30 m runs with no paint at all: no centre, lane or
    edge lines and no marked side lanes (their pavement stays, bare)."""
    bare = np.zeros(len(sn), bool)
    i = 0
    while i < len(sn):
        n = int(rng.uniform(12.0, 30.0) / DS)
        bare[i:i + n] = rng.random() < p
        i += n
    for k in ('center', 'lane', 'edge_r', 'edge_l'):
        st[k][bare] = 'none'
    for k in ('side_f', 'side_b'):
        st[k][bare] = 'none'
    return bare


def geometry(seed):
    """Everything drawn before any texture randomness -> reproducible alone."""
    rng = np.random.default_rng(seed)
    avoid = seed >= AVOID_FROM or seed in AVOID_EXTRA
    hard = seed >= HARD_FROM
    bare = seed >= BARE_FROM
    ref, nrm, sn, shape = ref_loop(rng, 'wiggly' if bare else None)
    m, a, st = plan(rng, len(ref), avoid, hard, 'two_way' if bare else None)
    if bare:
        m['bare_frac'] = float(unmark(np.random.default_rng(seed + 90000), st, sn,
                                      0.4 if is_block(seed) else 0.3 if seed >= CRASH_FROM else 0.5).mean())
    return layout(rng, avoid, hard, ref, nrm, sn, shape, m, a, st)


def layout(rng, avoid, hard, ref, nrm, sn, shape, m, a, st):
    f, b, side_f, side_b = cross_section(m, a)
    lines = marking_lines(m, a, st, f, b, side_f, side_b)
    left = side_b[1] if side_b is not None else f[0][0]
    right = side_f[1]
    lanes = []
    for lab, ex, wid, d in label_paths(m, a, f, b):
        xy = ref + lab[:, None] * nrm
        lanes.append((xy, ex, wid, d) if d > 0 else (xy[::-1], ex[::-1], wid[::-1], d))
    extent = 2 * (np.abs(ref).max() + max(np.abs(left).max(), np.abs(right).max()) + 2.0)
    return dict(rng=rng, avoid=avoid, hard=hard, ref=ref, nrm=nrm, sn=sn, shape=shape, m=m, a=a, st=st,
                f=f, b=b, side_f=side_f, side_b=side_b, lines=lines, left=left, right=right,
                lanes=lanes, extent=extent)


def save_truth(name, g, objs, meta):
    """Lane paths (labels), obstacles and the side lanes (bike / bus /
    parking / shoulder) the car must not drive in."""
    sb = g['side_b']
    np.savez_compressed(
        os.path.join(TRUTH, name + '.npz'),
        paths=np.array([l[0] for l in g['lanes']]), exists=np.array([l[1] for l in g['lanes']]),
        widths=np.array([l[2] for l in g['lanes']]), dirs=np.array([l[3] for l in g['lanes']]),
        objects=np.array([o[1:3] for o in objs if o[0] not in ('barrier', 'bus')]).reshape(-1, 2),
        obj_kind=np.array([o[0] for o in objs], dtype='U8'),
        obj_pose=np.array([o[1:4] for o in objs]).reshape(-1, 3),
        ref=g['ref'], ref_nrm=g['nrm'],
        side_f=np.array(g['side_f']), side_f_type=np.array(g['st']['side_f'], dtype='U8'),
        side_b=np.array(sb) if sb is not None else np.empty(0),
        side_b_type=np.array(g['st']['side_b'], dtype='U8') if sb is not None else np.empty(0),
        meta=repr(meta))


def drop_unavoidable(name, g, objs, meta, candidates):
    """Remove obstacles from `candidates` that block a lane with no free lane
    beside them -- an avoidance map must always have a right answer."""
    from lane_assist.eval_tools import Truth
    for _ in range(4):
        save_truth(name, g, objs, meta)
        bad = {j for _, _, j, tgt in Truth(name).maneuvers() if tgt < 0} & set(candidates)
        if not bad:
            return objs
        keep = [i for i in range(len(objs)) if i not in bad]
        candidates = [keep.index(i) for i in candidates if i in keep]
        objs = [objs[i] for i in keep]
    save_truth(name, g, objs, meta)
    return objs


def make(seed, maps_only=False, name=None):
    name = name or f'road_{seed:02d}'
    old = os.path.join(TRUTH, name + '.npz')
    if not maps_only and os.path.exists(old) and '--force' not in sys.argv:
        d = np.load(old)
        meta = eval(str(d['meta']))
        if int(meta.get('n_orig_obj', len(d['obj_kind']))) < len(d['obj_kind']):
            print(f'{name}: has obstacles added by --add-obstacles; regenerating would drop them '
                  f'(and its collected frames would no longer match). Skipped -- use --force.')
            return
    g = geometry(seed)
    rng, ref, nrm, sn, m, a, st = g['rng'], g['ref'], g['nrm'], g['sn'], g['m'], g['a'], g['st']
    lanes, extent = g['lanes'], g['extent']
    if maps_only:
        write_map(name, extent, ref, nrm, sn, g['lines'], g['left'], g['right'])
        print(f'{name}: map written')
        return
    cv = Canvas(extent)
    cone = seed >= CONE_FROM
    if g['avoid']:
        r2 = np.random.default_rng(seed + 50000)
        crash = seed >= CRASH_FROM
        full = (place_full_blocks(r2, lanes, [], clear_m=14.0 if is_block(seed) else 25.0,
                                  n_per_dir=BLOCK_PER_DIR if is_block(seed) else 2 if seed == STOP_TEST_SEED else 1,
                                  only_dir=(1 if seed % 2 else -1) if is_block(seed) else None)
                if crash else [])
        blk = (place_blockers(r2, lanes, CRASH_KINDS, spacing=(9.0, 14.0), right_bias=0.5, debris_col=debris_color,
                              keep_away=None if is_block(seed) else np.array([o[1:3] for o in full]).reshape(-1, 2))
               if crash
               else place_blockers(r2, lanes, BARE_KINDS, spacing=(13.0, 20.0), right_bias=0.5) if seed >= BARE_FROM
               else place_blockers(r2, lanes, CONE_KINDS, spacing=(13.0, 20.0), right_bias=0.6) if cone
               else place_blockers(r2, lanes, HARD_KINDS, spacing=(13.0, 20.0), right_bias=0.6) if g['hard']
               else place_blockers(r2, lanes, AVOID_KINDS, spacing=(13.0, 20.0)))
    tex, tmeta = paint_texture(rng, cv, ref, nrm, sn, m, a, st, g['f'], g['b'], g['side_f'],
                               g['side_b'], g['lines'], g['left'], g['right'], lanes,
                               weather=CONE_WEATHER[seed % len(CONE_WEATHER)] if cone else None)
    if cone:
        r3 = np.random.default_rng(seed + 80000)
        edges = np.linspace(0, len(ref), int(sn[-1] / 12.0) + 1).astype(int)
        tmeta['conds'] = [str(r3.choice(CRASH_CONDS if seed >= CRASH_FROM else CONE_CONDS)) for _ in edges[1:]]
        shade_at = [int(np.argmin(np.hypot(*(ref - o[1:3]).T))) for o in blk + full if o[0] == 'debris'] \
            if seed >= BARE_FROM else []
        tex = bake_conditions(r3, tex, cv, g, edges, tmeta['conds'], shade_at,
                              CRASH_GHOST if seed >= CRASH_FROM else None)
    objs = place_objects(rng, m, a, st, ref, nrm, g['side_f'], g['side_b'], lanes,
                         g['left'], g['right'], avoid=g['avoid'])
    meta = dict(m, **tmeta, shape=g['shape'], length=float(sn[-1]),
                work_zone=bool(a['work'].any()), avoid=g['avoid'], hard=g['hard'], n_orig_obj=len(objs))
    if g['avoid']:
        n0 = len(objs)
        # full road blocks are impassable ON PURPOSE: never candidates for dropping
        objs = objs + blk + full + side_decoys(r2, ref, nrm, st, g['side_f'], g['side_b'])
        os.makedirs(TRUTH, exist_ok=True)
        objs = drop_unavoidable(name, g, objs, meta, list(range(n0, n0 + len(blk))))
    write_model(name, tex, extent)
    meta.update(write_world(name, rng, objs, HARD_LIGHT if g['hard'] else None))
    write_map(name, extent, ref, nrm, sn, g['lines'], g['left'], g['right'], tex)
    os.makedirs(TRUTH, exist_ok=True)
    save_truth(name, g, objs, meta)
    write_spawn_for(name, lanes, objs)
    kinds = sorted(set(st['kind'])) + sorted(set(st['side_f']) | set(st['side_b'] if m['nb_max'] else []))
    from lane_assist.eval_tools import Truth
    mv = Truth(name).maneuvers()
    print(f'{name}: {m["road"]:8s} {g["shape"]:8s} L={sn[-1]:3.0f}m lanes={len(lanes)} '
          f'sec={m["n_sections"]:2d} {tmeta["surface"]}/{tmeta["ground"]}/{tmeta["weather"]}/'
          f'{meta["light"]} work={int(meta["work_zone"])} obj={len(objs)} '
          f'manoeuvres={sum(t >= 0 for *_, t in mv)} blocked={sum(t < 0 for *_, t in mv)} {",".join(kinds)}')


def write_spawn_for(name, lanes, objs):
    path, ex = lanes[0][0], lanes[0][1]
    ok = ex.copy()
    for o in objs:
        if o[0] not in ('barrier', 'bus'):
            ok &= np.hypot(*(path - np.array(o[1:3])).T) > 8.0
    if not ok.any():
        ok = ex.copy()
    i0 = int(np.flatnonzero(ok)[0])
    p, q = path[i0], path[(i0 + 1) % len(path)]
    write_spawn(name, p[0], p[1], math.atan2(q[1] - p[1], q[0] - p[0]))


def relight_noon(seed):
    """Re-light an existing world to noon in place: only the sun (diffuse,
    direction), scene ambient and sky change -- road texture, obstacles and
    truth geometry stay identical. For maps lit overcast / dusk before the
    daytime-only decision; their frames must then be recollected."""
    name = f'road_{seed:02d}'
    world = os.path.join(SIM, 'worlds', name + '.world')
    txt = open(world).read()
    r = np.random.default_rng(seed + 70000)
    d, amb = r.uniform(0.7, 1.0), r.uniform(0.3, 0.5)
    el, az = math.radians(r.uniform(50, 85)), r.uniform(0, 2 * math.pi)
    sun = (math.cos(az) * math.cos(el), math.sin(az) * math.cos(el), -math.sin(el))
    a0, a1 = txt.index("<light name='sun'"), txt.index('</light>')
    block = txt[a0:a1]
    block = re.sub(r'<diffuse>[^<]*</diffuse>', f'<diffuse>{d:.2f} {d:.2f} {d:.2f} 1</diffuse>', block, count=1)
    block = re.sub(r'<direction>[^<]*</direction>', '<direction>%.3f %.3f %.3f</direction>' % sun, block, count=1)
    txt = txt[:a0] + block + txt[a1:]
    s0, s1 = txt.index('<scene>'), txt.index('</scene>')
    scene = txt[s0:s1]
    scene = re.sub(r'<ambient>[^<]*</ambient>', f'<ambient>{amb:.2f} {amb:.2f} {amb:.2f} 1</ambient>', scene, count=1)
    scene = re.sub(r'<background>[^<]*</background>', '<background>0.55 0.70 0.90 1</background>', scene, count=1)
    txt = txt[:s0] + scene + txt[s1:]
    open(world, 'w').write(txt)
    p = os.path.join(TRUTH, name + '.npz')
    d_ = dict(np.load(p))
    meta = eval(str(d_['meta']))
    old = meta.get('light')
    meta['light'] = 'noon'
    d_['meta'] = np.array(repr(meta))
    np.savez_compressed(p, **d_)
    print(f'{name}: {old} -> noon')


def add_obstacles(seed, max_blockers=4):
    """road_00..47: add 2-4 lane-blocking cars that can be passed, WITHOUT
    repainting. Only the world file (3D objects) and the truth change; the
    texture, map and already-collected frames stay valid. Re-running replaces
    the previously added ones."""
    name = f'road_{seed:02d}'
    g = geometry(seed)
    d = np.load(os.path.join(TRUTH, name + '.npz'))
    meta = eval(str(d['meta']))
    n_orig = int(meta.get('n_orig_obj', len(d['obj_kind'])))
    meta['n_orig_obj'] = n_orig
    world = os.path.join(SIM, 'worlds', name + '.world')
    txt = open(world).read()
    txt = re.sub(r"    <link name='obj_(\d+)'>.*?</link>\n?",
                 lambda mt: mt.group(0) if int(mt.group(1)) < n_orig else '', txt, flags=re.S)
    objs = [(str(k), *map(float, p), (0.5, 0.5, 0.5)) for k, p in zip(d['obj_kind'][:n_orig], d['obj_pose'][:n_orig])]
    from lane_assist.eval_tools import Truth
    base = objs
    for attempt in range(40):                   # until the map has a passable obstacle
        r2 = np.random.default_rng(seed + 60000 + 1000 * attempt)
        sp = (10.0, 16.0) if attempt == 0 and max_blockers <= 4 else (4.0, 8.0)
        blk = place_blockers(r2, g['lanes'], {'car': 1}, spacing=sp, right_bias=0.3)
        if len(blk) > max_blockers:            # at most max_blockers per map, spread out
            blk = [blk[i] for i in sorted(r2.choice(len(blk), max_blockers, replace=False))]
        blk = [(k, x, y, yaw, tuple(r2.uniform(0.1, 0.9, 3))) for k, x, y, yaw, _ in blk]
        objs = drop_unavoidable(name, g, base + blk, meta, list(range(n_orig, n_orig + len(blk))))
        if any(t >= 0 for *_, t in Truth(name).maneuvers()):
            break
    new = objs[n_orig:]
    tail = "\n  </model>\n  </world>"
    assert tail in txt, f'unexpected world layout: {world}'
    txt = txt.replace(tail, ('\n' + object_sdf(new, n_orig) if new else '') + tail)
    open(world, 'w').write(txt)
    save_truth(name, g, objs, meta)
    print(f'{name}: +{len(new)} passable blocking cars ({len(blk) - len(new)} dropped: no free lane)')


# ── road_test: the evaluation map ─────────────────────────────────────────────
# A fixed script, not a random draw: every ~10 m section is one known hard
# condition and carries obstacles that force a lane change, in BOTH directions
# (two-way road, 2 lanes each way, bike / bus / parking lanes outside). Never
# collected for training (collect_all only globs road_NN).
#   cond     faded    paint smeared into the asphalt, barely visible
#            blur     soft-edged lines
#            missing  no markings at all
#            overpass full-width hard shadow: sun -> shadow -> sun
#            stripes  sun / shadow bars every 1.6 m (beams, tree rows)
#            building diagonal shadow edge crossing the lanes
#            trees    dappled shadow spilling in from both edges
#            ghost    old layout 0.35 m off, half visible
#            tar      tar snakes / cracks that look like lines
#            glare    bright washed-out patches
#   blk      (direction f|b, lane in|out, car|debris|closure[, where 0..1])
#            'out' = next to the bike / bus lane: the empty space beside the
#            obstacle is a lane the car must NOT take
TEST_SEED = 9000
TEST_SECTIONS = [
    dict(cond='', side=('bike', 'bus'), blk=[('f', 'out', 'car'), ('b', 'out', 'closure')]),
    dict(cond='faded', side=('bus', 'bike'), center='solid_dashed',
         blk=[('f', 'in', 'closure'), ('b', 'in', 'car')]),
    dict(cond='overpass', side=('parking', 'parking'), blk=[('f', 'out', 'closure'), ('b', 'in', 'debris')]),
    dict(cond='missing', side=('shoulder', 'none'), blk=[('f', 'in', 'car'), ('b', 'out', 'car')]),
    dict(cond='', side=('bike', 'bike'), shift=0.5, w=0.8,          # work zone, edge cones
         blk=[('f', 'out', 'closure'), ('b', 'in', 'closure')]),
    dict(cond='trees', side=('bus', 'bus'), kind='dots', blk=[('f', 'out', 'debris', 0.3), ('b', 'out', 'closure')]),
    dict(cond='', side=('none', 'bike'), drop=True, blk=[('b', 'in', 'car')]),     # right lane ends: merge
    dict(cond='blur building', side=('shoulder', 'bus'), gore=True, blk=[('b', 'in', 'closure')]),  # lane split
    dict(cond='blur building', side=('bike', 'bike'), blk=[('f', 'out', 'car', 0.7), ('b', 'out', 'car')]),
    dict(cond='ghost tar', side=('bus', 'parking'), kind='tabs', w=1.0,
         blk=[('f', 'in', 'closure', 0.85), ('b', 'out', 'debris')]),
    dict(cond='stripes', side=('bike', 'bus'), center='dashed_solid', blk=[('f', 'out', 'closure'), ('b', 'out', 'car')]),
    dict(cond='faded trees tar', side=('bus', 'bike'), blk=[('f', 'in', 'car'), ('b', 'in', 'debris')]),
    dict(cond='glare', side=('bike', 'bus'), w=1.0, blk=[('f', 'out', 'closure'), ('b', 'in', 'car')]),
]
SIDE_W = dict(none=0.0, shoulder=0.5, bike=0.42, bus=0.65, parking=0.5)


def ref_test():
    """Smooth closed loop, ~170 m: ~13 m per section, so one obstacle is
    passed before the next one's lane change has to start."""
    th = np.linspace(0.0, 2 * math.pi, 8000, endpoint=False)
    for amp in (1.0, 0.8, 0.6, 0.4, 0.2):
        r = 26.0 * (1 + amp * (0.10 * np.cos(2 * th + 0.4) + 0.06 * np.cos(3 * th + 1.9)
                               + 0.04 * np.cos(5 * th + 0.7)))
        xy, nrm, sn, kappa = resample(np.column_stack([1.2 * r * np.cos(th), r * np.sin(th) / 1.2]))
        if np.abs(kappa).max() <= 1.0 / R_MIN:
            return xy, nrm, sn
    raise RuntimeError('road_test loop violates R_MIN')


def plan_test(N):
    """TEST_SECTIONS -> the same (m, a, st) plan() returns, + section bounds."""
    m = dict(road='two_way', w0=0.9, nf_max=2, nb_max=2, line_w=0.07, dbl_gap=0.06,
             dash_on=0.5, dash_off=1.0, median=0.0, median_kind='grass', n_sections=len(TEST_SECTIONS))
    a = dict(wf=np.zeros((2, N)), wb=np.zeros((2, N)), wsec=np.zeros(N), tw=np.zeros(N),
             gore=np.zeros(N), shift=np.zeros(N), sf=np.zeros(N), sb=np.zeros(N))
    st = {k: np.empty(N, object) for k in
          ('center', 'lane', 'edge_r', 'edge_l', 'kind', 'side_f', 'side_b', 'green')}
    work = np.zeros(N, bool)
    edges = np.linspace(0, N, len(TEST_SECTIONS) + 1).astype(int)
    for s, i0, i1 in zip(TEST_SECTIONS, edges, edges[1:]):
        sl = slice(i0, i1)
        w = s.get('w', 0.9)
        a['wsec'][sl] = w
        a['wf'][0, sl], a['wf'][1, sl] = w, (0.0 if s.get('drop') else w)
        a['wb'][:, sl] = w
        a['gore'][sl] = 0.9 if s.get('gore') else 0.0
        a['shift'][sl] = s.get('shift', 0.0)
        work[sl] = 'shift' in s
        bare = s['cond'] == 'missing'
        st['center'][sl] = 'none' if bare else s.get('center', 'double_yellow')
        st['lane'][sl] = 'none' if bare else 'dashed_white'
        st['edge_r'][sl] = st['edge_l'][sl] = 'none' if bare else 'solid'
        st['kind'][sl] = s.get('kind', 'paint')
        st['side_f'][sl], st['side_b'][sl] = s['side']
        a['sf'][sl], a['sb'][sl] = SIDE_W[s['side'][0]], SIDE_W[s['side'][1]]
        st['green'][sl] = True
    smooth_plan(a, work, int(4.0 / DS))
    return m, a, st, edges


def bake_conditions(rng, tex, cv, g, edges, conds, shade_at=(), ghost=None):
    """Bake one condition string per stretch [edges[k], edges[k+1]) into the
    painted texture (see the cond list above), plus a hard full-width shadow
    band (3-7 m, sharp sun -> shadow edges) around each reference index in
    shade_at -- the obstacle sits inside it or right at its edge."""
    ref, nrm, left, right, N = g['ref'], g['nrm'], g['left'], g['right'], len(g['ref'])
    img = tex.astype(np.float32)

    def band(lo, hi, idx):
        idx = np.asarray(idx) % N
        return [np.array([ref[i] + lo[i] * nrm[i], ref[k] + lo[k] * nrm[k],
                          ref[k] + hi[k] * nrm[k], ref[i] + hi[i] * nrm[i]])
                for i, k in zip(idx, (idx + 1) % N)]

    def area(lo, hi, idx, soft=1.5):
        m_ = cv.draw(zeros(), band(lo, hi, idx), fill=True)
        return cv2.GaussianBlur(m_, (0, 0), soft).astype(np.float32)[..., None] / 255

    def mix(al, col):
        img[:] = img * (1 - al) + np.asarray(col, np.float32) * al

    smear = cv2.GaussianBlur(img, (0, 0), 14)
    for cond, i0, i1 in zip(conds, edges, edges[1:]):
        c, idx = cond.split(), np.arange(i0, i1)
        road = area(right - 0.1, left + 0.1, idx, 4)
        if 'faded' in c:
            mix(0.85 * road, 0)
            img += smear * 0.85 * road
        if 'blur' in c:
            img[:] = img * (1 - road) + cv2.GaussianBlur(img, (0, 0), 3) * road
        if 'ghost' in c:                         # worn OLD layout: dashed, faint (a ghost, not a line)
            # ghost = (alpha lo, hi, offset lo, hi): random per stretch (training
            # maps); None = road_test's fixed 0.25 / +0.35 m, no extra draws
            g_al, g_off = ((rng.uniform(ghost[0], ghost[1]), rng.choice([-1, 1]) * rng.uniform(ghost[2], ghost[3]))
                           if ghost else (0.25, 0.35))
            old = []
            dash = g['m']['dash_on'] + g['m']['dash_off']
            for o, on, _, _, pat in g['lines']:
                on = np.isin(np.arange(N), idx) & on
                if pat is not None:
                    on = on & (np.mod(g['sn'], dash) < g['m']['dash_on'])
                old += [ref[r] + (o[r] + g_off)[:, None] * nrm[r] for r in runs(on)]
            mix(g_al * cv.draw(zeros(), old, g['m']['line_w']).astype(np.float32)[..., None] / 255, WHITE)
        if 'tar' in c:
            tar = zeros()
            for _ in range(25):
                i = int(rng.choice(idx))
                p = ref[i] + rng.uniform(right[i], left[i]) * nrm[i]
                t = np.array([nrm[i][1], -nrm[i][0]])
                walk = p + np.cumsum(rng.normal(0, 0.02, (60, 2)) + 0.05 * t, 0)    # roughly along the lane
                cv.draw(tar, [walk], rng.uniform(0.01, 0.03))
            mix(tar.astype(np.float32)[..., None] / 255, (22, 22, 22))
        if 'glare' in c:
            gl = zeros()
            for _ in range(6):
                i = int(rng.choice(idx))
                cv.disc(gl, ref[i] + rng.uniform(right[i], left[i]) * nrm[i], rng.uniform(0.4, 0.9))
            mix(0.6 * cv2.GaussianBlur(gl, (0, 0), 25).astype(np.float32)[..., None] / 255, (250, 250, 245))
        shade = np.zeros((TEX, TEX, 1), np.float32)
        if 'overpass' in c:
            mid = (i0 + i1) // 2
            shade = np.maximum(shade, area(right - 3, left + 3, mid + np.arange(-int(2.0 / DS), int(2.0 / DS)), 1))
        if 'stripes' in c:
            for j in range(i0, i1, int(1.6 / DS)):
                shade = np.maximum(shade, area(right - 3, left + 3, j + np.arange(int(0.7 / DS)), 1))
        if 'building' in c:
            hi = right.copy()
            hi[idx] = right[idx] + (left[idx] - right[idx]) * np.linspace(0.15, 0.85, len(idx))
            shade = np.maximum(shade, area(right - 3, hi, idx, 1))
        if 'trees' in c:
            tr = zeros()
            for i in idx[::int(0.8 / DS)]:
                for edge, sgn in ((left, 1), (right, -1)):
                    for _ in range(6):
                        cv.disc(tr, ref[i] + (edge[i] + sgn * 0.3 - sgn * abs(rng.normal(0, 0.5))) * nrm[i]
                                + rng.normal(0, 0.3, 2), rng.uniform(0.1, 0.35))
            shade = np.maximum(shade, cv2.GaussianBlur(tr, (0, 0), 5).astype(np.float32)[..., None] / 255)
        img *= 1 - 0.6 * shade
    for i in shade_at:
        a0 = int(rng.uniform(-3.0, 0.5) / DS)
        sh = area(right - 3, left + 3, i + a0 + np.arange(int(rng.uniform(3.0, 7.0) / DS)), 1)
        img *= 1 - rng.uniform(0.5, 0.7) * sh
    return np.clip(img, 0, 255).astype(np.uint8)


def make_test():
    global TEX
    TEX = 6144                  # bigger loop, same ~1.3 cm/px paint sharpness as the training maps
    name = 'road_test'
    rng = np.random.default_rng(TEST_SEED)
    ref, nrm, sn = ref_test()
    m, a, st, edges = plan_test(len(ref))
    g = layout(rng, True, False, ref, nrm, sn, 'test', m, a, st)
    lanes = g['lanes']
    cv = Canvas(g['extent'])
    tex, tmeta = paint_texture(rng, cv, ref, nrm, sn, m, a, st, g['f'], g['b'], g['side_f'], g['side_b'],
                               g['lines'], g['left'], g['right'], lanes, weather='clear')
    tex = bake_conditions(rng, tex, cv, g, edges, [s['cond'] for s in TEST_SECTIONS])
    objs = place_objects(rng, m, a, st, ref, nrm, g['side_f'], g['side_b'], lanes, g['left'], g['right'], avoid=True)
    blk = []
    for s, i0, i1 in zip(TEST_SECTIONS, edges, edges[1:]):
        for d, lane, kind, *where in s.get('blk', []):
            i = int(i0 + (where[0] if where else 0.6) * (i1 - i0))
            k = (0 if d == 'f' else 2) + (lane == 'out')
            blk += blocker(rng, lanes[k][0], lanes[k][2], i if d == 'f' else len(ref) - 1 - i,
                           kind, 1 if lane == 'in' else -1)       # backward lane paths run reversed
    meta = dict(m, **tmeta, shape='test', length=float(sn[-1]), work_zone=bool(a['work'].any()),
                avoid=True, hard=False, n_orig_obj=len(objs), sections=[s['cond'] or 'clean' for s in TEST_SECTIONS])
    n0 = len(objs)
    dec = side_decoys(rng, ref, nrm, st, g['side_f'], g['side_b'])
    objs = objs + blk + dec
    os.makedirs(TRUTH, exist_ok=True)
    objs = drop_unavoidable(name, g, objs, meta, list(range(n0, n0 + len(blk))))
    write_model(name, tex, g['extent'])
    meta.update(write_world(name, rng, objs))
    write_map(name, g['extent'], ref, nrm, sn, g['lines'], g['left'], g['right'], tex)
    save_truth(name, g, objs, meta)
    write_spawn_for(name, lanes, objs)
    from lane_assist.eval_tools import Truth
    mv = Truth(name).maneuvers()
    print(f'{name}: L={sn[-1]:.0f} m, {len(TEST_SECTIONS)} sections, {len(objs)} objects, '
          f'{sum(t >= 0 for *_, t in mv)} lane changes forced, {sum(t < 0 for *_, t in mv)} impassable '
          f'(blocker objects kept {len(objs) - n0 - len(dec)} of {len(blk)})')


if __name__ == '__main__':
    sys.path.insert(0, os.path.normpath(os.path.join(HERE, '..', '..')))
    if '--test-stop' in sys.argv:
        make(STOP_TEST_SEED, name='road_test_stop')
        sys.exit()
    if '--test' in sys.argv:
        make_test()
        sys.exit()
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    n = int(args[0]) if args else 48
    first = int(args[1]) if len(args) > 1 else 0
    for seed in range(first, first + n):
        if '--relight-noon' in sys.argv:
            relight_noon(seed)
        elif '--add-obstacles' in sys.argv:
            mb = [int(a.split('=')[1]) for a in sys.argv if a.startswith('--max-blockers=')]
            add_obstacles(seed, mb[0] if mb else 4)
        else:
            make(seed, maps_only='--maps-only' in sys.argv)
