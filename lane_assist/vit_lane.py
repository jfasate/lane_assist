#!/usr/bin/env python3
"""ViT lane model: one camera frame -> lane-centre Y at fixed X stations.

PyTorch port of google-research/vision_transformer `vit_jax/models_vit.py`
(Dosovitskiy et al., "An Image is Worth 16x16 Words", arXiv:2010.11929).
Module and attribute names mirror the Flax ones (`Transformer.encoderblock_3.
MultiHeadDotProductAttention_1.query`, ...), so the official `.npz`
checkpoints load by renaming '/' to '.' — see load_official().

Pretrained weights: the official ViT-Ti/16 augreg checkpoint, i21k pretrain ->
in1k fine-tune at 384 px, from gs://vit_models/augreg:
  Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz
saved as models/Ti_16-i21k-in1k-384.npz. Exactly the paper's recipe: pretrain
large, fine-tune at a new resolution with the position embeddings resampled
and a fresh zero-initialised head.

Lane task (the only changes vs the classifier):
  input   the frame cropped below the horizon row, resized to 128 x 384
          (8 x 24 = 192 patches), scaled to [-1, 1] like the official pipeline
  output  16 numbers = the path to drive, Y [m, +left] in base_link at XS
          (+ optional 17th: free distance ahead * FREE_SCALE, see below),
          0..5 m: the lane centre, or the lane change around an obstacle.
          5 m (not 3) so an avoidance can be a smooth ~5.5 m lane change; the
          lane lines past ~3 m are faint, but obstacles are clear to ~7 m.
          X=0 is included on purpose: the camera cannot see the lane lines
          nearer than ~0.72 m, and the classic detector has to extrapolate
          there; here it is a supervised target.

Self-check (no ROS):  python3 vit_lane.py --selfcheck
"""

import math
import os
import sys

import numpy as np
import torch
from torch import nn

XS = np.linspace(0.0, 5.0, 16)
# Optional 17th output: free distance ahead [m] (eval_tools.free_distance,
# 0..FREE_MAX) scaled by FREE_SCALE so its range matches the path offsets in
# the loss. A checkpoint has it when its head has len(XS) + 1 rows.
FREE_SCALE = 0.25
IMG_H, IMG_W = 128, 384
TI16 = dict(patch=16, hidden=192, mlp_dim=768, heads=3, layers=12)
OFFICIAL_NPZ = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                            'models', 'Ti_16-i21k-in1k-384.npz')


def _ln(dim):
    return nn.LayerNorm(dim, eps=1e-6)          # flax LayerNorm default eps


class MlpBlock(nn.Module):
    def __init__(self, dim, mlp_dim, drop):
        super().__init__()
        self.Dense_0 = nn.Linear(dim, mlp_dim)
        self.Dense_1 = nn.Linear(mlp_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # flax nn.gelu is the tanh approximation, not torch's exact default.
        x = self.drop(nn.functional.gelu(self.Dense_0(x), approximate='tanh'))
        return self.drop(self.Dense_1(x))


class MultiHeadDotProductAttention(nn.Module):
    def __init__(self, dim, heads, drop):
        super().__init__()
        self.heads = heads
        self.attn_drop = drop
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x):
        n, t, d = x.shape
        q, k, v = (f(x).view(n, t, self.heads, -1).transpose(1, 2)
                   for f in (self.query, self.key, self.value))
        x = nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop if self.training else 0.0)
        return self.out(x.transpose(1, 2).reshape(n, t, d))


class Encoder1DBlock(nn.Module):
    def __init__(self, dim, mlp_dim, heads, drop, attn_drop):
        super().__init__()
        self.LayerNorm_0 = _ln(dim)
        self.MultiHeadDotProductAttention_1 = MultiHeadDotProductAttention(
            dim, heads, attn_drop)
        self.LayerNorm_2 = _ln(dim)
        self.MlpBlock_3 = MlpBlock(dim, mlp_dim, drop)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = x + self.drop(self.MultiHeadDotProductAttention_1(self.LayerNorm_0(x)))
        return x + self.MlpBlock_3(self.LayerNorm_2(x))


class Encoder(nn.Module):
    def __init__(self, n_tokens, dim, mlp_dim, heads, layers, drop, attn_drop):
        super().__init__()
        self.posembed_input = nn.Parameter(torch.randn(1, n_tokens, dim) * 0.02)
        self.drop = nn.Dropout(drop)
        self.blocks = nn.ModuleList(
            Encoder1DBlock(dim, mlp_dim, heads, drop, attn_drop)
            for _ in range(layers))
        self.encoder_norm = _ln(dim)

    def forward(self, x):
        x = self.drop(x + self.posembed_input)
        for blk in self.blocks:
            x = blk(x)
        return self.encoder_norm(x)


class VisionTransformer(nn.Module):
    """models_vit.VisionTransformer, classifier='token', no ResNet stem, no
    pre_logits (the Ti/16 augreg checkpoints have none)."""

    def __init__(self, num_classes, img_size=(IMG_H, IMG_W), patch=16,
                 hidden=192, mlp_dim=768, heads=3, layers=12,
                 dropout=0.0, attn_dropout=0.0):
        super().__init__()
        self.grid = (img_size[0] // patch, img_size[1] // patch)
        self.embedding = nn.Conv2d(3, hidden, patch, stride=patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden))
        self.Transformer = Encoder(1 + self.grid[0] * self.grid[1], hidden,
                                   mlp_dim, heads, layers, dropout, attn_dropout)
        self.head = nn.Linear(hidden, num_classes)
        nn.init.zeros_(self.head.weight)        # official: kernel_init=zeros
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        x = self.embedding(x).flatten(2).transpose(1, 2)   # (n, h*w, c), row-major
        x = torch.cat([self.cls.expand(x.shape[0], -1, -1), x], dim=1)
        return self.head(self.Transformer(x)[:, 0])


def resize_posemb(posemb, grid):
    """checkpoint.interpolate_posembed, generalised to a non-square grid: keep
    the class token, order-1 scipy zoom of the square grid to (gh, gw)."""
    import scipy.ndimage
    tok, g = posemb[:, :1], posemb[0, 1:]
    gs = int(math.isqrt(g.shape[0]))
    assert gs * gs == g.shape[0], g.shape
    if (gs, gs) == tuple(grid):
        return posemb
    g = scipy.ndimage.zoom(g.reshape(gs, gs, -1),
                           (grid[0] / gs, grid[1] / gs, 1), order=1)
    return np.concatenate([tok, g.reshape(1, grid[0] * grid[1], -1)], axis=1)


def load_official(model, path=OFFICIAL_NPZ):
    """Official vit_jax .npz -> this port. Like checkpoint.load_pretrained, the
    head is only loaded when its shape matches (i.e. never for the lane head)."""
    d = np.load(path)
    sd = {}
    for k in d.files:
        a = d[k]
        name = k.replace('/', '.').replace('encoderblock_', 'blocks.')
        if name == 'Transformer.posembed_input.pos_embedding':
            name, a = 'Transformer.posembed_input', resize_posemb(a, model.grid)
        elif name == 'embedding.kernel':
            a = a.transpose(3, 2, 0, 1)                        # HWIO -> OIHW
        elif name.endswith('.kernel'):
            # attention q/k/v (in, heads, hd), out (heads, hd, out), Dense (in, out)
            a = a.reshape(-1, a.shape[-1]) if name.endswith('out.kernel') \
                else a.reshape(a.shape[0], -1)
            a = a.T
        elif name.endswith('.bias'):
            a = a.reshape(-1)
        name = name.replace('.kernel', '.weight').replace('.scale', '.weight')
        sd[name] = torch.from_numpy(np.ascontiguousarray(a))
    own = model.state_dict()
    sd = {k: v for k, v in sd.items()
          if not (k.startswith('head.') and v.shape != own[k].shape)}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    assert set(missing) <= {'head.weight', 'head.bias'}, missing
    return model


def build_lane_model(npz=OFFICIAL_NPZ, dropout=0.0, n_out=len(XS)):
    """ViT-Ti/16 at 128x384, pretrained backbone; head = 16 path offsets
    (+1 scaled free distance when n_out = len(XS) + 1)."""
    m = VisionTransformer(n_out, (IMG_H, IMG_W), dropout=dropout, **TI16)
    return load_official(m, npz) if npz else m


def crop_resize(img_rgb, top):
    """HxWx3 uint8 RGB -> IMG_H x IMG_W x 3 uint8, rows from `top` down.

    `top` is the first row kept: camera_geometry.horizon_row()+2. Above it is
    sky/wall, which carries no lane geometry and would waste ~45% of tokens.
    """
    import cv2
    return cv2.resize(img_rgb[int(top):], (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)


def preprocess(img_rgb, top):
    """HxWx3 uint8 RGB -> (3, IMG_H, IMG_W) float in [-1, 1]."""
    x = torch.from_numpy(crop_resize(img_rgb, top)).permute(2, 0, 1).float()
    return x / 127.5 - 1.0


# ── Self-check ────────────────────────────────────────────────────────────────

def selfcheck():
    torch.manual_seed(0)

    # 1. The port matches an independent implementation on the official
    #    weights: timm's ViT loading the same .npz, with flax's tanh GELU.
    import functools
    import timm
    ours = VisionTransformer(1000, (384, 384), **TI16)
    load_official(ours, OFFICIAL_NPZ)
    ref = timm.create_model(
        'vit_tiny_patch16_384', pretrained=False,
        act_layer=functools.partial(nn.GELU, approximate='tanh'))
    ref.load_pretrained(OFFICIAL_NPZ)
    x = torch.randn(2, 3, 384, 384)
    with torch.no_grad():
        a, b = ours.eval()(x), ref.eval()(x)
    err = (a - b).abs().max().item()
    assert err < 1e-3, f'port differs from timm by {err}'
    assert torch.equal(a.argmax(1), b.argmax(1))
    print(f'  official Ti/16 @384: max |ours - timm| = {err:.2e}')

    # 2. Pos-embed resize: identity at the native grid, keeps the class token.
    pe = np.load(OFFICIAL_NPZ)['Transformer/posembed_input/pos_embedding']
    assert resize_posemb(pe, (24, 24)) is pe
    r = resize_posemb(pe, (8, 24))
    assert r.shape == (1, 1 + 8 * 24, 192) and np.array_equal(r[:, 0], pe[:, 0])

    # 3. Lane model: pretrained backbone, zero head -> zero output at init,
    #    and the regression loss is trainable end to end.
    m = build_lane_model()
    img = np.random.randint(0, 256, (480, 640, 3), np.uint8)
    xb = torch.stack([preprocess(img, 221), preprocess(img[:, ::-1].copy(), 221)])
    assert xb.shape == (2, 3, IMG_H, IMG_W) and -1.0 <= xb.min() and xb.max() <= 1.0
    y = m(xb)
    assert y.shape == (2, len(XS)) and torch.count_nonzero(y) == 0
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    tgt = torch.full((2, len(XS)), 0.3)
    l0 = nn.functional.smooth_l1_loss(m(xb), tgt).item()
    for _ in range(5):
        loss = nn.functional.smooth_l1_loss(m(xb), tgt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < l0, (l0, loss.item())
    n = sum(p.numel() for p in m.parameters()) / 1e6
    print(f'  lane ViT-Ti/16 @{IMG_H}x{IMG_W}: {n:.2f} M params, '
          f'{m.grid[0]}x{m.grid[1]} patches, loss {l0:.4f} -> {loss.item():.4f}')
    print('selfcheck OK')


if __name__ == '__main__':
    if '--selfcheck' in sys.argv:
        selfcheck()
