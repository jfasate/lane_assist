# How the ViT lane model works — from camera frame to reference line

The model is a Vision Transformer (ViT-Ti/16, Dosovitskiy et al., *An Image is Worth
16x16 Words*, arXiv:2010.11929). It is ported line for line from the official
google-research/vision_transformer code into PyTorch, in
[lane_assist/vit_lane.py](lane_assist/vit_lane.py).

It takes **one camera frame** and returns **16 numbers: the path the car should follow
over the next 5 m**. This document follows a single frame through every step.

```
camera frame 640x480x3
   │ 1. crop below the horizon, resize      → 128x384x3, values in [-1, 1]
   │ 2. cut into 16x16 patches              → 192 patches of 16x16x3
   │ 3. patch embedding                     → 192 tokens x 192 numbers
   │ 4. add [CLS] token                     → 193 x 192
   │ 5. add position embedding              → 193 x 192
   │ 6. 12 transformer blocks               → 193 x 192
   │ 7. final LayerNorm, keep [CLS]         → 192
   │ 8. linear head                         → 16
   ▼
16 numbers = path Y [m] at X = 0, 1/3, 2/3, ..., 5 m ahead
   │ 9. smooth over time (EMA)
   │ 10. turn into a path: heading, curvature, speed
   ▼
/planning/ref_path  →  pure pursuit  →  steering
```

---

## Step 1 — Crop and normalise the image

**In:** a 640×480 RGB frame from the front camera (40 cm high, pitched down 0.0463 rad).

The top part of the frame is sky. It carries no road geometry and would waste about
45% of the model's capacity, so it is cut off:

```
horizon row = cy − fy · tan(pitch) = 240.5 − 462.2 · tan(0.0463) ≈ 219.1
crop starts at ceil(horizon) + 2   = row 222
```

The remaining rows 222–479 (258×640) are resized to **128×384** (H×W, area
interpolation). Pixel values 0–255 are scaled to **[-1, 1]**:
`x = pixel / 127.5 − 1`, as in the official ViT pipeline.

**Out:** a tensor of shape (3, 128, 384).

This is the **only** input. The model gets no map, no position, no speed, no lidar and
no earlier frames: every decision comes from this one picture.

---

## Step 2 — Cut the image into patches

The image is split into a grid of non-overlapping **16×16 pixel patches**:

```
128 / 16 = 8 rows   x   384 / 16 = 24 columns   =   192 patches
```

Each patch is 16×16×3 = **768 numbers**. A ViT treats patches the way a language
model treats words: the image becomes a "sentence" of 192 "words". They are numbered
row by row, left to right: patch 0 is the top-left (far away, near the horizon), and
patch 191 is the bottom-right (the road just in front of the car, on the right).

---

## Step 3 — Patch embedding: each patch becomes a token

One **shared linear layer** turns each patch's 768 pixel values into a vector of
**192 numbers** (the model width, D = 192):

```
token_i = W · patch_i + b          W: 192 x 768,  b: 192
```

It is implemented as a convolution with kernel 16×16, stride 16 and 192 output
channels, which does exactly this for all patches at once. The same W is used for
every patch, so a lane line looks the same to the model wherever it appears.

**Out:** 192 tokens × 192 numbers.

---

## Step 4 — Add the [CLS] token

One extra learned vector of 192 numbers, the **[CLS] ("class") token**, is put in
front of the sequence. It belongs to no patch. Its job is to **collect information
from all the patches**, and at the end it alone is used to make the prediction.

**Out:** 193 tokens × 192.

---

## Step 5 — Add the position embedding

Self-attention (step 6) on its own does not know *where* a token came from: shuffle
the patches and it would give the same answer. For a lane model the position is
everything (a line at the left edge vs in the centre means a different path), so a
learned **position embedding**, one 192-vector per position, is **added** to each
token:

```
token_i = token_i + pos_i          pos: 193 x 192, learned
```

The pretrained checkpoint was made for a 24×24 grid (384×384 images). Its position
embeddings are resized to our **8×24** grid by bilinear interpolation, keeping the
[CLS] position unchanged. This is the paper's recipe for fine-tuning at a new
resolution.

After training, neighbouring patches have similar position embeddings. The model has
learned the 2D layout of the road.

**Out:** 193 tokens × 192, each carrying *what* it shows and *where* it is.

---

## Step 6 — 12 transformer blocks

The tokens go through **12 identical blocks** in sequence, each with its own weights.
Every block has two parts, each wrapped in a residual ("add back the input")
connection:

```
x = x + Attention( LayerNorm(x) )
x = x + MLP( LayerNorm(x) )
```

LayerNorm rescales each token to zero mean and unit variance (with learned scale and
shift; ε = 1e-6). The residual connections let each block *refine* the tokens instead
of rewriting them, which is what makes 12 layers trainable.

### 6a. Multi-head self-attention: tokens exchange information

Every token looks at **every other token** and pulls in what is relevant to it. For
each token three vectors are computed by learned linear maps:

- **query** q: "what am I looking for?"
- **key** k: "what do I contain?"
- **value** v: "what do I pass on?"

```
attention weights  A = softmax( Q · Kᵀ / sqrt(d) )      (193 x 193)
output             = A · V
```

`A[i, j]` is how much token *i* reads from token *j*. Each row sums to 1.

This is done with **3 heads** in parallel, each on its own 64-number slice
(3 × 64 = 192), so each head can look for something different: for example one
follows a lane line, another checks the far road for obstacles. The 3 head outputs
are concatenated and mixed by one more linear layer (192 → 192).

This is what makes a ViT different from a CNN: already in block 1, a patch at the
bottom of the image (the near lane line) can read from a patch at the top (cones
5 m ahead). Global context comes from the start, not only after many layers.

### 6b. MLP: each token thinks on its own

A small two-layer network applied to every token separately:

```
192 → 768 → GELU (tanh form, as in the official code) → 192
```

Attention moves information **between** tokens; the MLP **processes** it inside each
token.

### What the 12 blocks do in practice

The analysis figures (`log/closed_loop_*/analysis/5_vit_pipeline_*.png`) show, for
each block, how much the [CLS] token reads from every patch:

- **Early blocks** look broadly: lane lines, road edges, the verge.
- **Later blocks** focus on what decides the path: obstacles (cones, debris, cars),
  the edges of the lane lines, and the far road where the lane goes.

**Out:** still 193 tokens × 192, but now the [CLS] token holds a summary of the whole
scene.

---

## Step 7 — Final LayerNorm, keep only [CLS]

A last LayerNorm is applied, then the 192 patch tokens are **discarded**. Only the
**[CLS] token**, 192 numbers, goes on. Everything the model knows about this frame
(where the lane is, how it curves, whether something blocks it, which lane is free)
must be in these 192 numbers.

---

## Step 8 — The head: 16 numbers out

One linear layer maps the 192 numbers to **16 numbers**:

```
Y = W_head · cls + b_head           W_head: 16 x 192
```

These are the model's output:

```
Y[k] = lateral position of the path, in metres (+ = left),
       at X = XS[k] metres ahead of the car,  XS = 0, 0.333, 0.667, ..., 5.0
```

all in the car's own frame (x forward, y left, origin at the car).

**What the path means:** where the car **should be**, not where it is.

- **Normal lane:** the lane centre. A car drifted 0.2 m right of centre gets
  Y[0] = +0.2: "the centre is 0.2 m to your left".
- **Curve:** the Y values bend, e.g. 0, 0.05, 0.2, … 0.8 for a left curve.
- **Obstacle in the lane:** a smooth lane change into a free lane of the same
  direction, finished 1.5 m before the obstacle. Y rises to about ± one lane width
  by the far end.
- **Already too close:** a late lane change, starting now.

The head is initialised to **zero** (the official recipe for a new task), with its
bias set to the average training path, so training starts from "predict the average
road" and learns everything else from there.

### Size and speed

| | |
|---|---|
| patch size / tokens | 16 px / 192 + 1 [CLS] |
| width D / heads / MLP | 192 / 3 / 768 |
| blocks | 12 |
| parameters | 5.53 M |
| inference | ≈ 4 ms per frame on the GPU |

---

## Step 9 — Smooth over time

The model sees one frame at a time, so its output jitters slightly from frame to
frame. `vit_lane_node` smooths it with an exponential moving average:

```
Y_smooth = 0.6 · Y_new + 0.4 · Y_smooth_previous
```

---

## Step 10 — From 16 numbers to a reference path

The 16 points `(XS[k], Y[k])` become a full path, one row per point
(`path_from_pred`):

```
heading     psi   = atan( dY/dX )
curvature   kappa = d²Y/dX² / (1 + (dY/dX)²)^1.5
speed       vx    = min( 1.5 m/s,  sqrt( 3.0 / |kappa| ) )     slower in tight curves
arc length  s     = cumulative distance along the points
```

This is published 20 times a second on **`/planning/ref_path`** as a (16, 6) array
`[s, x, y, psi, kappa, vx]` in the car frame. That is the same message the classic
lane detector produced, so nothing downstream changes.

The pure-pursuit follower then picks the path point about 1.2 m ahead and steers
toward it:

```
steer = atan( 2 · wheelbase · sin(alpha) / lookahead )
```

where alpha is the angle to that point. That closes the loop: camera frame → 16
numbers → path → steering, about 20 times a second.

---

## Where the weights come from

1. **Pretraining (done by Google):** the same ViT-Ti/16 was trained on ImageNet-21k
   and then ImageNet-1k (official augreg checkpoint
   `Ti_16-i21k-in1k-384.npz`). Steps 3–7 therefore start out already able to see
   edges, textures and objects.
2. **Fine-tuning (ours):** the whole network plus the new 16-output head is trained on
   about 200,000 simulated frames from 99 generated road maps. Each frame's target is
   the 16 Y values of the expert path computed from the car's true pose. The loss is
   the smooth-L1 distance between the predicted and target Y values.

The model never sees a lane-line rule written down. It learns from examples alone
which pixels matter: paint, road edges, shadows that are *not* lines, cones and debris
that block the lane, and green or red lanes it must not enter.
