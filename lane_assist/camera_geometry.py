#!/usr/bin/env python3
"""Camera geometry for the ViT lane stack (ROS-free): pixel <-> road ground plane,
the horizon row the crop starts below, ego <-> world path transform.

Pinhole camera, K = (fx, fy, cx, cy); mount: height cam_h [m] above the road,
pitched down by cam_pitch [rad], cam_x [m] ahead of base_link. Ground frame is
base_link: x forward, y left, z up. Moved here unchanged from the classical
lane_detector.py (2026-10-08) when the classical detector left the package.
"""

import math

import numpy as np


def ground_from_pixel(u, v, K, cam_h, cam_pitch, cam_x):
    """Pixel(s) -> (X, Y) where the ray hits z=0, in base_link (x fwd, y left).

    Rays at or above the horizon return NaN. u, v may be arrays.
    """
    fx, fy, cx, cy = K
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    # optical frame (X right, Y down, Z fwd) -> camera_link (x fwd, y left, z up)
    dx = np.ones_like(u)
    dy = -(u - cx) / fx
    dz = -(v - cy) / fy
    # camera_link -> base_link: rotate by +pitch about +y (nose-down mount)
    ct, st = math.cos(cam_pitch), math.sin(cam_pitch)
    bx = ct * dx + st * dz
    by = dy
    bz = -st * dx + ct * dz
    with np.errstate(divide='ignore', invalid='ignore'):
        s = np.where(bz < -1e-6, cam_h / np.maximum(-bz, 1e-9), np.nan)
    return cam_x + s * bx, s * by


def pixel_from_ground(X, Y, K, cam_h, cam_pitch, cam_x):
    """Inverse of ground_from_pixel — used by the debug overlay and self-check."""
    fx, fy, cx, cy = K
    dx = np.asarray(X, dtype=float) - cam_x
    dy = np.asarray(Y, dtype=float)
    dz = -cam_h
    ct, st = math.cos(cam_pitch), math.sin(cam_pitch)
    cxx = ct * dx - st * dz
    cyy = dy
    czz = st * dx + ct * dz
    with np.errstate(divide='ignore', invalid='ignore'):
        u = cx - fx * cyy / cxx
        v = cy - fy * czz / cxx
    return u, v


def horizon_row(K, cam_pitch):
    """Image row where the ground plane vanishes. Everything above it is sky or
    wall and carries no lane geometry, so the mask starts below it."""
    _, fy, _, cy = K
    return cy - fy * math.tan(cam_pitch)


def ego_to_world(path, pose):
    """(N,6) ego horizon -> world frame, given pose (x, y, yaw)."""
    px, py, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    out = path.copy()
    out[:, 1] = px + path[:, 1] * c - path[:, 2] * s
    out[:, 2] = py + path[:, 1] * s + path[:, 2] * c
    out[:, 3] = path[:, 3] + yaw
    return out
