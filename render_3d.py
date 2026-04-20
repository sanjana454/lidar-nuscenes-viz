#!/usr/bin/env python3
"""
Render a colorized PLY point cloud to a high-resolution PNG using a
custom software perspective projection + painter's algorithm (no 3D library).

Produces a 4-panel image:  perspective · front · side · top-down

Usage:
    python render_3d.py [--ply colorized.ply] [--out render_3d.png]
                        [--width 1920] [--height 1080]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
from utils import read_ply
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image


# read_ply imported from utils


# ── Camera / projection helpers ────────────────────────────────────────────

def look_at(eye, target, up):
    """Return a (4×4) view matrix: world → camera."""
    fwd = target - eye;  fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up); right /= np.linalg.norm(right)
    true_up = np.cross(right, fwd)
    R = np.array([right, true_up, -fwd], dtype=np.float64)   # 3×3
    t = -R @ eye
    M = np.eye(4)
    M[:3, :3] = R
    M[:3,  3] = t
    return M


def perspective_project(pts_world, view_mat, fov_deg, img_w, img_h):
    """
    Project Nx3 world points → pixel (u, v) and depth.
    Returns u, v (float arrays), depth (float), valid (bool mask).
    """
    n = len(pts_world)
    ones = np.ones((n, 1))
    pts_h = np.hstack([pts_world, ones])         # Nx4
    pts_cam = (view_mat @ pts_h.T).T             # Nx4 → camera space

    x_c = pts_cam[:, 0]
    y_c = pts_cam[:, 1]
    z_c = pts_cam[:, 2]   # depth in camera frame (positive = in front)

    fov_rad = np.deg2rad(fov_deg)
    # focal length from vertical FOV
    fy = (img_h / 2) / np.tan(fov_rad / 2)
    fx = fy
    cx, cy = img_w / 2, img_h / 2

    valid = z_c > 0.1
    safe_z = np.where(valid, z_c, 1.0)

    u = fx * x_c / safe_z + cx
    v = fy * y_c / safe_z + cy

    in_frame = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
    valid &= in_frame

    return u, v, z_c, valid


# ── Software rasteriser ────────────────────────────────────────────────────

def rasterise(
    u, v, depth, rgb_pts,
    valid, img_w, img_h,
    pt_radius=1,
    bg=(18, 18, 18),
):
    """
    Paint points onto an RGBA image using the painter's algorithm (far-first).
    pt_radius: half-size in pixels of each point stamp (0 = 1px, 1 = 3×3, 2 = 5×5).
    """
    canvas = np.full((img_h, img_w, 3), bg, dtype=np.uint8)

    ui = u[valid].astype(np.int32)
    vi = v[valid].astype(np.int32)
    d  = depth[valid]
    col = rgb_pts[valid]

    # Sort far → near so near points overwrite
    order = np.argsort(d)[::-1]
    ui, vi, col = ui[order], vi[order], col[order]

    if pt_radius == 0:
        canvas[vi, ui] = col
    else:
        r = pt_radius
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx*dx + dy*dy <= r*r:
                    vi2 = np.clip(vi + dy, 0, img_h - 1)
                    ui2 = np.clip(ui + dx, 0, img_w - 1)
                    canvas[vi2, ui2] = col

    return canvas


# ── Orthographic renderer (for 2-D plan views) ────────────────────────────

def ortho_render(
    pts, rgb_pts,
    axis_h: int, axis_v: int,  # which world axes map to image x/y
    axis_depth: int,
    scale: float,              # pixels per metre
    img_w: int, img_h: int,
    bg=(18, 18, 18),
    pt_radius=0,
):
    """Orthographic projection along one world axis."""
    u = (pts[:, axis_h] * scale + img_w / 2).astype(np.float64)
    v = (pts[:, axis_v] * scale + img_h / 2).astype(np.float64)
    depth = pts[:, axis_depth]

    in_frame = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
    valid = in_frame

    return rasterise(u, v, depth, rgb_pts, valid, img_w, img_h,
                     pt_radius=pt_radius, bg=bg)


# ── Multi-panel figure ─────────────────────────────────────────────────────

def render_panels(xyz, rgb, out_path, dpi=150):
    """
    4-panel high-quality render:
      [0] Perspective (isometric-ish)
      [1] Front   (X-Z plane, looking along -Y)
      [2] Side    (Y-Z plane, looking along +X)
      [3] Top     (X-Y plane, looking along -Z)
    """
    # Separate colored vs uncolored for rendering
    colored_mask = (rgb > 0).any(axis=1)
    pts_col = xyz[colored_mask]
    rgb_col = rgb[colored_mask]

    # Assign grey to uncolored so they give context without drowning colour
    rgb_all = rgb.copy()
    rgb_all[~colored_mask] = [55, 55, 55]

    # --- Panel dimensions ---
    PERSP_W, PERSP_H = 1280, 720
    SMALL_W, SMALL_H = 640, 480
    BG = (16, 16, 20)

    # ── Perspective panel ────────────────────────────────────────────────
    # Camera hovers above and behind the cloud centre
    centre = np.median(xyz, axis=0)
    span   = np.percentile(np.linalg.norm(xyz - centre, axis=1), 90)

    eye    = centre + np.array([-span * 0.3, -span * 1.1, span * 0.55])
    target = centre + np.array([0, 0, -span * 0.1])
    up     = np.array([0, 0, 1.0])

    V  = look_at(eye, target, up)
    u_p, v_p, d_p, valid_p = perspective_project(
        xyz - centre, V, fov_deg=55, img_w=PERSP_W, img_h=PERSP_H
    )
    img_persp = rasterise(u_p, v_p, d_p, rgb_all, valid_p, PERSP_W, PERSP_H,
                          pt_radius=1, bg=BG)

    # ── Orthographic panels ──────────────────────────────────────────────
    pts_c = xyz - centre  # centred coords
    extent = np.percentile(np.abs(pts_c), 95)
    scale  = min(SMALL_W, SMALL_H) / 2.0 / (extent + 1)

    # Front: look along -Y → image axes: X (right), Z (up)
    img_front = ortho_render(pts_c, rgb_all,
                             axis_h=0, axis_v=2, axis_depth=1,
                             scale=scale, img_w=SMALL_W, img_h=SMALL_H,
                             bg=BG, pt_radius=0)

    # Side: look along +X → image axes: Y (right), Z (up)
    img_side  = ortho_render(pts_c, rgb_all,
                             axis_h=1, axis_v=2, axis_depth=0,
                             scale=scale, img_w=SMALL_W, img_h=SMALL_H,
                             bg=BG, pt_radius=0)

    # Top: look along -Z → image axes: X (right), Y (up)
    img_top   = ortho_render(pts_c, rgb_all,
                             axis_h=0, axis_v=1, axis_depth=2,
                             scale=scale, img_w=SMALL_W, img_h=SMALL_H,
                             bg=BG, pt_radius=0)

    # ── Compose into a 2-row figure ──────────────────────────────────────
    fig = plt.figure(figsize=(16, 10), dpi=dpi, facecolor="#101014")

    gs = gridspec.GridSpec(
        2, 3,
        figure=fig,
        left=0.02, right=0.98, top=0.93, bottom=0.02,
        hspace=0.08, wspace=0.04,
    )

    ax_p  = fig.add_subplot(gs[0, :])   # top row: full width perspective
    ax_fr = fig.add_subplot(gs[1, 0])   # bottom-left: front
    ax_si = fig.add_subplot(gs[1, 1])   # bottom-mid:  side
    ax_to = fig.add_subplot(gs[1, 2])   # bottom-right: top

    def _show(ax, img, title):
        ax.imshow(img, interpolation="nearest", aspect="auto")
        ax.set_title(title, color="#cccccc", fontsize=10, pad=4)
        ax.axis("off")

    n_total   = len(xyz)
    n_colored = colored_mask.sum()
    pct       = 100.0 * n_colored / n_total

    _show(ax_p,  img_persp, f"Perspective  —  {n_total:,} pts  |  {n_colored:,} colored ({pct:.1f}%)")
    _show(ax_fr, img_front, "Front  (X–Z)")
    _show(ax_si, img_side,  "Side  (Y–Z)")
    _show(ax_to, img_top,   "Top  (X–Y)")

    fig.suptitle("Colorized LiDAR Point Cloud — nuScenes",
                 color="white", fontsize=13, y=0.97)

    plt.savefig(out_path, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    size_mb = Path(out_path).stat().st_size / 1e6
    print(f"Saved: {out_path}  ({size_mb:.1f} MB)")


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ply",  default="colorized.ply")
    ap.add_argument("--out",  default="render_3d.png")
    ap.add_argument("--dpi",  type=int, default=150)
    args = ap.parse_args()

    print(f"Reading {args.ply}…")
    xyz, rgb = read_ply(args.ply)
    print(f"  {len(xyz):,} points")

    print("Rendering…")
    render_panels(xyz, rgb, args.out, dpi=args.dpi)


if __name__ == "__main__":
    main()
