#!/usr/bin/env python3
"""
Fusion Visualization: projects LiDAR depth onto all 6 camera images in a
single figure arranged by physical camera position.

Explicitly surfaces the three core sensor-fusion challenges:
  1. Overlapping camera FOVs  — points visible in 2+ cameras shown in orange
  2. Timing differences       — Δt (ms) between LiDAR and each camera shown per panel
  3. Moving objects / ego-motion between sweeps — sweep age shown per layer

Usage:
    python fuse_viz.py [--scene scene-0061] [--sample <token>]
                       [--sweeps 5] [--out fusion_viz.png]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from matplotlib.patches import Patch
from PIL import Image

from utils import zbuffer_nearest

# Reuse the core pipeline
sys.path.insert(0, str(Path(__file__).parent))
from colorize_pointcloud import (
    NuScenesColorizer,
    project_points,
    load_lidar_bin,
)

# ── Camera layout (2 rows × 3 cols matching physical arrangement) ──────────
#   row 0 (front): LEFT  |  FRONT  |  RIGHT
#   row 1 (rear) : LEFT  |  BACK   |  RIGHT
LAYOUT = [
    ["CAM_FRONT_LEFT",  "CAM_FRONT",  "CAM_FRONT_RIGHT"],
    ["CAM_BACK_LEFT",   "CAM_BACK",   "CAM_BACK_RIGHT"],
]
DEPTH_CMAP   = cm.plasma
MAX_DEPTH_M  = 50.0    # colormap saturates here
OVERLAP_CLR  = "#FF6B00"   # orange for multi-camera overlap points


def compute_per_camera_projections(colorizer, sample_token, xyz_world, ref_ts):
    """
    For each camera channel return a dict with all projection data needed
    for visualisation and overlap detection.
    """
    results = {}
    for row in LAYOUT:
        for ch in row:
            cam_sd  = colorizer.nearest_sd_by_time(ch, ref_ts)
            if cam_sd is None:
                continue
            cam_cs  = colorizer.cal_sensors[cam_sd["calibrated_sensor_token"]]
            K_list  = cam_cs.get("camera_intrinsic")
            if not K_list:
                continue
            K       = np.array(K_list, dtype=np.float64)
            img_path = colorizer.root / cam_sd["filename"]
            if not img_path.exists():
                continue

            img     = np.array(Image.open(img_path).convert("RGB"))
            H, W    = img.shape[:2]
            pts_cam = colorizer.world_to_camera(xyz_world, cam_sd)
            u, v, valid = project_points(pts_cam, K, W, H, min_depth=0.5)

            # Z-buffer via shared utility: nearest point per pixel wins
            gi = np.where(valid)[0]
            visible_global = zbuffer_nearest(
                u[valid], v[valid], pts_cam[valid, 2], gi, W, H
            )

            dt_ms = (cam_sd["timestamp"] - ref_ts) / 1000.0

            results[ch] = dict(
                img=img, K=K, H=H, W=W,
                u=u, v=v, valid=valid, depth=pts_cam[:, 2],
                visible_global=visible_global,
                dt_ms=dt_ms,
                cam_sd=cam_sd,
            )
    return results


def detect_overlap(per_cam, n_pts):
    """
    For each point, count how many cameras see it (after z-buffering).
    Returns a boolean array: True where 2+ cameras see the point.
    """
    count = np.zeros(n_pts, dtype=np.int32)
    for ch, d in per_cam.items():
        count[d["visible_global"]] += 1
    return count >= 2


def draw_panel(ax, ch, data, overlap_mask, depth_norm):
    """
    Draw one camera panel: image background + LiDAR depth overlay.
    """
    img = data["img"]
    u, v, valid = data["u"], data["v"], data["valid"]
    depth = data["depth"]
    vis = data["visible_global"]
    H, W = data["H"], data["W"]
    dt_ms = data["dt_ms"]

    ax.imshow(img, aspect="auto", interpolation="nearest")

    # All visible (z-buffered) points
    vi_pts = vis
    u_vis  = u[vi_pts]
    v_vis  = v[vi_pts]
    d_vis  = depth[vi_pts]
    is_ov  = overlap_mask[vi_pts]

    # Non-overlap: coloured by depth via plasma
    mask_single = ~is_ov
    if mask_single.any():
        sc = ax.scatter(
            u_vis[mask_single], v_vis[mask_single],
            c=d_vis[mask_single],
            cmap=DEPTH_CMAP, vmin=0, vmax=MAX_DEPTH_M,
            s=3, alpha=0.85, linewidths=0,
        )

    # Overlap points: orange, slightly larger
    if is_ov.any():
        ax.scatter(
            u_vis[is_ov], v_vis[is_ov],
            color=OVERLAP_CLR,
            s=5, alpha=0.9, linewidths=0,
        )

    # Panel annotation ─────────────────────────────────────────────────────
    label = ch.replace("CAM_", "")
    dt_str = f"Δt = {dt_ms:+.1f} ms"
    n_vis  = len(vi_pts)
    n_ov   = is_ov.sum()

    title = f"{label}   {dt_str}   {n_vis:,} pts ({n_ov:,} overlap)"
    ax.set_title(title, fontsize=7.5, color="white",
                 pad=3, backgroundcolor=(0,0,0,0.55))
    ax.axis("off")


def create_fusion_viz(
    scene_name: str,
    sample_token: str,
    n_sweeps: int = 5,
    out_path: str = "fusion_viz.png",
    dpi: int = 150,
):
    colorizer = NuScenesColorizer(data_root="data")

    scene  = colorizer.scenes_by_name[scene_name]
    if sample_token is None:
        sample_token = scene["first_sample_token"]

    print(f"Scene: {scene_name}  sample: {sample_token[:8]}…")

    # ── 1. Collect and aggregate LIDAR sweeps ─────────────────────────────
    lidar_sd   = colorizer.get_sample_data_for_channel(sample_token, "LIDAR_TOP")
    ref_ts     = lidar_sd["timestamp"]
    sweep_sds  = colorizer.adjacent_lidar_sds(lidar_sd, n_total=n_sweeps)

    print(f"  LIDAR sweeps: {len(sweep_sds)}")

    sweep_layers = []   # list of (world_xyz, age_ms) per sweep
    for sd in sweep_sds:
        pts = load_lidar_bin(str(colorizer.root / sd["filename"]))[:, :3]
        pts_world = colorizer.lidar_to_world(pts, sd)
        age_ms    = (sd["timestamp"] - ref_ts) / 1000.0
        sweep_layers.append((pts_world, age_ms))

    xyz_world = np.concatenate([l[0] for l in sweep_layers], axis=0)
    n_pts     = len(xyz_world)
    print(f"  Total points: {n_pts:,}")

    # ── 2. Project into all cameras ────────────────────────────────────────
    per_cam = compute_per_camera_projections(colorizer, sample_token, xyz_world, ref_ts)

    # ── 3. Overlap detection ───────────────────────────────────────────────
    overlap_mask = detect_overlap(per_cam, n_pts)
    n_overlap    = overlap_mask.sum()
    print(f"  Points in overlapping FOVs: {n_overlap:,}")

    # ── 4. Figure layout ───────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 10), dpi=dpi, facecolor="#0d0d0d")

    outer = gridspec.GridSpec(
        3, 1, figure=fig,
        height_ratios=[0.06, 1, 0.04],
        hspace=0.04,
        left=0.01, right=0.99, top=0.97, bottom=0.02,
    )

    # Title row
    ax_title = fig.add_subplot(outer[0])
    ax_title.set_facecolor("#0d0d0d")
    ax_title.axis("off")

    # Camera grid
    cam_gs = gridspec.GridSpecFromSubplotSpec(
        2, 3, subplot_spec=outer[1],
        hspace=0.06, wspace=0.03,
    )

    # Colorbar row
    ax_cb = fig.add_subplot(outer[2])
    ax_cb.set_facecolor("#0d0d0d")

    # ── 5. Draw each camera panel ──────────────────────────────────────────
    depth_norm = plt.Normalize(0, MAX_DEPTH_M)

    for ri, row in enumerate(LAYOUT):
        for ci, ch in enumerate(row):
            ax = fig.add_subplot(cam_gs[ri, ci])
            ax.set_facecolor("#111")
            if ch in per_cam:
                draw_panel(ax, ch, per_cam[ch], overlap_mask, depth_norm)
            else:
                ax.text(0.5, 0.5, f"{ch}\nno data",
                        ha="center", va="center", color="grey",
                        transform=ax.transAxes)
                ax.axis("off")

    # ── 6. Header ──────────────────────────────────────────────────────────
    # Timing summary across all cameras
    dts = [d["dt_ms"] for d in per_cam.values()]
    dt_min, dt_max = min(dts), max(dts)

    n_colored = sum(len(d["visible_global"]) for d in per_cam.values())
    pct_total = 100.0 * sum(
        len(np.unique(d["visible_global"])) for d in per_cam.values()
    ) / n_pts

    title_txt = (
        f"LiDAR–Camera Sensor Fusion  |  {scene_name}  "
        f"|  {len(sweep_sds)} sweeps · {n_pts:,} pts"
    )
    sub_txt = (
        f"Camera timing offsets: {dt_min:+.1f} ms … {dt_max:+.1f} ms from LiDAR   "
        f"|   Overlapping-FOV points: {n_overlap:,}   "
        f"|   Total camera projections: {n_colored:,} ({pct_total:.0f}%)"
    )
    ax_title.text(0.5, 0.7, title_txt, ha="center", va="center",
                  color="white", fontsize=13, fontweight="bold",
                  transform=ax_title.transAxes)
    ax_title.text(0.5, 0.15, sub_txt, ha="center", va="center",
                  color="#aaaaaa", fontsize=9,
                  transform=ax_title.transAxes)

    # ── 7. Colorbar + legend ───────────────────────────────────────────────
    sm = cm.ScalarMappable(cmap=DEPTH_CMAP, norm=depth_norm)
    sm.set_array([])
    cb = plt.colorbar(sm, cax=ax_cb, orientation="horizontal")
    cb.set_label("LiDAR depth (m)", color="white", fontsize=9)
    cb.ax.xaxis.set_tick_params(color="white")
    plt.setp(cb.ax.xaxis.get_ticklabels(), color="white", fontsize=8)
    cb.outline.set_edgecolor("white")

    # Overlap legend patch
    ax_cb.add_patch(plt.Rectangle((0.78, -0.5), 0.015, 2.0,
                                   facecolor=OVERLAP_CLR, clip_on=False,
                                   transform=ax_cb.transAxes))
    ax_cb.text(0.797, 0.5, "Overlapping FOV",
               color="white", fontsize=8, va="center",
               transform=ax_cb.transAxes)

    # ── 8. Save ────────────────────────────────────────────────────────────
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    mb = Path(out_path).stat().st_size / 1e6
    print(f"Saved: {out_path}  ({mb:.1f} MB)")


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene",   default="scene-0061")
    ap.add_argument("--sample",  default="c923fe08b2ff4e27975d2bf30934383b",
                    help="Sample token (default: 5th sample of scene-0061)")
    ap.add_argument("--sweeps",  type=int, default=5)
    ap.add_argument("--out",     default="fusion_viz.png")
    ap.add_argument("--dpi",     type=int, default=150)
    args = ap.parse_args()

    create_fusion_viz(
        scene_name=args.scene,
        sample_token=args.sample,
        n_sweeps=args.sweeps,
        out_path=args.out,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
