#!/usr/bin/env python3
"""
Bird's-Eye-View (BEV) visualization of the colorized LiDAR point cloud.

Renders:
  • Colorized LiDAR points (actual RGB from camera fusion, grey for uncolored)
  • Per-camera frustum wedges (physical FOV cones in top-down view)
  • Ego-vehicle footprint at scan origin
  • Range rings every 10 m with labels
  • Overlap indicator: points visible in 2+ camera FOVs
  • Sensor-fusion challenge annotations

Usage:
    python bev_viz.py [--ply colorized.ply] [--scene scene-0061]
                      [--sample <token>] [--out bev.png] [--range 60]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
from utils import read_ply
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Wedge, FancyBboxPatch
from scipy.spatial.transform import Rotation

DATA_ROOT = Path("data")
META_ROOT = DATA_ROOT / "v1.0-mini"

# Camera display colours for frustum wedges
CAM_COLORS = {
    "CAM_FRONT":       "#00BFFF",   # deep sky blue
    "CAM_FRONT_LEFT":  "#32CD32",   # lime green
    "CAM_FRONT_RIGHT": "#FFD700",   # gold
    "CAM_BACK":        "#FF4500",   # orange red
    "CAM_BACK_LEFT":   "#DA70D6",   # orchid
    "CAM_BACK_RIGHT":  "#FF69B4",   # hot pink
}

# Typical vehicle footprint (metres)
VEH_LENGTH = 4.6
VEH_WIDTH  = 1.9


# read_ply imported from utils


# ── Calibration helpers ───────────────────────────────────────────────────

def load_camera_frustums(sample_token: str):
    """
    Return per-camera dict with:
      pos_ego  : (x, y) camera position in ego frame (metres)
      yaw_deg  : optical-axis angle in XY-plane (0 = forward)
      hfov_deg : horizontal field of view (degrees)
      color    : display colour
    """
    with open(META_ROOT / "calibrated_sensor.json") as f:
        cs_all = json.load(f)
    with open(META_ROOT / "sensor.json") as f:
        sensors = {s["token"]: s for s in json.load(f)}
    with open(META_ROOT / "sample_data.json") as f:
        sd_all = json.load(f)

    # Find calibrated_sensor tokens used by this sample
    sample_cs_tokens = {}
    for sd in sd_all:
        if sd["sample_token"] != sample_token:
            continue
        cs = next((c for c in cs_all if c["token"] == sd["calibrated_sensor_token"]), None)
        if cs is None:
            continue
        sensor = sensors.get(cs["sensor_token"], {})
        ch = sensor.get("channel", "")
        if sensor.get("modality") == "camera" and ch not in sample_cs_tokens:
            sample_cs_tokens[ch] = cs

    frustums = {}
    for ch, cs in sample_cs_tokens.items():
        K = cs.get("camera_intrinsic")
        if not K:
            continue
        fx  = K[0][0]
        img_w = 1600     # nuScenes standard
        hfov = 2 * np.degrees(np.arctan(img_w / (2 * fx)))

        q = cs["rotation"]   # [w, x, y, z]
        R = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        axis = R @ np.array([0.0, 0.0, 1.0])     # camera z-axis in ego frame
        yaw  = np.degrees(np.arctan2(axis[1], axis[0]))
        pos  = (cs["translation"][0], cs["translation"][1])

        frustums[ch] = dict(pos=pos, yaw_deg=yaw, hfov_deg=hfov,
                            color=CAM_COLORS.get(ch, "white"))
    return frustums


# ── Overlap detection in top-down view ───────────────────────────────────

def point_in_frustum(xy: np.ndarray, cam: dict, max_range: float = 55.0) -> np.ndarray:
    """Boolean mask: which Nx2 points fall inside this camera's 2-D FOV cone."""
    cx, cy = cam["pos"]
    dx = xy[:, 0] - cx
    dy = xy[:, 1] - cy
    dist = np.sqrt(dx**2 + dy**2)
    angle_to_pt = np.degrees(np.arctan2(dy, dx))

    # Angular difference from optical axis (wrapped to [-180, 180])
    diff = (angle_to_pt - cam["yaw_deg"] + 180) % 360 - 180
    half = cam["hfov_deg"] / 2
    return (np.abs(diff) <= half) & (dist <= max_range)


def compute_overlap_mask(xy: np.ndarray, frustums: dict) -> np.ndarray:
    """True for points that fall inside 2+ camera frustums."""
    count = np.zeros(len(xy), dtype=np.int32)
    for cam in frustums.values():
        count += point_in_frustum(xy, cam).astype(np.int32)
    return count >= 2


# ── BEV renderer ─────────────────────────────────────────────────────────

def render_bev(
    xyz: np.ndarray,
    rgb: np.ndarray,
    frustums: dict,
    out_path: str,
    scene_name: str,
    sample_token: str,
    range_m: float = 60.0,
    dpi: int = 180,
):
    # Centre on median (proxy for ego vehicle position)
    centre = np.median(xyz, axis=0)
    pts = xyz - centre          # shift so ego is at ~(0,0)
    xy  = pts[:, :2]
    z   = pts[:, 2]

    colored_mask = (rgb > 0).any(axis=1)
    overlap_mask = compute_overlap_mask(xy, frustums)

    # Annotation summary
    n_total   = len(pts)
    n_colored = colored_mask.sum()
    n_overlap = overlap_mask.sum()
    pct_col   = 100.0 * n_colored / n_total

    # ── Figure ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 12), dpi=dpi, facecolor="#0d0d12")
    ax.set_facecolor("#0d0d12")

    # ── Range rings ───────────────────────────────────────────────────────
    for r in range(10, int(range_m) + 1, 10):
        circle = plt.Circle((0, 0), r, color="#2a2a3a", linewidth=0.8,
                             fill=False, linestyle="--", zorder=1)
        ax.add_patch(circle)
        ax.text(0, r + 0.8, f"{r} m", ha="center", va="bottom",
                color="#555566", fontsize=7, zorder=2)

    # ── Camera frustums ───────────────────────────────────────────────────
    frustum_range = range_m * 0.92
    for ch, cam in frustums.items():
        cx, cy   = cam["pos"]
        yaw      = cam["yaw_deg"]
        half_fov = cam["hfov_deg"] / 2
        color    = cam["color"]

        wedge = Wedge(
            center=(cx, cy),
            r=frustum_range,
            theta1=yaw - half_fov,
            theta2=yaw + half_fov,
            color=color, alpha=0.07, zorder=2,
        )
        ax.add_patch(wedge)

        # Frustum boundary lines
        for sign in (-1, 1):
            angle_rad = np.radians(yaw + sign * half_fov)
            ax.plot(
                [cx, cx + frustum_range * np.cos(angle_rad)],
                [cy, cy + frustum_range * np.sin(angle_rad)],
                color=color, linewidth=0.7, alpha=0.55, zorder=3,
            )

        # Camera label at mid-range
        label_r = frustum_range * 0.55
        label_angle = np.radians(yaw)
        lx = cx + label_r * np.cos(label_angle)
        ly = cy + label_r * np.sin(label_angle)
        short = ch.replace("CAM_", "")
        ax.text(lx, ly, short, ha="center", va="center",
                color=color, fontsize=7.5, fontweight="bold",
                alpha=0.9, zorder=4)

    # ── Point cloud — three layers ─────────────────────────────────────────
    # Crop to view range
    in_range = (np.abs(xy[:, 0]) < range_m) & (np.abs(xy[:, 1]) < range_m)

    # 1. Uncolored points — subtle grey
    mask_unc = in_range & ~colored_mask
    if mask_unc.any():
        ax.scatter(xy[mask_unc, 0], xy[mask_unc, 1],
                   c="#3a3a4a", s=0.5, linewidths=0, alpha=0.5, zorder=5)

    # 2. Colored points (not in overlap zone)
    mask_col_no_ov = in_range & colored_mask & ~overlap_mask
    if mask_col_no_ov.any():
        colors_norm = rgb[mask_col_no_ov].astype(np.float32) / 255.0
        ax.scatter(xy[mask_col_no_ov, 0], xy[mask_col_no_ov, 1],
                   c=colors_norm, s=1.5, linewidths=0, alpha=0.92, zorder=6)

    # 3. Overlap points — draw actual color but with an orange halo
    mask_ov = in_range & colored_mask & overlap_mask
    if mask_ov.any():
        colors_norm = rgb[mask_ov].astype(np.float32) / 255.0
        # Halo
        ax.scatter(xy[mask_ov, 0], xy[mask_ov, 1],
                   c="#FF6B00", s=8, linewidths=0, alpha=0.4, zorder=7)
        # Actual colour on top
        ax.scatter(xy[mask_ov, 0], xy[mask_ov, 1],
                   c=colors_norm, s=1.8, linewidths=0, alpha=0.95, zorder=8)

    # ── Ego vehicle rectangle ─────────────────────────────────────────────
    # Draw a simple car footprint centred at (0, 0), oriented along +X
    car = FancyBboxPatch(
        (-VEH_LENGTH / 2, -VEH_WIDTH / 2),
        VEH_LENGTH, VEH_WIDTH,
        boxstyle="round,pad=0.2",
        linewidth=1.5, edgecolor="white", facecolor="#1a1a2e", zorder=10,
    )
    ax.add_patch(car)
    # Forward arrow
    ax.annotate("", xy=(VEH_LENGTH / 2 + 1.5, 0), xytext=(VEH_LENGTH / 2, 0),
                arrowprops=dict(arrowstyle="->", color="white", lw=1.5), zorder=11)
    ax.text(0, 0, "EGO", ha="center", va="center",
            color="white", fontsize=7, fontweight="bold", zorder=12)

    # ── Compass rose ──────────────────────────────────────────────────────
    compass_x, compass_y = range_m * 0.82, range_m * 0.82
    ax.annotate("N", xy=(compass_x, compass_y + 3),
                xytext=(compass_x, compass_y),
                arrowprops=dict(arrowstyle="->", color="#cccccc", lw=1.5),
                ha="center", va="center", color="#cccccc", fontsize=9, zorder=15)

    # ── Scale bar ─────────────────────────────────────────────────────────
    sb_x0, sb_y = -range_m * 0.85, -range_m * 0.90
    ax.plot([sb_x0, sb_x0 + 20], [sb_y, sb_y],
            color="white", lw=2, zorder=15)
    ax.text(sb_x0 + 10, sb_y - 2, "20 m", ha="center", va="top",
            color="white", fontsize=8, zorder=15)

    # ── Axis limits & styling ─────────────────────────────────────────────
    ax.set_xlim(-range_m, range_m)
    ax.set_ylim(-range_m, range_m)
    ax.set_aspect("equal")
    ax.tick_params(colors="#444455")
    ax.set_xlabel("X (m) — forward →", color="#666677", fontsize=8)
    ax.set_ylabel("Y (m) — left ↑", color="#666677", fontsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#222233")

    # ── Title & stats ─────────────────────────────────────────────────────
    ax.set_title(
        f"Bird's-Eye View — {scene_name}  |  sample {sample_token[:8]}…\n"
        f"{n_total:,} pts total  ·  {n_colored:,} colored ({pct_col:.1f}%)  "
        f"·  {n_overlap:,} in overlapping FOVs",
        color="white", fontsize=10, pad=10,
    )

    # ── Legend ────────────────────────────────────────────────────────────
    handles = [
        mpatches.Patch(facecolor=cam["color"], alpha=0.7,
                       label=ch.replace("CAM_", ""))
        for ch, cam in frustums.items()
    ]
    handles += [
        mpatches.Patch(facecolor="#FF6B00", alpha=0.7, label="Overlap FOV"),
        mpatches.Patch(facecolor="#3a3a4a", label="Uncolored pts"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=7,
              facecolor="#111122", edgecolor="#333344", labelcolor="white",
              framealpha=0.85, ncol=2)

    # ── Fusion challenge annotations ──────────────────────────────────────
    challenges = [
        "⊕ Overlap zones (orange): same LiDAR point visible in 2+ cameras;\n"
        "  best-view selection (min angular offset from optical centre) picks winner.",
        "⊘ Timing: cameras fire within ±40 ms of LiDAR sweep;\n"
        "  each sweep transformed through its own ego-pose to world frame.",
        "⊙ Multi-sweep: 5 sweeps aggregated (±~150 ms); motion\n"
        "  compensation via per-sweep world-frame projection.",
    ]
    note = "\n".join(challenges)
    ax.text(-range_m + 0.5, -range_m + 0.5, note,
            ha="left", va="bottom", color="#8888aa", fontsize=6.5,
            linespacing=1.5, zorder=20,
            bbox=dict(facecolor="#0d0d12", edgecolor="#333344",
                      alpha=0.85, boxstyle="round,pad=0.4"))

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    mb = Path(out_path).stat().st_size / 1e6
    print(f"Saved BEV: {out_path}  ({mb:.1f} MB)")


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ply",     default="colorized.ply")
    ap.add_argument("--scene",   default="scene-0061")
    ap.add_argument("--sample",  default="c923fe08b2ff4e27975d2bf30934383b")
    ap.add_argument("--out",     default="bev.png")
    ap.add_argument("--range",   type=float, default=60.0,
                    help="Display range radius in metres (default 60)")
    ap.add_argument("--dpi",     type=int, default=180)
    args = ap.parse_args()

    print(f"Reading {args.ply}…")
    xyz, rgb = read_ply(args.ply)
    print(f"  {len(xyz):,} points  |  {(rgb>0).any(1).sum():,} colored")

    print("Loading camera calibration…")
    frustums = load_camera_frustums(args.sample)
    for ch, f in frustums.items():
        print(f"  {ch:22s} yaw={f['yaw_deg']:+6.1f}°  hFOV={f['hfov_deg']:.1f}°")

    print("Rendering BEV…")
    render_bev(xyz, rgb, frustums,
               out_path=args.out,
               scene_name=args.scene,
               sample_token=args.sample,
               range_m=args.range,
               dpi=args.dpi)


if __name__ == "__main__":
    main()
