#!/usr/bin/env python3
"""
Fusion quality metrics for the colorized LiDAR point cloud.

Reports:
  • Coverage      — colored / uncolored / per-camera breakdown
  • Timing        — camera-to-LiDAR timestamp offsets
  • Geometry      — depth range, point density, z-buffer contention
  • Color quality — brightness, saturation, entropy
  • Overlap       — points visible in 2+ camera FOVs
  • Comparison    — side-by-side table for two scenes / runs

Usage:
    python metrics.py --ply colorized.ply --scene scene-0061 --sample <tok>
    python metrics.py --compare colorized.ply colorized_1094.ply \
                      --scenes scene-0061 scene-1094 \
                      --samples <tok0> <tok1>
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from colorize_pointcloud import (
    NuScenesColorizer, project_points, load_lidar_bin, quat_to_R
)
from utils import read_ply, zbuffer_nearest

DATA_ROOT = Path("data")
CAM_CHANNELS = [
    "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    "CAM_BACK",  "CAM_BACK_LEFT",  "CAM_BACK_RIGHT",
]


# ── Color quality helpers ──────────────────────────────────────────────────

def rgb_to_hsv_s(rgb_f):
    """Return saturation channel (0-1) for Nx3 float32 RGB in [0,1]."""
    r, g, b = rgb_f[:,0], rgb_f[:,1], rgb_f[:,2]
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin
    s = np.where(cmax > 0, delta / (cmax + 1e-9), 0.0)
    return s


def color_entropy(rgb_u8, bins=32):
    """Shannon entropy of the joint RGB histogram (bits)."""
    r = (rgb_u8[:,0] // (256 // bins)).astype(np.int32)
    g = (rgb_u8[:,1] // (256 // bins)).astype(np.int32)
    b = (rgb_u8[:,2] // (256 // bins)).astype(np.int32)
    flat = r * bins * bins + g * bins + b
    counts = np.bincount(flat, minlength=bins**3).astype(np.float64)
    probs = counts[counts > 0] / counts.sum()
    return float(-np.sum(probs * np.log2(probs)))


# ── Per-camera projection metrics ─────────────────────────────────────────

def per_camera_metrics(colorizer, sample_token, xyz_world, ref_ts):
    results = {}
    for ch in CAM_CHANNELS:
        cam_sd = colorizer.nearest_sd_by_time(ch, ref_ts)
        if cam_sd is None:
            continue
        cam_cs = colorizer.cal_sensors[cam_sd["calibrated_sensor_token"]]
        K_list = cam_cs.get("camera_intrinsic")
        if not K_list:
            continue
        K = np.array(K_list, dtype=np.float64)
        img_path = colorizer.root / cam_sd["filename"]
        if not img_path.exists():
            continue

        img = np.array(Image.open(img_path).convert("RGB"))
        H, W = img.shape[:2]
        pts_cam = colorizer.world_to_camera(xyz_world, cam_sd)
        u, v, valid = project_points(pts_cam, K, W, H, min_depth=0.5)

        # Z-buffer via shared utility
        gi = np.where(valid)[0]
        visible_global = zbuffer_nearest(u[valid], v[valid],
                                         pts_cam[valid, 2], gi, W, H)
        fst_depths = pts_cam[visible_global, 2]

        n_valid   = valid.sum()
        n_visible = len(visible_global)               # after z-buffer
        n_pixels  = H * W
        # contention: projected - visible = points that lost the z-buffer
        n_contend = n_valid - n_visible
        dt_ms     = (cam_sd["timestamp"] - ref_ts) / 1000.0

        # depth stats for visible points
        d_vis = fst_depths
        results[ch] = dict(
            dt_ms=dt_ms,
            n_projected=int(n_valid),
            n_visible=int(n_visible),
            n_zbuf_lost=int(n_contend),
            zbuf_contention_pct=100.0 * n_contend / max(n_valid, 1),
            depth_mean=float(d_vis.mean()) if len(d_vis) else 0,
            depth_median=float(np.median(d_vis)) if len(d_vis) else 0,
            depth_p95=float(np.percentile(d_vis, 95)) if len(d_vis) else 0,
            pixel_fill_pct=100.0 * n_visible / n_pixels,
        )
    return results


# ── Overlap detection (2-D FOV, matches bev_viz.py) ──────────────────────

def count_overlap_pts(xyz_world, colorizer, sample_token, ref_ts):
    """Count points visible in 2+ cameras after z-buffering."""
    visible_count = np.zeros(len(xyz_world), dtype=np.int32)
    for ch in CAM_CHANNELS:
        cam_sd = colorizer.nearest_sd_by_time(ch, ref_ts)
        if not cam_sd:
            continue
        cam_cs = colorizer.cal_sensors[cam_sd["calibrated_sensor_token"]]
        K_list = cam_cs.get("camera_intrinsic")
        if not K_list:
            continue
        K = np.array(K_list, dtype=np.float64)
        img_path = colorizer.root / cam_sd["filename"]
        if not img_path.exists():
            continue
        H, W = int(K[1, 2] * 2), int(K[0, 2] * 2)   # approx from principal point
        pts_cam = colorizer.world_to_camera(xyz_world, cam_sd)
        u, v, valid = project_points(pts_cam, K, W, H)
        gi = np.where(valid)[0]
        vis = zbuffer_nearest(u[valid], v[valid], pts_cam[valid, 2], gi, W, H)
        visible_count[vis] += 1
    return (visible_count >= 2).sum()


# ── Ego velocity ───────────────────────────────────────────────────────────

def estimate_ego_speed(colorizer, lidar_sd):
    """Approximate ego speed (m/s) from consecutive ego poses."""
    ep_curr = colorizer.ego_poses[lidar_sd["ego_pose_token"]]
    # find next LIDAR sd
    next_tok = lidar_sd.get("next")
    if not next_tok:
        return None
    next_sd = colorizer.sample_data.get(next_tok)
    if not next_sd:
        return None
    ep_next = colorizer.ego_poses.get(next_sd.get("ego_pose_token", ""))
    if not ep_next:
        return None
    dt = (ep_next["timestamp"] - ep_curr["timestamp"]) / 1e6
    dp = np.array(ep_next["translation"]) - np.array(ep_curr["translation"])
    return float(np.linalg.norm(dp) / max(dt, 1e-6))


# ── Main compute ───────────────────────────────────────────────────────────

def compute_metrics(ply_path, scene_name, sample_token=None, n_sweeps=5):
    colorizer = NuScenesColorizer(data_root="data")
    scene = colorizer.scenes_by_name[scene_name]
    if sample_token is None:
        sample_token = scene["first_sample_token"]

    lidar_sd  = colorizer.get_sample_data_for_channel(sample_token, "LIDAR_TOP")
    ref_ts    = lidar_sd["timestamp"]
    sweep_sds = colorizer.adjacent_lidar_sds(lidar_sd, n_total=n_sweeps)

    xyz_world = np.concatenate([
        colorizer.lidar_to_world(
            load_lidar_bin(str(colorizer.root / sd["filename"]))[:, :3], sd)
        for sd in sweep_sds
    ], axis=0)

    xyz, rgb = read_ply(ply_path)
    colored_mask = (rgb > 0).any(axis=1)
    col_rgb = rgb[colored_mask].astype(np.float32) / 255.0

    n_total   = len(xyz)
    n_colored = colored_mask.sum()
    n_uncolor = n_total - n_colored

    # Color quality (on colored points only)
    brightness  = col_rgb.mean()
    saturation  = rgb_to_hsv_s(col_rgb).mean()
    entropy     = color_entropy(rgb[colored_mask])

    # Geometry
    z_vals = xyz[:, 2]
    pts_c  = xyz - np.median(xyz, axis=0)
    xy_dist = np.sqrt(pts_c[:,0]**2 + pts_c[:,1]**2)

    # Per-camera
    cam_stats = per_camera_metrics(colorizer, sample_token, xyz_world, ref_ts)

    # Overlap
    n_overlap = count_overlap_pts(xyz_world, colorizer, sample_token, ref_ts)

    # Ego speed
    ego_speed = estimate_ego_speed(colorizer, lidar_sd)

    return dict(
        scene=scene_name,
        sample=sample_token[:8],
        n_sweeps=len(sweep_sds),
        ply_path=str(ply_path),
        # ── Coverage ──────────────────────────────────────────────────────
        n_total=int(n_total),
        n_colored=int(n_colored),
        n_uncolored=int(n_uncolor),
        pct_colored=100.0 * n_colored / n_total,
        n_overlap=int(n_overlap),
        pct_overlap=100.0 * n_overlap / n_total,
        # ── Color quality ─────────────────────────────────────────────────
        mean_brightness=float(brightness),
        mean_saturation=float(saturation),
        color_entropy_bits=float(entropy),
        # ── Geometry ──────────────────────────────────────────────────────
        depth_range_m=float(xy_dist.max()),
        z_min=float(z_vals.min()),
        z_max=float(z_vals.max()),
        pts_per_sweep=n_total // max(len(sweep_sds), 1),
        # ── Timing ────────────────────────────────────────────────────────
        cam_dt_ms={ch: round(v["dt_ms"], 2) for ch, v in cam_stats.items()},
        max_cam_dt_ms=max(abs(v["dt_ms"]) for v in cam_stats.values()),
        # ── Per-camera ────────────────────────────────────────────────────
        cam_projected={ch: v["n_projected"] for ch, v in cam_stats.items()},
        cam_visible={ch: v["n_visible"]   for ch, v in cam_stats.items()},
        cam_zbuf_contention_pct={ch: round(v["zbuf_contention_pct"],1)
                                 for ch, v in cam_stats.items()},
        cam_depth_median={ch: round(v["depth_median"],1) for ch, v in cam_stats.items()},
        cam_pixel_fill_pct={ch: round(v["pixel_fill_pct"],2) for ch, v in cam_stats.items()},
        # ── Dynamics ──────────────────────────────────────────────────────
        ego_speed_mps=round(ego_speed, 2) if ego_speed else None,
    )


# ── Pretty printer ─────────────────────────────────────────────────────────

def print_report(m, label=""):
    W = 58
    hdr = f"  {label or m['scene']}  "
    print("\n" + "═" * W)
    print(hdr.center(W))
    print("═" * W)

    def row(k, v):
        print(f"  {k:<36s} {str(v):>18s}")

    def section(title):
        print(f"\n  ── {title} {'─'*(W-6-len(title))}")

    section("Run configuration")
    row("Scene",              m["scene"])
    row("Sample token",       m["sample"] + "…")
    row("Sweeps aggregated",  m["n_sweeps"])
    row("PLY file",           Path(m["ply_path"]).name)

    section("Coverage")
    row("Total points",       f"{m['n_total']:,}")
    row("Colored points",     f"{m['n_colored']:,}  ({m['pct_colored']:.1f}%)")
    row("Uncolored points",   f"{m['n_uncolored']:,}  ({100-m['pct_colored']:.1f}%)")
    row("Overlap-FOV points", f"{m['n_overlap']:,}  ({m['pct_overlap']:.1f}%)")

    section("Color quality  (colored points only)")
    row("Mean brightness",    f"{m['mean_brightness']:.3f}  (0=black, 1=white)")
    row("Mean saturation",    f"{m['mean_saturation']:.3f}  (0=grey,  1=vivid)")
    row("Color entropy",      f"{m['color_entropy_bits']:.2f} bits  (max ~15 for 32³)")

    section("Geometry")
    row("Max radial range",   f"{m['depth_range_m']:.1f} m")
    row("Z range",            f"{m['z_min']:.1f} → {m['z_max']:.1f} m")
    row("Points per sweep",   f"{m['pts_per_sweep']:,}")

    section("Timing offsets  (camera − LiDAR)")
    for ch, dt in m["cam_dt_ms"].items():
        row(f"  {ch.replace('CAM_','')}", f"{dt:+.1f} ms")
    row("Max |Δt|",           f"{m['max_cam_dt_ms']:.1f} ms")
    if m["ego_speed_mps"] is not None:
        max_shift = m["ego_speed_mps"] * m["max_cam_dt_ms"] / 1000
        row("Ego speed",      f"{m['ego_speed_mps']:.2f} m/s")
        row("Max ego shift",  f"{max_shift*100:.1f} cm  @ max Δt")

    section("Per-camera  (after z-buffer)")
    row(f"  {'Camera':<18s} {'Proj':>7s} {'Vis':>7s} {'Zbuf%':>6s} {'Fill%':>6s}  {'Med-d':>6s}", "")
    for ch in CAM_CHANNELS:
        if ch not in m["cam_projected"]:
            continue
        short = ch.replace("CAM_", "")
        row(f"  {short:<18s}"
            f" {m['cam_projected'][ch]:>7,}"
            f" {m['cam_visible'][ch]:>7,}"
            f" {m['cam_zbuf_contention_pct'][ch]:>5.1f}%"
            f" {m['cam_pixel_fill_pct'][ch]:>5.2f}%"
            f"  {m['cam_depth_median'][ch]:>5.1f}m", "")

    print("\n" + "═" * W + "\n")


def print_comparison(m0, m1):
    """Side-by-side comparison table."""
    W = 80
    print("\n" + "═" * W)
    print("  SCENE COMPARISON".center(W))
    print("═" * W)
    fmt = "  {:<34s}  {:>18s}  {:>18s}"
    print(fmt.format("Metric", m0["scene"], m1["scene"]))
    print("  " + "─" * (W - 2))

    def row(label, k, fmt_fn=str):
        v0 = fmt_fn(m0[k]) if k in m0 else "—"
        v1 = fmt_fn(m1[k]) if k in m1 else "—"
        print(fmt.format(label, v0, v1))

    row("Total points",          "n_total",       lambda v: f"{v:,}")
    row("Colored (%)",           "pct_colored",   lambda v: f"{v:.1f}%")
    row("Overlap-FOV pts (%)",   "pct_overlap",   lambda v: f"{v:.1f}%")
    row("Mean brightness",       "mean_brightness", lambda v: f"{v:.3f}")
    row("Mean saturation",       "mean_saturation", lambda v: f"{v:.3f}")
    row("Color entropy (bits)",  "color_entropy_bits", lambda v: f"{v:.2f}")
    row("Max radial range (m)",  "depth_range_m", lambda v: f"{v:.1f}")
    row("Points per sweep",      "pts_per_sweep", lambda v: f"{v:,}")
    row("Max |cam Δt| (ms)",     "max_cam_dt_ms", lambda v: f"{v:.1f}")
    row("Ego speed (m/s)",       "ego_speed_mps", lambda v: f"{v:.2f}" if v else "—")

    print("═" * W + "\n")


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ply",     default="colorized.ply")
    ap.add_argument("--scene",   default="scene-0061")
    ap.add_argument("--sample",  default="c923fe08b2ff4e27975d2bf30934383b")
    ap.add_argument("--sweeps",  type=int, default=5)
    ap.add_argument("--compare", nargs=2, metavar=("PLY0", "PLY1"),
                    help="Compare two PLY files side-by-side")
    ap.add_argument("--scenes",  nargs=2, metavar=("S0", "S1"),
                    default=["scene-0061", "scene-1094"])
    ap.add_argument("--samples", nargs=2, metavar=("T0", "T1"),
                    default=["c923fe08b2ff4e27975d2bf30934383b",
                             "273c930b3f384bd58b2bbae02fe81d1e"])
    args = ap.parse_args()

    if args.compare:
        print(f"Computing metrics for {args.scenes[0]}…")
        m0 = compute_metrics(args.compare[0], args.scenes[0],
                             args.samples[0], n_sweeps=5)
        print(f"Computing metrics for {args.scenes[1]}…")
        m1 = compute_metrics(args.compare[1], args.scenes[1],
                             args.samples[1], n_sweeps=13)
        print_report(m0, label=args.scenes[0])
        print_report(m1, label=args.scenes[1])
        print_comparison(m0, m1)
    else:
        print(f"Computing metrics for {args.scene}…")
        m = compute_metrics(args.ply, args.scene, args.sample, args.sweeps)
        print_report(m)


if __name__ == "__main__":
    main()
