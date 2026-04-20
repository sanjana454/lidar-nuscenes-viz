#!/usr/bin/env python3
"""
nuScenes LiDAR-Camera Colorization Pipeline.

Fuses LiDAR point clouds with multi-camera imagery using:
  - Full calibrated sensor extrinsics (LiDAR → ego → world → ego → camera)
  - Per-camera z-buffer occlusion filtering for crisp color assignment
  - Multi-camera coverage with best-view selection (most centred projection wins)
  - Optional multi-sweep aggregation for higher point density

Usage:
    python colorize_pointcloud.py [--scene scene-0061] [--sweeps 5] [--out out.ply]
"""

import argparse
import bisect
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

DATA_ROOT = Path("data")
META_ROOT = DATA_ROOT / "v1.0-mini"

# Camera channel order — processed in priority order for tie-breaking.
# Front cameras have higher intrinsic quality and are the primary reference.
CAM_CHANNELS = [
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]

# ── Geometry helpers ──────────────────────────────────────────────────────────

def quat_to_R(q: List[float]) -> np.ndarray:
    """Quaternion [w, x, y, z] → 3×3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - w*z),       2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def rigid_transform(pts: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Apply rigid body transform R, t to Nx3 points."""
    return (R @ pts.T).T + t


def rigid_transform_inv(pts: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Apply inverse rigid body transform (R^T, -R^T t) to Nx3 points."""
    return (R.T @ (pts - t).T).T


# ── Data loading ──────────────────────────────────────────────────────────────

def load_json(name: str) -> List[Dict]:
    with open(META_ROOT / name) as f:
        return json.load(f)


def build_lookup(records: List[Dict], key: str = "token") -> Dict[str, Dict]:
    return {r[key]: r for r in records}


def load_lidar_bin(filepath: str) -> np.ndarray:
    """Load nuScenes LiDAR binary → Nx5 float32 (x, y, z, intensity, ring)."""
    pts = np.fromfile(filepath, dtype=np.float32).reshape(-1, 5)
    return pts


# ── Projection & colorization ─────────────────────────────────────────────────

def project_points(
    pts_cam: np.ndarray,
    K: np.ndarray,
    img_w: int,
    img_h: int,
    min_depth: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project Nx3 camera-frame points to pixel coordinates.

    Returns:
        u, v   : float pixel coordinates (N,)
        valid  : boolean mask — in front of camera AND inside image
    """
    depth = pts_cam[:, 2]
    in_front = depth > min_depth

    # Avoid divide-by-zero for masked points
    safe_z = np.where(in_front, depth, 1.0)
    u = K[0, 0] * pts_cam[:, 0] / safe_z + K[0, 2]
    v = K[1, 1] * pts_cam[:, 1] / safe_z + K[1, 2]

    in_image = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
    valid = in_front & in_image
    return u, v, valid


def zbuffer_assign(
    pts_cam: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    valid: np.ndarray,
    img: np.ndarray,
    img_w: int,
    img_h: int,
    K: np.ndarray,
    shadow_margin: int = 2,
    shadow_depth_ratio: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Z-buffer pass with shadow-margin filtering.

    Step 1 — Z-buffer: for each pixel keep only the nearest projected point.

    Step 2 — Shadow-margin filter: reject points whose depth is much larger
    than the minimum depth in any neighbouring pixel within `shadow_margin`
    pixels.  These are background returns sitting in the "shadow cone" behind
    a near foreground edge — they would receive the foreground pixel's color
    if left unfiltered, producing edge bleeding / ghosting.

    Specifically: a point at depth d is rejected if there exists a neighbour
    pixel whose z-buffer depth is < d / shadow_depth_ratio.  The ratio of 2.0
    means a point is only rejected when a foreground surface at least half as
    close exists nearby — conservative enough to keep legitimate background
    points in open areas while stripping shadow artifacts at object edges.

    Returns:
        rgb        : Nx3 uint8 (zeros where not assigned)
        assigned   : boolean mask of points that received a color
        centrality : float (normalised dist from image centre, 0 = best view)
    """
    n = len(pts_cam)
    depth = pts_cam[:, 2]

    ui = u[valid].astype(np.int32)
    vi = v[valid].astype(np.int32)
    d  = depth[valid]
    idx = np.where(valid)[0]

    # ── Step 1: z-buffer — sort near-first, first occurrence = nearest ────
    order = np.argsort(d)
    ui_s, vi_s, d_s, idx_s = ui[order], vi[order], d[order], idx[order]

    flat_s = vi_s * img_w + ui_s
    _, first_occ = np.unique(flat_s, return_index=True)

    ui_u  = ui_s[first_occ]
    vi_u  = vi_s[first_occ]
    d_u   = d_s[first_occ]
    idx_u = idx_s[first_occ]

    # ── Step 2: shadow-margin filter ──────────────────────────────────────
    # Build a floating-point depth image (inf where no point projects).
    depth_img = np.full((img_h, img_w), np.inf, dtype=np.float32)
    depth_img[vi_u, img_w - 1 - (img_w - 1 - ui_u)] = d_u   # same as depth_img[vi_u, ui_u]
    depth_img[vi_u, ui_u] = d_u

    # Minimum depth in a (2r+1)×(2r+1) neighbourhood via a sliding-window
    # minimum — implemented with a simple dilation over the depth image.
    # scipy.ndimage is available; use minimum_filter for speed.
    from scipy.ndimage import minimum_filter
    r = shadow_margin
    local_min = minimum_filter(depth_img, size=2 * r + 1, mode="constant",
                               cval=np.inf)

    # A z-buffer winner is a shadow candidate if a much nearer point exists
    # within shadow_margin pixels.
    shadowed = local_min[vi_u, ui_u] * shadow_depth_ratio < d_u

    # Keep only non-shadowed winners
    keep = ~shadowed
    idx_u_final = idx_u[keep]

    assigned_mask = np.zeros(n, dtype=bool)
    assigned_mask[idx_u_final] = True
    assigned_mask &= valid

    # ── Color sampling: clip floats before int cast ────────────────────────
    rgb = np.zeros((n, 3), dtype=np.uint8)
    if assigned_mask.any():
        u_a = np.clip(u[assigned_mask], 0, img_w - 1).astype(np.int32)
        v_a = np.clip(v[assigned_mask], 0, img_h - 1).astype(np.int32)
        rgb[assigned_mask] = img[v_a, u_a]

    # ── Centrality: dist from optical centre / half-diagonal ──────────────
    # 0.0 = optical centre (best view), ≈1.0 = image corner (worst view).
    cx, cy = K[0, 2], K[1, 2]
    half_diag = np.sqrt(cx**2 + cy**2)
    dist = np.sqrt((u - cx)**2 + (v - cy)**2)
    centrality = np.where(valid, dist / half_diag, np.inf)

    return rgb, assigned_mask, centrality


# ── Main fusion pipeline ──────────────────────────────────────────────────────

class NuScenesColorizer:

    def __init__(self, data_root: str = "data"):
        self.root = Path(data_root)
        self._load_metadata()

    def _load_metadata(self):
        scenes        = load_json("scene.json")
        samples       = load_json("sample.json")
        sample_data   = load_json("sample_data.json")
        cal_sensors   = load_json("calibrated_sensor.json")
        ego_poses     = load_json("ego_pose.json")
        sensors       = load_json("sensor.json")

        self.scenes       = build_lookup(scenes,      "token")
        self.scenes_by_name = {s["name"]: s for s in scenes}
        self.samples      = build_lookup(samples,     "token")
        self.sample_data  = build_lookup(sample_data, "token")
        self.cal_sensors  = build_lookup(cal_sensors, "token")
        self.ego_poses    = build_lookup(ego_poses,   "token")
        self.sensors      = build_lookup(sensors,     "token")

        # Build channel-name → sensor_token mapping
        self.channel_to_sensor_token = {s["channel"]: s["token"] for s in sensors}

        # sample_token → list of sample_data tokens
        self.sample_to_sd: Dict[str, List[str]] = {}
        for sd in sample_data:
            self.sample_to_sd.setdefault(sd["sample_token"], []).append(sd["token"])

        # For each sensor channel, build sorted list of (timestamp, sd_token)
        self.channel_timeline: Dict[str, List[Tuple[int, str]]] = {}
        for sd in sample_data:
            cs = self.cal_sensors[sd["calibrated_sensor_token"]]
            sensor = self.sensors[cs["sensor_token"]]
            ch = sensor["channel"]
            self.channel_timeline.setdefault(ch, []).append(
                (sd["timestamp"], sd["token"])
            )
        for ch in self.channel_timeline:
            self.channel_timeline[ch].sort()

    # ── Lookup helpers ───────────────────────────────────────────────────────

    def get_sample_data_for_channel(
        self, sample_token: str, channel: str
    ) -> Optional[Dict]:
        """Return sample_data entry matching sample + channel."""
        for sd_tok in self.sample_to_sd.get(sample_token, []):
            sd = self.sample_data[sd_tok]
            cs = self.cal_sensors[sd["calibrated_sensor_token"]]
            sensor = self.sensors[cs["sensor_token"]]
            if sensor["channel"] == channel:
                return sd
        return None

    def nearest_sd_by_time(self, channel: str, timestamp: int) -> Optional[Dict]:
        """Return sample_data for the channel-frame nearest to timestamp.

        Uses bisect on the pre-sorted timeline for O(log n) lookup instead of
        building a numpy array and doing a full O(n) argmin scan.
        """
        timeline = self.channel_timeline.get(channel, [])
        if not timeline:
            return None
        # timeline is sorted by timestamp; bisect finds the insertion point
        ts_list = [t for t, _ in timeline]
        pos = bisect.bisect_left(ts_list, timestamp)
        # compare neighbours and pick closer one
        if pos == 0:
            idx = 0
        elif pos >= len(timeline):
            idx = len(timeline) - 1
        else:
            before = timestamp - ts_list[pos - 1]
            after  = ts_list[pos] - timestamp
            idx = pos - 1 if before <= after else pos
        return self.sample_data[timeline[idx][1]]

    def adjacent_lidar_sds(
        self, lidar_sd: Dict, n_total: int = 5, max_dt_us: int = 300_000
    ) -> List[Dict]:
        """
        Collect up to n_total LIDAR sweeps centred on lidar_sd.

        Walks the prev/next linked list symmetrically, but when one direction
        is exhausted the remaining quota is drawn from the other direction.
        Only sweeps within max_dt_us microseconds of the key frame are included.
        """
        ref_ts = lidar_sd["timestamp"]

        before, after = [], []
        sd = lidar_sd
        while sd.get("prev") and len(before) < n_total:
            sd = self.sample_data[sd["prev"]]
            if abs(sd["timestamp"] - ref_ts) > max_dt_us:
                break
            before.append(sd)

        sd = lidar_sd
        while sd.get("next") and len(after) < n_total:
            sd = self.sample_data[sd["next"]]
            if abs(sd["timestamp"] - ref_ts) > max_dt_us:
                break
            after.append(sd)

        # Balance: take up to half from each side, fill remainder from the other
        half = (n_total - 1) // 2
        taken_b = before[:half]
        taken_a = after[:half]
        remaining = (n_total - 1) - len(taken_b) - len(taken_a)
        # Use leftover quota from whichever side has more available
        if len(before) > half:
            taken_b += before[half: half + remaining]
        elif len(after) > half:
            taken_a += after[half: half + remaining]

        return taken_b + [lidar_sd] + taken_a

    # ── Transform chain ──────────────────────────────────────────────────────

    def lidar_to_world(
        self, pts: np.ndarray, lidar_sd: Dict
    ) -> np.ndarray:
        """LiDAR sensor frame → world frame using lidar's own ego pose."""
        cs = self.cal_sensors[lidar_sd["calibrated_sensor_token"]]
        ep = self.ego_poses[lidar_sd["ego_pose_token"]]

        R_cs = quat_to_R(cs["rotation"])
        t_cs = np.array(cs["translation"])
        R_ep = quat_to_R(ep["rotation"])
        t_ep = np.array(ep["translation"])

        pts_ego   = rigid_transform(pts, R_cs, t_cs)      # lidar → ego
        pts_world = rigid_transform(pts_ego, R_ep, t_ep)  # ego   → world
        return pts_world

    def world_to_camera(
        self, pts_world: np.ndarray, cam_sd: Dict
    ) -> np.ndarray:
        """World frame → camera sensor frame using camera's ego pose + extrinsic."""
        cs = self.cal_sensors[cam_sd["calibrated_sensor_token"]]
        ep = self.ego_poses[cam_sd["ego_pose_token"]]

        R_ep = quat_to_R(ep["rotation"])
        t_ep = np.array(ep["translation"])
        R_cs = quat_to_R(cs["rotation"])
        t_cs = np.array(cs["translation"])

        pts_ego = rigid_transform_inv(pts_world, R_ep, t_ep)  # world → ego
        pts_cam = rigid_transform_inv(pts_ego, R_cs, t_cs)    # ego   → camera
        return pts_cam

    # ── Public API ───────────────────────────────────────────────────────────

    def colorize(
        self,
        scene_name: str,
        sample_token: Optional[str] = None,
        n_sweeps: int = 1,
        shadow_margin: int = 2,
        shadow_depth_ratio: float = 2.0,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Produce a colorized point cloud for one key sample.

        Args:
            scene_name   : e.g. "scene-0061"
            sample_token : override to pick a specific sample; if None uses first
            n_sweeps     : total LIDAR sweeps to aggregate (1 = key frame only)
            verbose      : print progress

        Returns:
            xyz   : (N, 3) float32 — point positions in world frame
            rgb   : (N, 3) uint8  — RGB colors (0 = uncolored)
        """
        scene = self.scenes_by_name[scene_name]
        if sample_token is None:
            sample_token = scene["first_sample_token"]

        if verbose:
            print(f"Scene: {scene_name}  sample: {sample_token[:8]}…")

        # ── 1. Collect LIDAR sweeps ──────────────────────────────────────────
        lidar_sd = self.get_sample_data_for_channel(sample_token, "LIDAR_TOP")
        assert lidar_sd, "No LIDAR_TOP sample_data found for this sample."

        sweep_sds = self.adjacent_lidar_sds(lidar_sd, n_total=n_sweeps)
        if verbose:
            print(f"  LIDAR sweeps: {len(sweep_sds)}")

        # ── 2. Load and aggregate all sweeps in world frame ─────────────────
        all_xyz_world = []
        for sd in sweep_sds:
            pts = load_lidar_bin(str(self.root / sd["filename"]))[:, :3]  # x,y,z
            pts_world = self.lidar_to_world(pts, sd)
            all_xyz_world.append(pts_world)

        xyz_world = np.concatenate(all_xyz_world, axis=0).astype(np.float64)
        n_pts = len(xyz_world)
        if verbose:
            print(f"  Total points: {n_pts:,}")

        # ── 3. Per-point tracking ────────────────────────────────────────────
        final_rgb        = np.zeros((n_pts, 3), dtype=np.uint8)
        final_centrality = np.full(n_pts, np.inf, dtype=np.float64)

        # ── 4. Project into each camera ──────────────────────────────────────
        ref_ts = lidar_sd["timestamp"]  # reference timestamp for camera selection

        for ch in CAM_CHANNELS:
            # Find camera frame nearest in time to the key LiDAR frame
            cam_sd = self.nearest_sd_by_time(ch, ref_ts)
            if cam_sd is None:
                if verbose:
                    print(f"  [{ch}] no data — skip")
                continue

            cam_cs = self.cal_sensors[cam_sd["calibrated_sensor_token"]]
            K_list = cam_cs["camera_intrinsic"]
            if not K_list:
                continue
            K = np.array(K_list, dtype=np.float64)

            img_path = self.root / cam_sd["filename"]
            if not img_path.exists():
                if verbose:
                    print(f"  [{ch}] image missing — skip")
                continue

            img = np.array(Image.open(img_path).convert("RGB"))
            img_h, img_w = img.shape[:2]

            # Transform world points to camera frame
            pts_cam = self.world_to_camera(xyz_world, cam_sd)

            u, v, valid = project_points(pts_cam, K, img_w, img_h)

            if not valid.any():
                if verbose:
                    print(f"  [{ch}] 0 valid projections")
                continue

            rgb_cam, assigned, centrality = zbuffer_assign(
                pts_cam, u, v, valid, img, img_w, img_h, K,
                shadow_margin=shadow_margin,
                shadow_depth_ratio=shadow_depth_ratio,
            )

            # Update global color where this camera gives a better (more centred) view
            improve = assigned & (centrality < final_centrality)
            final_rgb[improve]        = rgb_cam[improve]
            final_centrality[improve] = centrality[improve]

            n_colored = improve.sum()
            if verbose:
                print(f"  [{ch}] valid={valid.sum():,}  assigned={assigned.sum():,}  "
                      f"improved={n_colored:,}")

        colored_mask = final_centrality < np.inf
        if verbose:
            pct = 100.0 * colored_mask.sum() / n_pts
            print(f"  Colored: {colored_mask.sum():,}/{n_pts:,} ({pct:.1f}%)")

        return xyz_world.astype(np.float32), final_rgb


# ── PLY writer ────────────────────────────────────────────────────────────────

def write_ply(path: str, xyz: np.ndarray, rgb: np.ndarray):
    """Write ASCII PLY with XYZ + RGB."""
    n = len(xyz)
    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        data = np.hstack([xyz.astype(np.float32), rgb.astype(np.uint8)])
        fmt = "%.6f %.6f %.6f %d %d %d"
        np.savetxt(f, data, fmt=fmt)
    print(f"Saved: {path}  ({n:,} points)")


def write_ply_binary(path: str, xyz: np.ndarray, rgb: np.ndarray):
    """Alias kept for backwards compatibility — delegates to write_ply_fast."""
    write_ply_fast(path, xyz, rgb)


def write_ply_fast(path: str, xyz: np.ndarray, rgb: np.ndarray):
    """
    Write binary PLY efficiently using numpy structured arrays.
    Much faster than the per-vertex loop version.
    """
    n = len(xyz)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    dtype = np.dtype([
        ("x", np.float32), ("y", np.float32), ("z", np.float32),
        ("r", np.uint8),   ("g", np.uint8),   ("b", np.uint8),
    ])
    data = np.zeros(n, dtype=dtype)
    data["x"] = xyz[:, 0]
    data["y"] = xyz[:, 1]
    data["z"] = xyz[:, 2]
    data["r"] = rgb[:, 0]
    data["g"] = rgb[:, 1]
    data["b"] = rgb[:, 2]
    with open(path, "wb") as f:
        f.write(header.encode("utf-8"))
        f.write(data.tobytes())
    print(f"Saved: {path}  ({n:,} points)")


# ── Visualization ─────────────────────────────────────────────────────────────

def visualize_top_down(
    xyz: np.ndarray,
    rgb: np.ndarray,
    out_path: str = "viz_top_down.png",
    resolution: float = 0.1,
    z_min: float = -2.5,
    z_max: float = 8.0,
    margin_m: float = 1.0,
):
    """
    Render a crisp bird's-eye-view image of the colorized point cloud.

    Strategy:
      - Only colored points are rendered (uncolored = black are excluded).
      - Each grid cell takes the color of the highest-z point inside it
        (top-down visibility: higher objects occlude the ground).
      - A dark-grey background makes colors pop.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colored = (rgb > 0).any(axis=1)
    mask = colored & (xyz[:, 2] >= z_min) & (xyz[:, 2] <= z_max)
    pts = xyz[mask]
    col = rgb[mask]

    if len(pts) == 0:
        print("  [top-down] no colored points to render")
        return

    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    x0, y0 = x.min() - margin_m, y.min() - margin_m
    xi = ((x - x0) / resolution).astype(np.int32)
    yi = ((y - y0) / resolution).astype(np.int32)

    W = xi.max() + 1
    H = yi.max() + 1
    canvas_rgb = np.full((H, W, 3), 30, dtype=np.uint8)   # dark grey background
    canvas_z   = np.full((H, W), -np.inf, dtype=np.float32)

    # Vectorised: sort ascending by z, then scatter-assign (last write wins = highest)
    order = np.argsort(z)
    xi_s, yi_s = xi[order], yi[order]
    z_s  = z[order]
    col_s = col[order]

    canvas_z[yi_s, xi_s]      = z_s        # numpy fancy-index: last write per cell wins
    canvas_rgb[yi_s, xi_s]    = col_s

    fig, ax = plt.subplots(figsize=(16, 16), dpi=150)
    ax.imshow(canvas_rgb, origin="lower", interpolation="nearest")
    ax.set_title("Colorized LiDAR — bird's-eye view (colored points only)", fontsize=13)
    ax.set_xlabel(f"X  ({resolution:.2f} m/px)")
    ax.set_ylabel(f"Y  ({resolution:.2f} m/px)")
    # Mark ego vehicle position (origin of LiDAR = centre of scan)
    ego_xi = int((-x0) / resolution)
    ego_yi = int((-y0) / resolution)
    ax.plot(ego_xi, ego_yi, "r+", markersize=18, markeredgewidth=2.5, label="ego")
    ax.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Top-down visualization: {out_path}")


def visualize_camera_projections(
    colorizer: "NuScenesColorizer",
    sample_token: str,
    xyz_world: np.ndarray,
    rgb: np.ndarray,
    out_dir: str = "viz_projections",
):
    """
    For each camera, draw projected LiDAR points (coloured by distance)
    overlaid on the camera image — a standard calibration sanity check.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    os.makedirs(out_dir, exist_ok=True)
    lidar_sd = colorizer.get_sample_data_for_channel(sample_token, "LIDAR_TOP")
    ref_ts = lidar_sd["timestamp"]

    for ch in CAM_CHANNELS:
        cam_sd = colorizer.nearest_sd_by_time(ch, ref_ts)
        if cam_sd is None:
            continue
        cam_cs = colorizer.cal_sensors[cam_sd["calibrated_sensor_token"]]
        K = np.array(cam_cs["camera_intrinsic"], dtype=np.float64)
        img_path = colorizer.root / cam_sd["filename"]
        if not img_path.exists():
            continue

        img = np.array(Image.open(img_path).convert("RGB"))
        img_h, img_w = img.shape[:2]

        pts_cam = colorizer.world_to_camera(xyz_world.astype(np.float64), cam_sd)
        u, v, valid = project_points(pts_cam, K, img_w, img_h)

        fig, ax = plt.subplots(figsize=(16, 9), dpi=100)
        ax.imshow(img)
        if valid.any():
            depth = pts_cam[valid, 2]
            sc = ax.scatter(
                u[valid], v[valid],
                c=depth, cmap="plasma_r",
                s=1.5, alpha=0.7, vmin=0, vmax=50,
                linewidths=0,
            )
            plt.colorbar(sc, ax=ax, label="Depth (m)", fraction=0.02)
        ax.set_title(f"{ch} — LiDAR projection overlay", fontsize=12)
        ax.axis("off")
        out = os.path.join(out_dir, f"{ch}.png")
        plt.tight_layout()
        plt.savefig(out, dpi=100, bbox_inches="tight")
        plt.close()
        print(f"  Projection viz: {out}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene",    default="scene-0061",
                   help="Scene name (default: scene-0061)")
    p.add_argument("--sample",   default=None,
                   help="Override sample token (default: first in scene)")
    p.add_argument("--sweeps",   type=int, default=5,
                   help="Number of LIDAR sweeps to aggregate (default: 5)")
    p.add_argument("--out",      default="colorized.ply",
                   help="Output PLY file (default: colorized.ply)")
    p.add_argument("--viz",      action="store_true",
                   help="Generate visualization images")
    p.add_argument("--no-uncolored", action="store_true",
                   help="Exclude uncolored points from output")
    p.add_argument("--shadow-margin", type=int, default=2,
                   help="Shadow filter neighbourhood radius in pixels (default: 2). "
                        "Set 0 to disable.")
    p.add_argument("--shadow-ratio", type=float, default=2.0,
                   help="A point is rejected as shadowed if a neighbour within "
                        "--shadow-margin px is this many times closer (default: 2.0).")
    return p.parse_args()


def main():
    args = parse_args()

    colorizer = NuScenesColorizer(data_root="data")

    xyz, rgb = colorizer.colorize(
        scene_name=args.scene,
        sample_token=args.sample,
        n_sweeps=args.sweeps,
        shadow_margin=args.shadow_margin,
        shadow_depth_ratio=args.shadow_ratio,
        verbose=True,
    )

    if args.no_uncolored:
        colored = (rgb > 0).any(axis=1)
        xyz, rgb = xyz[colored], rgb[colored]
        print(f"  Retained colored-only: {len(xyz):,} points")

    write_ply_fast(args.out, xyz, rgb)

    if args.viz:
        print("\nGenerating visualizations…")
        visualize_top_down(xyz, rgb, out_path="viz_top_down.png")

        # Re-derive sample token for projection viz
        scene = colorizer.scenes_by_name[args.scene]
        sample_token = args.sample or scene["first_sample_token"]
        visualize_camera_projections(
            colorizer, sample_token, xyz.astype(np.float64), rgb,
            out_dir="viz_projections",
        )


if __name__ == "__main__":
    main()
