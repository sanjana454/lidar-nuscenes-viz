"""
Shared utilities for the nuScenes LiDAR–camera fusion pipeline.

Centralises:
  - PLY I/O  (read_ply / write_ply_fast)
  - Z-buffer  (zbuffer_nearest)

All other scripts import from here so there is a single implementation
to audit, fix, and test.
"""

from pathlib import Path
from typing import Tuple

import numpy as np


# ── PLY I/O ────────────────────────────────────────────────────────────────

def read_ply(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read a binary-little-endian or ASCII PLY produced by this pipeline.

    Expected vertex layout:  float32 x y z  +  uint8 r g b

    Returns
    -------
    xyz : float32 (N, 3)
    rgb : uint8   (N, 3)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PLY not found: {path}")

    with open(path, "rb") as f:
        header_lines: list = []
        while True:
            line = f.readline().decode("utf-8").strip()
            header_lines.append(line)
            if line == "end_header":
                break

        header = "\n".join(header_lines)
        n = int(next(l.split()[-1] for l in header_lines
                     if l.startswith("element vertex")))
        is_binary = "binary_little_endian" in header

        if is_binary:
            dtype = np.dtype([
                ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                ("r", "u1"),  ("g", "u1"),  ("b", "u1"),
            ])
            raw = np.frombuffer(f.read(n * dtype.itemsize), dtype=dtype)
            xyz = np.stack([raw["x"], raw["y"], raw["z"]], axis=1)
            rgb = np.stack([raw["r"], raw["g"], raw["b"]], axis=1)
        else:
            data = np.loadtxt(f, max_rows=n)
            xyz = data[:, :3].astype(np.float32)
            rgb = data[:, 3:6].astype(np.uint8)

    return xyz, rgb


def write_ply_fast(path: str, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """
    Write a binary-little-endian PLY with XYZ + RGB using numpy structured
    arrays (single tobytes() call — no per-vertex Python loop).

    Vertex layout:  float32 x y z  +  uint8 r g b  (15 bytes/vertex)
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
    buf = np.zeros(n, dtype=dtype)
    buf["x"] = xyz[:, 0];  buf["y"] = xyz[:, 1];  buf["z"] = xyz[:, 2]
    buf["r"] = rgb[:, 0];  buf["g"] = rgb[:, 1];  buf["b"] = rgb[:, 2]

    with open(path, "wb") as f:
        f.write(header.encode("utf-8"))
        f.write(buf.tobytes())

    size_mb = Path(path).stat().st_size / 1e6
    print(f"Saved: {path}  ({n:,} points, {size_mb:.1f} MB)")


# ── Z-buffer ───────────────────────────────────────────────────────────────

def zbuffer_nearest(
    u: np.ndarray,
    v: np.ndarray,
    depth: np.ndarray,
    global_indices: np.ndarray,
    img_w: int,
    img_h: int,
) -> np.ndarray:
    """
    Z-buffer filter: for each image pixel return the index of the nearest
    projected point (smallest depth).

    Parameters
    ----------
    u, v            : float pixel coordinates (M,) for the M valid projections
    depth           : camera-frame depth (M,) — must be > 0
    global_indices  : original point indices (M,) into the full N-point array
    img_w, img_h    : image dimensions

    Returns
    -------
    visible_global : int array of global point indices that survive the z-buffer
                     (one per unique pixel; len ≤ M)

    Algorithm
    ---------
    Sort valid points near-first, then np.unique on flattened pixel coordinates
    returns the *first* occurrence per pixel — which is the nearest point.
    This is O(M log M) and fully vectorised.
    """
    ui = np.clip(u, 0, img_w - 1).astype(np.int32)
    vi = np.clip(v, 0, img_h - 1).astype(np.int32)

    # Ascending sort by depth → first occurrence per pixel = nearest
    order = np.argsort(depth)
    ui_s = ui[order];  vi_s = vi[order];  gi_s = global_indices[order]

    flat_s = vi_s * img_w + ui_s
    _, first_occ = np.unique(flat_s, return_index=True)

    return gi_s[first_occ]
