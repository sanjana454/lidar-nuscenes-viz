# LiDAR–Camera Fusion: Colorized Point Cloud

Fuses a nuScenes LiDAR sweep with all 6 cameras to produce a colorized 3-D point cloud, then renders it as a bird's-eye view (BEV), a perspective PNG, and a browser-based interactive 3-D viewer.

---

## Pipeline overview

```
LiDAR .pcd.bin  ──┐
                  ├──  colorize_pointcloud.py  ──►  colorized.ply
6× Camera .jpg  ──┘
                         │
          ┌──────────────┼──────────────────────┐
          ▼              ▼                      ▼
      bev_viz.py     fuse_viz.py           view_3d.py / render_3d.py
       bev.png       fusion_viz.png        viewer_3d.html / render_3d.png
```

### Coordinate transform chain

For each LiDAR point and each camera the full rigid-body chain is applied:

```
p_lidar  ──[cs_lidar]──►  p_ego_lidar  ──[ego_pose_lidar]──►  p_world
p_world  ──[ego_pose_cam⁻¹]──►  p_ego_cam  ──[cs_cam⁻¹]──►  p_cam
p_cam  ──[K]──►  (u, v)  ──►  sample RGB
```

Each sensor has its own ego-pose at its own timestamp, so camera–LiDAR timing offsets (up to ~40 ms) are handled correctly without approximation.

### Color assignment and sensor fusion challenges

| Challenge | Handling |
|-----------|----------|
| **Overlapping FOVs** | Per-camera z-buffer selects the nearest LiDAR point per pixel. Among cameras, the one where the point projects closest to the image optical centre ("best view") wins. |
| **Timing differences** | Each sweep is transformed to world frame using its *own* ego-pose timestamp; camera images are selected by nearest-timestamp lookup. |
| **Moving objects / ego-motion** | Adjacent sweeps (±~150 ms) are individually world-registered before aggregation, so ego motion is compensated. |
| **Occlusion / shadowing** | Z-buffer ensures only the nearest LiDAR point along each camera ray receives a color; occluded points remain uncolored rather than receiving a wrong color. |

---

## Approach

### Core idea

The goal is to assign a physically correct RGB color to every LiDAR point by finding which camera pixel that point projects onto. The challenge is that LiDAR and cameras are different sensors mounted at different positions, firing at slightly different times, and covering overlapping but non-identical fields of view.

The pipeline treats color assignment as a visibility problem: a LiDAR point should receive the color of whichever camera pixel it is *actually visible from*, accounting for depth, occlusion, and sensor geometry.

### Step 1 — Multi-sweep aggregation

A single LiDAR sweep (~34K points) is sparse. The pipeline collects up to N adjacent sweeps from the linked-list chain in `sample_data.json`, balanced ±N/2 around the key frame. Each sweep has its own timestamp and therefore its own ego-pose. Every sweep is independently transformed:

```
LiDAR frame → ego frame (lidar calibration) → world frame (ego pose at sweep timestamp)
```

Aggregating in world frame means ego motion between sweeps is automatically compensated — points from different sweeps are placed where they physically were in the world, not where the vehicle was at a different moment. With 5 sweeps (~300 ms window) the cloud grows to ~170K points with much better surface coverage.

### Step 2 — Full calibrated transform chain

For each camera the world-frame points are projected using the complete four-step rigid-body chain:

```
p_world
  → ego frame at camera timestamp     [ego_pose_cam⁻¹]
  → camera sensor frame               [calibrated_sensor_cam⁻¹]
  → image plane                       [camera intrinsic K]
```

Using the camera's *own* ego-pose (not the LiDAR's) is what makes timing-offset handling exact: the camera image was captured when the vehicle was at a slightly different position, and we transform to that position before projecting.

### Step 3 — Per-camera z-buffer

Multiple LiDAR points can project onto the same image pixel (a near point on a car and a far point on a wall, for example). Only the nearest point per pixel is a valid match — the farther ones are occluded from that camera's perspective. A z-buffer resolves this:

```python
order = np.argsort(depth)              # near-first
_, first_occ = np.unique(pixel_flat, return_index=True)   # first = nearest per pixel
```

Points that lose the z-buffer contest are left uncolored rather than assigned a wrong background color.

### Step 4 — Best-view multi-camera selection

All 6 cameras project some LiDAR points. In overlap zones the same point is visible in 2+ cameras. The pipeline keeps the color from the camera where the point projects *most centrally* — smallest normalized distance from the optical axis. Points at the periphery of a camera's FOV suffer more lens distortion and perspective stretching; a neighbouring camera that sees the same point more head-on produces a cleaner color.

```
centrality = pixel_distance_from_optical_centre / half_image_diagonal
```

A corner pixel scores ≈ 1.0; the image centre scores 0.0. The camera with the lowest centrality score for a given point wins. This is a dimensionless, resolution-independent metric that works correctly across cameras with different image dimensions.

### Design note — minimum projection depth

Points closer than `min_depth = 0.5 m` in camera space are rejected before projection. This guards against numerical instability (division by near-zero depth) and aligns with the physical minimum focus distance of the nuScenes cameras (~0.3 m). At the nuScenes LiDAR mounting height of 1.84 m, no scene surface legitimately falls within 0.5 m of any camera lens.

---

## Edge cases, artifacts, and mitigations

### 1. Depth-discontinuity ghosting (edge bleeding)

**What it is:** At the boundary between a near object (e.g. a car at 8 m) and the background (e.g. a wall at 25 m), a LiDAR beam can graze the object edge and return a point that sits geometrically at the near surface but projects onto a background pixel in the image. That point gets the background color even though it belongs to the foreground object.

**Mitigation implemented:** The z-buffer ensures that among all LiDAR points projecting to the *same* pixel, only the nearest gets the color. This eliminates same-pixel bleeding.

**Residual limitation:** If the grazing-return point projects to an *adjacent* pixel rather than the exact same one as the foreground point, the z-buffer cannot help. The point lands one pixel into the background texture.

**Given more time:** Apply a depth-discontinuity mask — flag pixels where the image depth gradient is large (edge of an object) and reject LiDAR points that project there but whose depth differs significantly from the local z-buffer minimum. Alternatively, dilate the foreground LiDAR mask by 1–2 pixels before sampling to "pull in" nearby background colors.

---

### 2. Timing offsets between LiDAR and cameras

**What it is:** nuScenes cameras fire asynchronously from the LiDAR, with offsets of up to ~40 ms per camera. At highway speed (~30 m/s) a 40 ms offset means ~1.2 m of ego motion and proportional displacement of moving scene objects.

**Mitigation implemented:** Each camera's image is selected by nearest-timestamp lookup, and the world→camera transform uses that camera's *own* ego-pose, not the LiDAR's. Ego motion of the recording vehicle is therefore exactly compensated — a stationary wall will project correctly regardless of timing.

**Residual limitation:** Other moving objects (pedestrians, cars) are not ego-vehicle, so their positions in the LiDAR sweep and in the camera image can differ by up to ~0.5 m at typical urban speeds. LiDAR points on a moving car may land slightly off the car's image footprint.

**Given more time:** Track detected dynamic objects across timestamps and apply per-object rigid-body compensation using velocity estimates from the nuScenes annotation boxes. For objects without annotations, a scene-flow or optical-flow based approach could estimate pixel-level displacements.

---

### 3. Overlapping camera FOVs — color consistency

**What it is:** Adjacent cameras (e.g. CAM_FRONT and CAM_FRONT_LEFT) share a ~10–15° overlap region. A LiDAR point in that zone will project validly into both images, which may have slightly different exposures, white-balance, or perspective angles. Naively picking either one produces inconsistent colors at the seam.

**Mitigation implemented:** The centrality score picks the camera that sees the point most squarely, which tends to minimize both perspective distortion and vignetting. The result is smooth rather than having a hard seam at the camera boundary.

**Given more time:** Radiometric calibration across cameras (normalising exposure and white-balance) would further reduce color discontinuities at seams. A blending weight proportional to angular distance from the FOV centre could also smoothly interpolate colors in the overlap zone.

---

### 4. Uncolored points (~40% of the cloud)

**What it is:** After fusion, roughly 40% of points remain black (no camera sees them). These fall into two categories: (a) points directly above or below the camera coverage band (cameras are mounted at a fixed height and have a limited vertical FOV), and (b) points occluded in every camera by nearer surfaces.

**Mitigation implemented:** Aggregating 5 sweeps instead of 1 increases coverage by ~5% because the vehicle moves slightly between sweeps, exposing previously occluded surfaces to at least one camera. The `--no-uncolored` flag can strip these points from the output if a clean-colored cloud is preferred.

**Given more time:** Height-adaptive colorization — for ground-plane points that no camera covers, assign a synthetic color based on the LiDAR intensity return or a learned ground-texture model. For occluded points, nearest-neighbor color propagation from the colored surface nearby would give a plausible result.

---

### 5. LiDAR ring-pattern artifacts in the BEV

**What it is:** A single LiDAR sweep has 32 rings at fixed elevation angles. In the BEV this creates concentric arc patterns rather than dense surface fills, most visible on flat ground.

**Mitigation implemented:** Multi-sweep aggregation fills in the inter-ring gaps. With 5 sweeps the pattern largely disappears on flat surfaces.

**Given more time:** Point cloud upsampling (e.g. PU-Net or simple radial interpolation) or ground-plane fitting with dense rasterization would produce a more film-like BEV without visible ring artifacts.

---

## Requirements

```bash
pip install numpy Pillow scipy matplotlib
```

The dataset must be **nuScenes v1.0-mini** placed under `data/`:

```
data/
  v1.0-mini/          ← JSON metadata
  samples/            ← key-frame images + LiDAR .pcd.bin
  sweeps/             ← intermediate sweep files
```

---

## Scripts

### 1. `colorize_pointcloud.py` — core fusion pipeline

Produces a colorized PLY point cloud by fusing LiDAR with all 6 cameras.

```bash
python colorize_pointcloud.py \
    --scene   scene-0061 \
    --sample  c923fe08b2ff4e27975d2bf30934383b \
    --sweeps  5 \
    --out     colorized.ply \
    --viz
```

| Flag | Default | Description |
|------|---------|-------------|
| `--scene` | `scene-0061` | Scene name |
| `--sample` | first in scene | Override specific sample token |
| `--sweeps` | `5` | Number of LiDAR sweeps to aggregate (balanced ±) |
| `--out` | `colorized.ply` | Output binary PLY path |
| `--viz` | off | Write top-down PNG + per-camera projection PNGs |
| `--no-uncolored` | off | Strip uncolored points from output PLY |

**Output:** binary PLY (`float32 x y z` + `uint8 r g b` per vertex).

---

### 2. `bev_viz.py` — bird's-eye view

Renders the colorized PLY from above with camera frustum cones, range rings, ego-vehicle footprint, and overlap-zone annotations.

```bash
python bev_viz.py \
    --ply    colorized.ply \
    --scene  scene-0061 \
    --sample c923fe08b2ff4e27975d2bf30934383b \
    --out    bev.png \
    --range  60
```

| Flag | Default | Description |
|------|---------|-------------|
| `--ply` | `colorized.ply` | Input PLY |
| `--range` | `60` | Display radius in metres |
| `--dpi` | `180` | Output image DPI |

**Output:** `bev.png` — top-down view with frustum cones, overlap zones, range rings, and ego vehicle.

---

### 3. `fuse_viz.py` — camera-image overlay

Renders all 6 camera images in a 2×3 grid (front-left / front / front-right on top; rear cameras below) with LiDAR depth overlaid. Timing offsets and overlap count are annotated per panel.

```bash
python fuse_viz.py \
    --scene   scene-0061 \
    --sample  c923fe08b2ff4e27975d2bf30934383b \
    --sweeps  5 \
    --out     fusion_viz.png
```

**Output:** `fusion_viz.png`.

---

### 4. `view_3d.py` — interactive browser viewer

Encodes the PLY as a base64 blob and writes a self-contained HTML file rendered in WebGL (Three.js). Open `viewer_3d.html` in any modern browser for full orbit / zoom / pan.

```bash
python view_3d.py --ply colorized.ply --out viewer_3d.html --point-size 1.5
```

**Output:** `viewer_3d.html` (~3.5 MB).

---

### 5. `render_3d.py` — static 3-D perspective PNG

Software perspective renderer (no 3-D library required). Produces a 4-panel PNG: perspective · front · side · top-down.

```bash
python render_3d.py --ply colorized.ply --out render_3d.png
```

### 6. `utils.py` — shared utilities

Single-source-of-truth module imported by all scripts. Defines:

| Symbol | Purpose |
|--------|---------|
| `read_ply(path)` | Binary or ASCII PLY reader → `(xyz float32, rgb uint8)` |
| `write_ply_fast(path, xyz, rgb)` | Binary PLY writer via numpy `tobytes()` — no per-vertex loop |
| `zbuffer_nearest(u, v, depth, gi, W, H)` | O(M log M) z-buffer: returns global indices of nearest point per pixel |

Centralising these eliminates the maintenance risk of diverging implementations across scripts.

---

## Quick start

```bash
# 1. Fuse LiDAR with cameras → colorized.ply
python colorize_pointcloud.py --scene scene-0061 --sweeps 5 --out colorized.ply

# 2. Bird's-eye view
python bev_viz.py --ply colorized.ply --out bev.png

# 3. All-camera overlay
python fuse_viz.py --out fusion_viz.png

```

---

## Output files

| File | Description |
|------|-------------|
| `colorized.ply` | Colorized point cloud — binary PLY, ~2.6 MB |
| `bev.png` | Bird's-eye view with camera frustums and range rings |
| `fusion_viz.png` | 6-camera image overlay with LiDAR depth |
| `render_3d.png` | Static 4-panel 3-D perspective render |
| `viz_top_down.png` | Simple top-down render (from `--viz` flag) |
| `viz_projections/` | Per-camera projection sanity-check images |