# Camera calibration and 3D wound reconstruction

**Dilip Goswami · ARAIAS technical take-home**

A CPU photogrammetry pipeline for calibrating an iPhone camera and reconstructing a
silicone wound moulage from RGB photographs. It produces calibrated cameras, sparse
and dense point clouds, a vertex-coloured mesh and inspectable PLY exports.
Experimental wound descriptors include boundary and reference-surface sensitivity.

**Status:** research prototype. Physical scale and independent surface accuracy are
unresolved. The supplied report and model archive describe `175d788`; correctness
and integrity fixes were introduced in `c74e455`.

[Gallery](#reconstruction-gallery) · [Quick start](#quick-start) · [Method](#method) · [Results](#results) ·
[Inspect outputs](#inspect-outputs) · [Physical scale](#physical-scale) ·
[Revisions](#revisions-and-reproducibility) · [Development](#development)

## Reconstruction gallery

Selected renders from the **submitted archive (`175d788`)**, matching the report's
158,786-vertex mesh. The subject is a silicone wound phantom. Physical scale is
unresolved; the depth colours describe an inferred reference surface.

<p align="center">
  <img src="assets/previews/05_textured_oblique.png" width="780" alt="Vertex-coloured reconstructed silicone wound mesh, oblique view; scale unknown">
</p>

| Surface geometry | Experimental depth |
|---|---|
| ![Neutral oblique rendering of the reconstructed mesh](assets/previews/02_geometry_oblique.png) | ![Experimental wound depth relative to the fitted intact-surface reference](assets/previews/03_wound_depth.png) |
| Neutral shading exposes relief and open outer boundaries. | Purple is zero; yellow is P99, about 3.27% of wound length. |

Open an image for full resolution. The preview files are unchanged pipeline outputs;
[PLY files and inspection instructions](#inspect-outputs) are described below.

## Quick start

Use **Python 3.13** for the pinned environment. The reference run used Python 3.13.5
on macOS arm64 with four OpenCV threads. The package declares Python 3.12+ support;
the supplied lock file was resolved for Python 3.13. No GPU, model training,
pretrained weights or remote image service is required.

Keep the supplied data outside the repository:

```text
workspace/
├── checkerboard/                  # 22 calibration photographs
├── images/                        # 23 wound photographs
└── araias-wound-reconstruction/    # source repository
```

From the repository root:

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
.venv/bin/python -m pip install --no-deps -e '.[dev]'

.venv/bin/wound-reconstruct --data-root .. --output-dir outputs --threads 4
.venv/bin/wound-verify --output-dir outputs --data-root ..
```

Use a fresh output directory per full run and a separate virtual environment per
checkout. Configuration is in `configs/default.yaml`; packaged defaults are in
`src/woundrecon/resources/default.yaml`. Use one OpenCV distribution:
`opencv-python-headless`.

The archived reference recorded **14.1 minutes of stage computation**, excluding
startup and verification. This is an observed runtime, not a controlled benchmark.
Logs and `run_manifest.json` record stage outcomes, timings, settings, source/input
hashes, library versions and artifact hashes. PDF typesetting is separate.

## Method

```text
audit → calibrate → sfm → dense → mesh → scale → measure → render
```

| Stage | Responsibility |
|---|---|
| `audit` | Check resolution, orientation, colour profile, exposure, sharpness and highlights |
| `calibrate` | Detect corners; compare five lens models on held-out views; bootstrap uncertainty |
| `sfm` | RootSIFT matching, verified tracks, incremental camera registration and robust bundle adjustment |
| `dense` | Rectified SGBM stereo; try ranked reserves until six usable partners or candidate exhaustion; fuse depths agreeing across two pairs |
| `mesh` | TSDF fusion and marching cubes; require all eight cell corners to have support from at least two depth maps |
| `scale` | Validate a supplied scale reference and align geometry to the inspection frame |
| `measure` | Estimate heuristic wound labels; fit a periwound reference; integrate descriptors and test definition sensitivity |
| `render` | Produce photographic-colour, neutral-geometry, depth and camera views |

Key decisions:

- **Coordinates and colour:** map images to the native sensor frame; use half-resolution
  features/stereo and report reprojection errors in native pixels. Average colours
  in linear light.
- **Observation support:** retain missing regions rather than fabricate surfaces.
  Release fails when an open edge touches the saved heuristic wound mask. Pair
  estimates share images/cameras and are correlated.
- **Geometry checks:** require finite/unit normals and at least 95% agreement with
  triangle winding by both face count and area. XYZ uses float64, normals float32
  and RGB uint8.
- **Traceability:** bind caches to inputs, settings, code and libraries. Partial reruns
  require unchanged upstream dependencies and label reused stages in the manifest.

## Results

Unless stated otherwise, numerical results in this section describe the **submitted
archive (`175d788`)**, matching the accompanying report and models.

| Evidence | Recorded result |
|---|---|
| Calibration | 22/22 boards; fit RMS 0.528 px; mean held-out RMS 0.608 px |
| Registered wound photographs | 23/23 |
| Sparse landmarks | 12,403 raw; 3,509 exported with observations in at least three views |
| Accepted depth maps / dense points | 21 / 2,647,657 |
| Mesh vertices / triangles | 158,786 / 298,961 |
| Components / non-manifold edges | 1 / 0 |
| Open edges touching saved wound mask | 0; outer boundaries remain open |
| Normal/winding agreement, faces / area | 99.61% / 99.70% |
| Sparse-to-mesh median distance | 0.26% of wound length within the mesh footprint |

Sparse-to-mesh agreement is internal consistency under shared camera/image
assumptions; it is not independent surface accuracy. Zero wound-mask open edges is
a coverage check under a heuristic mask, not a clinical boundary validation.

### Calibration and uncertainty

Intrinsics use the native **4032 × 3024** landscape sensor frame:

```text
K = [[3081.934,    0.000, 1982.549],
     [   0.000, 3074.910, 1510.107],
     [   0.000,    0.000,    1.000]]

distortion [k1, k2, p1, p2, k3] = [0.236163, -1.315694, 0, 0, 2.369599]
```

The selected model uses three radial terms with tangential terms fixed at zero.
Model selection chooses the simplest fit within 2% of the lowest held-out error:
the five-coefficient model scored 0.599 px versus 0.608 px for the radial model.
Shared target geometry is refined by release-object calibration; improved prediction
does not establish the physical cause of the improvement.

Camera/target fits exclude each held-out view. Pose uses even-index corners and
scoring uses odd-index corners; the score averages per-view RMS values. Rational
fits are controls; reconstruction rejects unsupported distortion vectors rather
than silently dropping coefficients.

| Bootstrap SD | fx | fy | cx | cy | k1 | k2 | k3 |
|---|---|---|---|---|---|---|---|
| Value | 15.7 px | 19.0 px | 17.3 px | 3.3 px | 0.013 | 0.120 | 0.327 |

Boards occupy **45.3%** of a diagnostic sensor grid. Conditional bootstrap
pixel-displacement SD is 0.145 px centrally and 76.9 px at a sensor corner: precision
under the model, not measured physical error. Feature extraction limits extrapolation
to 1.25 times the observed corner radius.

Fixed checkerboard intrinsics were retained for reconstruction: the paired held-out
difference interval included zero. Camera controls are rebuilt on training images
and compared on common held-out associations with disjoint pose/scoring partitions.

### Experimental wound descriptors

For nominal wound length **L = 3.718 reconstruction units**:

| Descriptor | Nominal | Definition sensitivity |
|---|---|---|
| Width | 0.517 L | 0.508–0.518 L |
| Projected area | 0.384 L² | 0.323–0.394 L² |
| Depth P99 | 0.0327 L | 0.0291–0.0473 L |
| Volume below fitted reference | 0.0070 L³ | 0.0047–0.0111 L³ |

Each variant uses its own wound length. These one-at-a-time ranges are not confidence
intervals. Depth is measured along canonical z; P99 is the projected-area-weighted
percentile of clipped triangle mean depths. The intact reference surface is inferred,
and the boundary is unreviewed. These are not validated clinical measurements.

## Inspect outputs

Paths below are relative to the selected output directory. Original photographs,
full reconstruction files and reports are delivered separately. Only the three
curated gallery previews are included here.

| Artifact | Purpose |
|---|---|
| `scale/mesh_aligned.ply` | Vertex-coloured mesh; x along the forearm, z towards the cameras |
| `scale/dense_aligned_thinned.ply` | Dense inspection cloud in the same frame |
| `scale/sparse_aligned.ply` | Exported sparse landmarks in the same frame |
| `scale/cameras_aligned.ply` | Registered camera frusta in the same frame |
| `measure/wound_depth.ply` | Same mesh with experimental depth colours |
| `dense/dense.ply` | Full dense cloud in the original SfM frame |
| `renders/*.png` | Colour, neutral geometry, depth and camera views |
| `calibration/calibration.json` | Intrinsics, distortion, fit errors and bootstrap uncertainty |
| `calibration/bootstrap_samples.npz` | Recorded parameter draws and spatial displacements |
| `measure/measurements.json` | Descriptor definitions, sensitivity and status |
| `verification.json`, `run_manifest.json` | Release checks and run lineage |

Import the aligned PLYs together in MeshLab or CloudCompare. To inspect relief,
disable vertex colours on `mesh_aligned.ply` and use directional lighting. The full
dense cloud requires the transform in `scale/scale.json` before overlaying; read
units from that JSON and the PLY headers.

Start with `01_textured_top.png`, `02_geometry_oblique.png` and
`05_textured_oblique.png` in `renders/`. Photographic texture alone can conceal
geometry defects.

## Physical scale

Scaling the scene and camera positions together preserves monocular projections.
The brief describes a known-size reference, but its identity/dimension was not
established. The checkerboard is absent from the wound views and its square size
was unspecified. Visual inspection and searches over eight ArUco/AprilTag dictionaries
found no verified scene reference; a negative coded-marker search cannot rule out
an uncoded reference. Metric scale therefore remains an unmet requirement.

Set one source in `configs/default.yaml`, then run the full pipeline into a new
output directory:

| Setting | Required reference |
|---|---|
| `scale.marker_side_mm` | Measured ArUco/AprilTag marker visible in the photographs |
| `scale.reference_annotation` | Measured rigid segment with endpoints marked in at least two photographs |
| `scale.known_length_mm` + `scale.reference_points` | Measured separation of two points in `dense/dense.ply` |

Annotation example, using original displayed-image pixels; values are illustrative:

```json
{
  "description": "Measured rigid segment",
  "length_mm": 20.0,
  "observations": [
    {"image": "IMG_0001.png", "endpoint_a": [100, 200], "endpoint_b": [300, 200]},
    {"image": "IMG_0002.png", "endpoint_a": [120, 210], "endpoint_b": [310, 205]}
  ]
}
```

Annotation paths are relative to the data root. References must be finite, positive
and non-degenerate. Image-based references also require parallax, positive camera
depth and acceptable reprojection error. Validate the scale against a separate
measured length excluded from fitting and report reference/endpoint uncertainty.
A 1% scale error contributes about 2% area and 3% volume error, before shape errors.

## Revisions and reproducibility

The original implementation is `175d788`; the fixes are in `c74e455`. To inspect the
original separately, run the following from the repository root, then follow Quick
start in that checkout:

```sh
git worktree add --detach ../araias-wound-reference 175d788
```

Verify archived outputs with their matching source revision. A current checkout
cannot verify an older source identity. Source-only documentation/metadata changes
do not change the package-code fingerprint recorded by the pipeline.

### Correctness and integrity fixes

| Fix | Cause and resulting behaviour |
|---|---|
| Camera rollback | Focal refinement used a camera already changed before acceptance. Validate copied camera/pose/point state; rejection or validation exceptions leave retained state unchanged. Report candidate and retained RMS separately and reject negative solver statuses. |
| Render hashes | Rendering wrote `renders/`; collection searched `render/`. Share the folder mapping and hash all five PNGs. A real-renderer regression alters a copied PNG and requires verification to fail. |
| Stereo reserves | Six candidates were truncated before feasibility checks. Try reserves until six usable partners or exhaustion; retain existing disparity, agreement and TSDF support thresholds. |

No rejected adjustment was recorded in the submitted archive, so a rollback effect
on its geometry was not demonstrated. Missing screenshot hashes weakened integrity
checks without changing coordinates. Reserve matching changes coverage and can
change the derived descriptors.

### Recorded runs

All four runs used the same input hashes, macOS arm64 and Python 3.13.5. Each completed
all eight stages and has saved passing verification. L is each run's nominal wound
length. These outcomes are not reproduction tolerances or physical-accuracy bounds.

| Evidence | Submitted archive | Corrected archive | Submitted repeat | Corrected repeat |
|---|---:|---:|---:|---:|
| Source revision | `175d788` | `c74e455` | `175d788` | `c74e455` |
| Date (2026) | 28 Oct | 28 Oct | 4 Nov | 4 Nov |
| Accepted depth maps | 21 | 22 | 20 | 22 |
| Dense points | 2,647,657 | 2,894,366 | 2,578,291 | 2,876,364 |
| Mesh vertices / triangles | 158,786 / 298,961 | 155,647 / 292,891 | 152,964 / 288,545 | 150,887 / 285,063 |
| Relative depth P99 / L | 0.03273 | 0.03008 | 0.03112 | 0.02996 |
| Relative volume / L³ | 0.006977 | 0.005936 | 0.006260 | 0.005905 |
| Hashed render PNGs | 0 | 5 | 0 | 5 |

All registered 23/23 photographs with zero open edges touching the saved wound mask;
physical scale remained unresolved and measurement status remained uncertain.
Between archived runs, P99 changed -8.1% and volume -14.9%. Full reruns are not a
controlled ablation: numerical optimisation and threshold decisions also affect
poses and downstream support. Lower descriptors do not establish better accuracy.

## Development

Run the synthetic suite and static checks without task images:

```sh
.venv/bin/python -m pytest -q
.venv/bin/ruff check src tests scripts
.venv/bin/ruff format --check src tests scripts
.venv/bin/python -m pip check
```

The current suite has **73 tests** covering calibration, projection, stereo,
bundle-adjustment failure paths, TSDF support/normals, integration, scale, PLY
precision, cache invalidation, release rejection, rendering/tamper detection and
packaging. Tests establish specified software behaviour, not physical accuracy.

The [source-quality workflow](https://github.com/goapu/araias-wound-reconstruction/actions/workflows/quality.yml)
runs on pushes and pull requests, or manually. It checks dependencies, lint/format,
synthetic tests, wheel construction, packaged defaults and all three installed CLI
entry points. Its result applies to the corresponding pushed commit.

## Packaging results

Package a verified run with an existing PDF describing that same run:

```sh
.venv/bin/wound-export --output-dir outputs \
  --report /path/to/report.pdf --dest submission
```

The exporter validates the run, PDF and inspection images, then creates the report
copy, source/model ZIPs, screenshots, metrics and SHA-256 inventory. The source
archive uses an allowlist to exclude task data, derived assets and credentials;
inspection images are included separately in the delivery. The gallery above is
available in the repository.
Source-checkout wrappers for all three commands are in `scripts/`.

## Repository layout

```text
.github/workflows/quality.yml  automated source and package checks
assets/previews/              three submitted-run inspection renders
configs/default.yaml          pipeline settings
src/woundrecon/                numerical modules, CLI and stage orchestration
  stages/                     audit, calibrate, sfm, dense, mesh, scale, measure, render
  resources/default.yaml      installed-package defaults
tests/                        synthetic unit, regression and integration checks
scripts/                      source-checkout command wrappers
pyproject.toml                package metadata, dependencies and lint settings
requirements.lock.txt         pinned development environment
.editorconfig, .gitattributes text formatting and line-ending rules
```

Each stage module orchestrates work; numerical modules implement calibration,
geometry and measurement. `provenance.py` and `reuse.py` bind dependencies;
`validation.py` checks saved outputs; `delivery.py` assembles verified exports.

## Limitations and next steps

The main gaps are verified scale and independent surface truth. Peripheral
calibration uncertainty, unlocked focus/exposure, specular regions, possible motion
and unreviewed boundary/reference definitions also limit measurements. Point count
and visual smoothness are not accuracy evidence.

Priorities: identify/measure an in-scene reference; improve rigid-target calibration
with edge/tilt coverage at the intended focus distance; obtain reference phantom
geometry and reviewed boundaries. Then quantify surface error, completeness and
repeatability across sessions/operators using common regions and matched resolution.
Agree acceptance tolerances before comparing methods and request recapture when
observation support is inadequate.

## Methods and tools

[Zhang calibration](https://doi.org/10.1109/34.888718) ·
[Imperfect-target calibration](https://doi.org/10.1109/ICCVW.2011.6130369) ·
[SIFT](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf) ·
[RootSIFT](https://www.robots.ox.ac.uk/~vgg/publications/2012/Arandjelovic12/arandjelovic12.pdf) ·
[Bundle adjustment](https://doi.org/10.1007/3-540-44480-7_21) ·
[Semi-global matching](https://doi.org/10.1109/TPAMI.2007.1166) ·
[TSDF fusion](https://graphics.stanford.edu/papers/volrange/)

Built with OpenCV, NumPy, SciPy, scikit-image, Pillow, Matplotlib, PyYAML and pypdf.
