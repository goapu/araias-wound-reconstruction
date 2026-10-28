# Camera calibration and 3D wound reconstruction

**Dilip Goswami · ARAIAS technical take-home**

A CPU-only photogrammetry pipeline that calibrates an iPhone 13 Pro Max main camera
from checkerboard photos and reconstructs a silicone wound moulage on a forearm as
sparse and dense point clouds and a mesh with vertex colours. Experimental wound
descriptors include their sensitivity to the boundary and reference definition.

| Result | Value |
|---|---|
| Calibration | 22 / 22 boards detected; fit RMS 0.53 px, held-out RMS 0.61 px |
| Registration | 23 / 23 wound photos |
| Point clouds | 12,403 sparse points (3,509 seen in ≥ 3 views); 2.65 M dense points |
| Dense support | 21 / 23 reference depth maps accepted |
| Mesh | 158,786 vertices; 298,961 triangles; 1 component; 0 open edges touching the saved wound mask |
| Internal consistency | Sparse points over the mesh footprint lie a median 0.26% of wound length from the mesh |
| Software checks | 56 local tests passing; lint and formatting clean |

No scene reference with a verified physical size was established. Outputs retain
reconstruction units; see [Physical scale](#physical-scale). The wound boundary and
inferred reference surface also lack independent validation.

[Quick start](#quick-start) ·
[Results](#results) ·
[Outputs](#outputs) ·
[Physical scale](#physical-scale) ·
[How it works](#how-it-works) ·
[Tests](#tests) ·
[Packaging](#packaging-results) ·
[Repository layout](#repository-layout)

## Quick start

Use Python 3.13 for the pinned environment; the reference execution used Python
3.13.5 on macOS arm64 with four OpenCV threads. The package declares Python 3.12+
support. No GPU, training, pretrained weights or remote image service is required.

Place the task data next to the repository (it is not included here):

```text
workspace/
├── checkerboard/                    # 22 calibration photos
├── images/                          # 23 wound photos
└── araias-wound-reconstruction/     # this repository
```

Install and run from the repository folder:

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
.venv/bin/python -m pip install --no-deps -e '.[dev]'

.venv/bin/wound-reconstruct --data-root .. --output-dir outputs --threads 4
.venv/bin/wound-verify      --output-dir outputs --data-root ..
```

The reference run recorded 14.1 minutes of stage computation, excluding startup
and verification overhead. Runtime depends on hardware and settings. Logs and
`outputs/run_manifest.json` record stage outcomes/times, source/input/artifact hashes,
settings and library versions. Numerical reproduction ends with `wound-verify`;
PDF typesetting is separate. Use editable installation and one OpenCV distribution
(`opencv-python-headless`). Source wrappers are in `scripts/`.

## Results

### Camera calibration

Selected model: three radial terms (k1, k2, k3), with tangential terms fixed at zero
and shared target geometry refined by release-object calibration. It was selected
on held-out views from five lens models. This relaxes the planar-target assumption;
it does not establish the physical cause of the improvement. Intrinsics use the
native 4032 × 3024 landscape sensor frame.

Reconstruction supports the five-coefficient OpenCV camera model. Rational fits
remain calibration controls; transferring a longer distortion vector is explicitly
rejected rather than discarding coefficients.

Camera/target fits exclude each held-out view. Pose uses even-index corners and
scoring uses odd-index corners; the held-out RMS averages the per-view RMS values.

```text
K = [[3081.934,    0.000, 1982.549],
     [   0.000, 3074.910, 1510.107],
     [   0.000,    0.000,    1.000]]

distortion [k1, k2, p1, p2, k3] = [0.236163, -1.315694, 0, 0, 2.369599]
```

| Parameter | fx | fy | cx | cy | k1 | k2 | k3 |
|---|---|---|---|---|---|---|---|
| Bootstrap SD | 15.7 px | 19.0 px | 17.3 px | 3.3 px | 0.013 | 0.120 | 0.327 |

The boards occupy 45.3% of a sensor grid. Conditional bootstrap displacement SD is
0.145 px centrally and 76.9 px at a sensor corner; these are precision estimates,
not physical error bounds. Reconstruction limits features to 1.25 times the observed
corner radius to restrict extrapolation. Fixed checkerboard intrinsics were retained:
the paired held-out difference interval included zero, so refining intrinsics did
not establish better predictive performance. Wound-camera controls are rebuilt
on training images and scored on common held-out associations with disjoint
pose/test partitions.

### Wound measurements

Relative to nominal wound length L = 3.718 reconstruction units:

| Descriptor | Nominal | Range across definition variants |
|---|---|---|
| Width | 0.517 L | 0.508–0.518 L |
| Projected area | 0.384 L² | 0.323–0.394 L² |
| Depth (99th percentile) | 0.0327 L | 0.0291–0.0473 L |
| Volume below the fitted reference | 0.0070 L³ | 0.0047–0.0111 L³ |

Each variant is normalized by its own wound length. Width varies by about 1.8%;
projected area changes by -15.8% to +2.8% from nominal. Depth and volume are more
sensitive to the counterfactual reference surface. The ranges are one-at-a-time
definition sensitivity, not confidence intervals. Depth is measured along the
canonical z axis; P99 is the projected-area-weighted percentile of clipped triangle
mean depths. These descriptors are not validated clinical measurements.

## Outputs

Paths are relative to `outputs/`.

| File | Contents |
|---|---|
| `scale/mesh_aligned.ply` | Vertex-coloured mesh in a canonical inspection frame (x along the forearm, z towards the cameras) |
| `scale/dense_aligned_thinned.ply` | Dense cloud thinned to mesh resolution, same frame |
| `scale/sparse_aligned.ply` | Sparse landmarks seen in at least three views, same frame |
| `scale/cameras_aligned.ply` | Frusta of the 23 registered cameras, same frame |
| `measure/wound_depth.ply` | Mesh coloured by experimental depth below the fitted reference |
| `dense/dense.ply` | Full dense cloud in the original reconstruction frame |
| `renders/*.png` | Photographic-colour, geometry-only, depth and camera views |
| `calibration/calibration.json` | K, distortion, errors and bootstrap uncertainty |
| `calibration/bootstrap_samples.npz` | Recorded parameter draws and spatial displacements behind the uncertainty figure |
| `measure/measurements.json` | Wound descriptors, sensitivity variants and status |
| `verification.json`, `run_manifest.json` | Release checks; hashes, settings and stage times |

The aligned files overlay directly in MeshLab or CloudCompare. The full dense cloud
uses the original SfM frame; apply the transform in `scale/scale.json` before overlaying
it. Read units from PLY headers and that JSON. To inspect relief, open
`scale/mesh_aligned.ply`, disable vertex colours and use directional lighting.
Outer boundaries remain open. Zero open edges touching the saved heuristic mask
establishes support under that mask, not a clinical boundary. Three standard inspection
views are `01_textured_top.png`, `02_geometry_oblique.png` and
`05_textured_oblique.png` under `renders/`.

## Physical scale

Calibrated monocular geometry has an arbitrary similarity transform. Scaling the
scene and camera positions together preserves image projections. A verified length
sets metric scale; it does not establish shape accuracy. The checkerboard square
size was unspecified, and no verified dimension was established in the wound scene.
The eight-dictionary ArUco/AprilTag search found no detections; it does not rule out
uncoded references.

To make the outputs metric, set one source in `configs/default.yaml` and run the full
pipeline again:

| Setting | What it needs |
|---|---|
| `scale.marker_side_mm` | An ArUco/AprilTag marker of measured size visible in the photos |
| `scale.reference_annotation` | A JSON file with a measured segment marked in at least two photos (format below) |
| `scale.known_length_mm` + `scale.reference_points` | A measured distance between two points of `dense/dense.ply` |

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

These values illustrate the schema, not a phantom measurement. Pixels refer to the
original displayed photographs; annotation paths are relative to the data root.
Lengths/factors must be finite and positive, with separated endpoints. Image-based
references additionally require parallax, positive camera depth and acceptable
reprojection error. Invalid sources are rejected with a reason. A 1% scale-factor
error propagates to about 2% in area and 3% in volume, before other errors.

## How it works

| Stage | What it does |
|---|---|
| `audit` | Checks resolution, orientation, colour profile, exposure, sharpness and highlights |
| `calibrate` | Detects corners, compares five lens models on held-out views, bootstraps uncertainty |
| `sfm` | RootSIFT matching, verified tracks, incremental registration, robust bundle adjustment; tests fixed vs. refined intrinsics on held-out photos |
| `dense` | Rectified SGBM stereo with six partners per view; keeps depths supported by two distinct pairs, which share a reference image and are correlated |
| `mesh` | TSDF fusion and marching cubes; keeps a face only where all eight voxel corners were observed by at least two depth maps; colours from the photos in linear light |
| `scale` | Searches for a scale reference, applies it if valid, aligns to the anatomical frame |
| `measure` | Estimates heuristic wound labels, selects a periwound reference by spatial cross-validation, integrates descriptors and tests alternative definitions |
| `render` | Renders the inspection views |

Settings are in `configs/default.yaml`; installed-package defaults match under
`src/woundrecon/resources/default.yaml`. Changes to configuration require a full run
into a separate output directory.

Design choices:

- **Coordinates.** Images are mapped to the native sensor frame. Features and stereo
  run at half resolution; reported reprojection errors are in native pixels.
- **Observation support.** Unsupported regions stay open; release fails if an open
  edge touches the saved heuristic wound mask. This is a coverage check.
- **Normals and storage.** Normals must be finite/unit and agree with triangle winding
  on at least 95% of faces and area (observed: 99.61% / 99.70%). PLY stores XYZ as
  float64, normals as float32 and RGB as uint8; narrow triangles survive translation.
- **Traceable reruns.** Caches are keyed by input bytes, settings, code and library
  versions. A partial rerun (`--stages render`, or any suffix from `dense` onwards) is
  accepted only if upstream code, inputs, settings and environment are unchanged, and
  reused stages are labelled in the manifest. Randomized steps are seeded; native
  execution and threshold decisions can still vary across runs.

## Tests

```sh
.venv/bin/python -m pytest -q
.venv/bin/ruff check src tests scripts
.venv/bin/ruff format --check src tests scripts
```

The 56 local tests use synthetic data only. They cover calibration/model subsets,
camera projection, horizontal/vertical stereo, bundle-adjustment guards, TSDF support
and normals, clipped integration, scale, PLY precision, cache invalidation, incomplete
run rejection and PDF/source packaging. CI uses the same commands without task images.
Hosted status applies to the corresponding pushed commit.

## Packaging results

Package a verified run with an existing PDF:

```sh
.venv/bin/wound-export --output-dir outputs \
  --report /path/to/report.pdf --dest submission
```

The exporter validates the run, PDF and screenshots, then creates the PDF copy,
source/model ZIPs, inspection images, metrics and SHA-256 inventory in `submission/`.
The source-checkout equivalent is `scripts/export_submission.py`.

## Repository layout

```text
configs/default.yaml         all pipeline settings
src/woundrecon/
  calibration.py             corner detection, lens models, held-out scoring, bootstrap
  io_utils.py, config.py     image/PLY I/O, colour/orientation and YAML settings
  camera.py                  pinhole and distortion model, triangulation
  features.py, sfm.py        RootSIFT matching, tracks, incremental reconstruction
  bundle_adjustment.py       sparse robust least squares with acceptance checks
  heldout.py                 held-out comparison of camera models
  dense.py                   rectification, SGBM stereo, depth fusion
  mesh.py, geometry.py       TSDF surface, colour transfer, exact triangle geometry
  scale.py, measure.py       scale references, anatomical frame, wound descriptors
  render.py, plots.py        software renderer and figures
  provenance.py, reuse.py    run manifest, cache keys, checked partial reruns
  validation.py              release checks on saved outputs
  delivery.py                packaging of a verified run
  cli.py, stages/            command line and one module per pipeline stage
tests/                       synthetic unit and regression tests
scripts/                     source-checkout wrappers for the installed commands
```

## Limitations

No independent surface truth was supplied. Remaining limitations are unresolved
scale, peripheral calibration uncertainty, unlocked focus/exposure, specular regions,
possible motion and unreviewed boundary/reference definitions. Point count and
smoothness do not establish physical accuracy.

Priorities are a measured in-scene reference, rigid-target calibration with edge/tilt
coverage at the intended focus distance, independent phantom geometry and reviewed
boundaries. Then assess surface error/completeness across sessions and operators,
using common regions and matched resolution when comparing methods. Define acceptance
tolerances before benchmarking; request recapture when support is inadequate.

## Methods and tools

[Zhang camera calibration](https://doi.org/10.1109/34.888718) ·
[calibration with an imperfect target](https://doi.org/10.1109/ICCVW.2011.6130369) ·
[SIFT](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf) ·
[RootSIFT](https://www.robots.ox.ac.uk/~vgg/publications/2012/Arandjelovic12/arandjelovic12.pdf) ·
[bundle adjustment](https://doi.org/10.1007/3-540-44480-7_21) ·
[semi-global matching](https://doi.org/10.1109/TPAMI.2007.1166) ·
[TSDF fusion](https://graphics.stanford.edu/papers/volrange/)

Built with OpenCV, NumPy, SciPy, scikit-image, Pillow, Matplotlib, PyYAML and pypdf.
