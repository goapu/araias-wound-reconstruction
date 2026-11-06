# Camera calibration and 3D wound reconstruction

**Dilip Goswami** · [![Source quality](https://github.com/goapu/araias-wound-reconstruction/actions/workflows/quality.yml/badge.svg?branch=main)](https://github.com/goapu/araias-wound-reconstruction/actions/workflows/quality.yml)

A CPU pipeline for iPhone 13 Pro Max camera calibration and silicone wound reconstruction from RGB photographs.
It exports calibrated cameras, sparse/dense point clouds, a vertex-coloured mesh and PLY files. Experimental
wound descriptors include definition sensitivity; the gallery and metrics describe the same complete execution.

[Gallery](#reconstruction-gallery) · [Quick start](#quick-start) · [Method](#method) ·
[Results](#results) · [Outputs](#inspect-outputs) · [Scale](#physical-scale) · [Reproducibility](#reproducibility-and-verification)

## Reconstruction gallery

Colour, geometry and depth views of the **150,887-vertex mesh** of a silicone wound phantom.

<img src="assets/previews/05_textured_oblique.png" width="780" alt="Vertex-coloured reconstructed wound mesh, oblique view">

| Surface geometry | Experimental depth |
|---|---|
| ![Neutral oblique rendering](assets/previews/02_geometry_oblique.png) | ![Depth relative to the fitted reference](assets/previews/03_wound_depth.png) |
| Neutral shading shows relief and open boundaries. | Purple: zero; yellow: P99, about 3.0% of wound length. |

## Quick start

Use **Python 3.13** (recorded: 3.13.5, macOS arm64, four OpenCV threads). Package metadata declares Python
3.12+. No GPU or training is required. Keep data outside Git and run commands from the repository root.

```text
workspace/
├── checkerboard/                  # 22 calibration photographs
├── images/                        # 23 wound photographs
└── araias-wound-reconstruction/    # source repository
```

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
.venv/bin/python -m pip install --no-deps -e '.[dev]'
.venv/bin/wound-reconstruct --data-root .. --output-dir outputs --threads 4
.venv/bin/wound-verify --output-dir outputs --data-root ..
```

Use a fresh output directory and one environment per checkout. Settings in `configs/default.yaml` match packaged
defaults in `src/woundrecon/resources/default.yaml`; use `opencv-python-headless` only.

## Method

| Stage | Approach |
|---|---|
| `audit` | Resolution, orientation, colour profile, exposure, sharpness and highlights |
| `calibrate` | Checkerboard corners, five lens models, held-out scoring and bootstrap uncertainty |
| `sfm` | RootSIFT, geometric match verification, tracks, incremental poses and robust bundle adjustment |
| `dense` | Rectified SGBM; ranked reserves until six usable partners or exhaustion; depth agreement across two pairs |
| `mesh` | TSDF and marching cubes; all eight cell corners require support from at least two depth maps |
| `scale` | Validate a physical reference and align the inspection frame |
| `measure` | Heuristic boundary, cross-validated periwound reference and descriptor sensitivity |
| `render` | Colour, neutral geometry, depth and camera views |

Features/stereo use half resolution; reprojection errors use native pixels. Colours are averaged in linear
light. Missing support stays open; release requires finite/unit normals, at least 95% winding agreement by face
count/area and no open edges touching the saved wound mask. XYZ is float64, normals float32 and RGB uint8.

## Results

All eight stages completed and saved-output verification passed.

| Evidence | Recorded result |
|---|---|
| Calibration | 22/22 boards; fit RMS 0.528 px; mean held-out RMS 0.608 px |
| Registered photographs | 23/23 |
| Accepted reference depth maps | 22/23 |
| Sparse landmarks | 12,353 raw; 3,481 exported with observations in at least three views |
| Dense cloud / thinned inspection cloud | 2,876,364 / 553,189 points |
| Mesh vertices / triangles | 150,887 / 285,063 |
| Components / non-manifold edges | 1 / 0 |
| Open edges | 20,621 total; 0 touching the saved wound mask |
| Normal/winding agreement, faces / area | 99.45% / 99.57% |
| Sparse-to-mesh distance within footprint | Median 0.257% of wound length; P95 1.158% |
| Hashed inspection renders | 5 PNGs |

Dense counts are after voxel aggregation; `dense_stats.json` records 10,293,539 samples before aggregation,
including overlapping surface observations.

**Scope.** Geometry uses reconstruction units; metric scale requires a verified reference. Sparse-to-mesh
agreement and mask coverage are internal checks. Boundary and reference lack independent validation;
descriptors remain uncertain. Software verification does not establish clinical accuracy.

### Calibration (4032 × 3024 native pixels)

```text
K = [[3081.934,    0.000, 1982.549],
     [   0.000, 3074.910, 1510.107],
     [   0.000,    0.000,    1.000]]
distortion [k1,k2,p1,p2,k3] = [0.236163,-1.315694,0,0,2.369599]
```

Three radial terms are selected within 2% of the best held-out RMS (0.608 versus 0.599 px); shared target
geometry is refined. Held-out views are excluded from fitting; even corners estimate pose, odd corners score it.
Fixed intrinsics are retained because the paired wound-camera improvement interval includes zero.

| Bootstrap SD | fx | fy | cx | cy | k1 | k2 | k3 |
|---|---|---|---|---|---|---|---|
| Value | 15.7 px | 19.0 px | 17.3 px | 3.3 px | 0.013 | 0.120 | 0.327 |

Diagnostic grid coverage is 45.3%. Conditional bootstrap pixel-displacement SD is 0.145 px centrally and 76.9 px
at a corner; feature extraction is limited to 1.25 times the observed checkerboard corner radius.

### Experimental wound descriptors

For nominal wound length **L = 3.730 reconstruction units**:

| Descriptor | Nominal | Definition sensitivity |
|---|---|---|
| Width | 0.517 L | 0.497–0.517 L |
| Projected area | 0.373 L² | 0.336–0.394 L² |
| Surface area | 0.394987 L² | 0.354472–0.417815 L² |
| Depth P99 | 0.02996 L | 0.02868–0.05012 L |
| Signed mean depth | 0.015723 L | 0.013984–0.030097 L |
| Volume below reference | 0.005905 L³ | 0.004734–0.011916 L³ |

Reference order **4** was selected by leave-one-sector-out cross-validation over eight sectors. The planar
reference was rejected because it placed the wound above the reference. Depth follows canonical z; P99 weights
clipped triangle means by projected area, and signed mean depth counts above-reference regions negatively. Each
variant uses its own L; **sensitivity ranges are not confidence intervals**.

## Inspect outputs

Artifact paths are relative to the output directory.

| Artifact | Contents |
|---|---|
| `scale/mesh_aligned.ply` | Coloured mesh; x along forearm, z towards cameras |
| `scale/dense_aligned_thinned.ply` | Thinned dense inspection cloud |
| `scale/sparse_aligned.ply` | Exported sparse landmarks |
| `scale/cameras_aligned.ply` | Registered camera frusta |
| `measure/wound_depth.ply` | Same mesh with experimental depth colours |
| `dense/dense.ply`, `dense/dense_stats.json` | Final dense cloud and pre-/post-aggregation counts |
| `calibration/calibration.json`, `calibration/bootstrap_samples.npz` | Intrinsics, errors, parameter draws and displacements |
| `measure/measurements.json` | Descriptors, sensitivity and status |
| `verification.json`, `run_manifest.json` | Release checks, settings and hashes |

Aligned PLYs overlay directly in MeshLab/CloudCompare; apply `scale/scale.json` to the full dense cloud before
overlaying it. Inspect with vertex colours disabled and directional lighting. Preview files are
`05_textured_oblique.png`, `02_geometry_oblique.png` and `03_wound_depth.png`, also generated under `renders/`.

## Physical scale

Monocular scale requires a verified length. The board is absent from wound views and its square size is
unspecified. Searching eight coded-marker dictionaries found no marker but cannot exclude an uncoded reference.
Set one scale source and rerun the pipeline into a fresh output directory:

| Configuration | Required reference |
|---|---|
| `scale.marker_side_mm` | Measured coded marker visible in the photographs |
| `scale.reference_annotation` | Measured rigid segment marked in at least two photographs |
| `scale.known_length_mm` + `scale.reference_points` | Measured separation of two points in `dense/dense.ply` |

Example annotation uses original displayed-image pixels; values are illustrative:

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

Paths are relative to the data root. Require positive finite lengths and separated endpoints; image annotations
also need parallax, positive depth and low reprojection error. Verify on a separate measured length. Scale error
propagates approximately 1:2:3 into length, area and volume error.

## Reproducibility and verification

The manifest binds 30 package sources, 45 input images and 103 output artifacts to their settings/environment.
Caches and partial reruns check upstream dependencies.

| Safeguard | Rule |
|---|---|
| Bundle adjustment | Validate copied candidate state; rejection leaves retained parameters unchanged |
| Render integrity | Hash all five PNGs; a modified screenshot fails verification |
| Stereo support | Use ranked reserves while retaining disparity, agreement and TSDF support limits |

### Repeating the experiment

Two executions used matching code, input hashes, settings, libraries and threads. Fixed seeds still permit
native numerical variation; these observed ranges do not characterise capture repeatability.

| Metric | Observed minimum–maximum |
|---|---|
| Dense points | 2,876,364–2,894,366 |
| Mesh vertices | 150,887–155,647 |
| Depth P99 / L | 0.02996–0.03008 |
| Volume / L³ | 0.005905–0.005936 |

## Development

```sh
.venv/bin/python -m pytest -q
.venv/bin/ruff check src tests scripts
.venv/bin/ruff format --check src tests scripts
.venv/bin/python -m pip check
```

The **73 synthetic tests** cover geometry, calibration, stereo, failure paths, provenance, rendering and
packaging. CI adds wheel/defaults/installed-CLI checks on pushes, pull requests and manual dispatch.

## Export

```sh
.venv/bin/wound-export --output-dir outputs --report /path/to/report.pdf --dest submission
```

Exports a verified run with the matching PDF, source/model ZIPs, screenshots, metrics and SHA-256 inventory.

## Repository layout

```text
.github/workflows/quality.yml  source and package checks
assets/previews/              three reconstruction renders
configs/default.yaml          pipeline configuration
src/woundrecon/                numerical modules, CLI, stages/ and packaged resources/
tests/                        synthetic tests
scripts/                      command wrappers
pyproject.toml, requirements.lock.txt  packaging, lint settings and pinned dependencies
.editorconfig, .gitattributes formatting and line endings
```

Stages orchestrate numerical modules; provenance tracks dependencies, validation checks files and delivery
exports.

## Limitations and next steps

| Limitation | Consequence and next step |
|---|---|
| Unknown physical scale | Supply a measured rigid reference and verify on a separate length |
| Peripheral calibration uncertainty | Recapture with a rigid target, varied tilt/distance and edge/corner coverage |
| `IMG_2976.png` has no depth map | All 18 candidates exceeded the 384-pixel disparity budget; inspect pose/orientation and acquisition geometry |
| Open outer boundaries | Improve viewpoint coverage and measure completeness against reference geometry |
| Heuristic boundary and fitted intact surface | Obtain expert labels and known phantom geometry before relying on depth/volume |
| One phantom; unlocked focus/exposure, glare and possible motion | Test repeated captures across sessions, operators and devices with agreed acceptance criteria |

## References

[Zhang calibration](https://doi.org/10.1109/34.888718) ·
[Imperfect-target calibration](https://doi.org/10.1109/ICCVW.2011.6130369) ·
[SIFT](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf) · [RootSIFT](https://www.robots.ox.ac.uk/~vgg/publications/2012/Arandjelovic12/arandjelovic12.pdf) ·
[Bundle adjustment](https://doi.org/10.1007/3-540-44480-7_21) · [SGBM](https://doi.org/10.1109/TPAMI.2007.1166) · [TSDF](https://graphics.stanford.edu/papers/volrange/)
