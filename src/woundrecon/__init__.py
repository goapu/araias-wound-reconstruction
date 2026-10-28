"""woundrecon: camera calibration and multi-view 3D reconstruction of a wound moulage.

The package is organised as one module per pipeline stage:

    io_utils            image loading, orientation handling, colour management, PLY I/O
    config              YAML configuration loading
    calibration         checkerboard calibration, model selection and uncertainty
    features            SIFT extraction, matching and two-view geometric verification
    sfm                 incremental structure-from-motion
    bundle_adjustment   sparse non-linear least squares refinement
    dense               pairwise stereo depth maps and multi-view consistency fusion
    mesh                TSDF fusion and surface extraction
    scale               metric scale estimation
    measure             wound segmentation and surface measurements
    render              software point-cloud / mesh rendering for screenshots
"""

__version__ = "0.1.0"
