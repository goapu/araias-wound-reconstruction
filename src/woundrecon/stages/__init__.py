"""Pipeline stages. Each stage module exposes ``run(cfg) -> dict`` and reads the
outputs of earlier stages from ``cfg['paths']['output']``."""

from importlib import import_module

STAGE_ORDER = ["audit", "calibrate", "sfm", "dense", "mesh", "scale", "measure", "render"]
# One layout is shared by stage writers and artifact hashing. Public stage
# names remain stable even when the on-disk folder uses a descriptive noun.
STAGE_OUTPUT_FOLDERS = {
    name: {"calibrate": "calibration", "render": "renders"}.get(name, name) for name in STAGE_ORDER
}


def get_stage(name: str):
    """Import a stage module lazily so that a failing stage does not block the others."""
    if name not in STAGE_ORDER:
        raise ValueError(f"Unknown stage '{name}'. Available: {', '.join(STAGE_ORDER)}")
    return import_module(f"{__name__}.{name}")
