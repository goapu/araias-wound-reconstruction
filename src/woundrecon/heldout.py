"""Held-out wound-view evaluation of the camera model (intrinsics A vs. B).

Training-set error cannot choose between the checkerboard intrinsics (A) and
the intrinsics refined on the wound images (B): B is fitted to those very
observations, so extra fit flexibility cannot establish predictive quality.
This module tests both on
images that took no part in building the 3D model:

1. Every ``every``-th image is held out; the incremental reconstruction and both
   variants are rebuilt from the remaining (training) images only.
2. For a held-out image, a keypoint is associated with a training track when it
   is matched (verified pairwise match) to that track in at least two training
   images; associations are one-to-one.
3. The associations are split at random into two disjoint halves. The camera
   pose is estimated from the first half (PnP + RANSAC) and the error is
   measured on the second half only, so no observation is used both to fit and
   to score. Both variants are scored on the *same* associations.
4. The paired difference (error A - error B) is summarised with a bootstrap
   over held-out images (associations within an image share one pose estimate
   and are not independent). Variant B is adopted only if the 95 % interval of
   the median paired difference lies entirely above zero; otherwise the model
   with fewer free parameters (A) is kept.

This addresses camera-model selection only. Agreement with held-out images is
internal consistency; it does not measure 3D accuracy.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from . import features as feat
from . import io_utils
from .camera import Camera
from .sfm import IncrementalSfM, SfMParams

log = logging.getLogger(__name__)


@dataclass
class HeldOutConfig:
    """Deterministic image hold-out, training-vote support and image-bootstrap budgets."""

    every: int = 5  # Stride: offset, offset + every, offset + 2 * every, ...
    offset: int = 1
    min_votes: int = 2  # training images that must agree on an association
    min_associations: int = 60
    bootstrap: int = 2000
    seed: int = 0


def heldout_indices(n_images: int, c: HeldOutConfig) -> list[int]:
    """Return zero-based held-out image indices from the configured offset and stride."""
    return list(range(c.offset, n_images, c.every))


def _subset_pairs(pairs: dict[tuple[int, int], feat.PairMatches], train: list[int]) -> dict:
    """Pairs among training images, re-indexed to positions in ``train``."""
    local = {g: k for k, g in enumerate(train)}
    out = {}
    for (i, j), m in pairs.items():
        if i in local and j in local:
            li, lj = local[i], local[j]
            out[(li, lj)] = feat.PairMatches(
                li, lj, m.idx_i, m.idx_j, m.n_putative, m.n_homography_inliers
            )
    return out


def reconstruct_training(
    feats,
    camera: Camera,
    rules: list[str],
    params: SfMParams,
    pairs: dict,
    train: list[int],
    free: tuple[str, ...],
) -> dict[str, Any]:
    """Incremental SfM and both intrinsics variants on the training images only."""
    # Exclude held-out images before track building and adjustment: training
    # 3D points cannot absorb the eventual test-view reprojection errors.
    sub = IncrementalSfM(
        [feats[i] for i in train], camera.copy(), [rules[i] for i in train], params
    )
    sub.pairs = _subset_pairs(pairs, train)
    sub.run()
    rec_a, cam_a = copy.deepcopy(sub.rec), sub.cam.copy()
    kp_track_a = [k.copy() for k in sub.kp_track]
    for _ in range(2):
        sub.bundle_adjust(free=free, max_nfev=200)
    sub.bundle_adjust(max_nfev=100)
    return {
        "A": (rec_a, cam_a, kp_track_a),
        "B": (sub.rec, sub.cam, sub.kp_track),
        "registered": int(sub.rec.registered.sum()),
        "train": train,
    }


def associations(
    h: int, kp_h_count: int, train_model: dict, pairs: dict, min_votes: int
) -> np.ndarray:
    """One-to-one (keypoint of held-out image, track) pairs supported by >= ``min_votes`` training images.

    Only tracks triangulated in both variants, and training observations still
    attached to their track in both variants, are used, so A and B are scored
    on identical associations.
    """
    rec_a, _, kpt_a = train_model["A"]
    rec_b, _, kpt_b = train_model["B"]
    # Require the same track to survive both fits for a strictly paired A/B score.
    ok_track = np.isfinite(rec_a.track_xyz[:, 0]) & np.isfinite(rec_b.track_xyz[:, 0])
    votes: dict[tuple[int, int], set[int]] = {}
    for lt, gt in enumerate(train_model["train"]):
        key = (min(h, gt), max(h, gt))
        if key not in pairs:
            continue
        m = pairs[key]
        idx_h, idx_t = (m.idx_i, m.idx_j) if key[0] == h else (m.idx_j, m.idx_i)
        ta, tb = kpt_a[lt][idx_t], kpt_b[lt][idx_t]
        same = (ta >= 0) & (ta == tb)
        for kh, t in zip(idx_h[same], ta[same], strict=True):
            if ok_track[t]:
                votes.setdefault((int(kh), int(t)), set()).add(gt)
    chosen, used_kp, used_track = [], set(), set()
    for (kh, t), imgs in sorted(votes.items(), key=lambda kv: -len(kv[1])):
        if len(imgs) >= min_votes and kh not in used_kp and t not in used_track:
            used_kp.add(kh)
            used_track.add(t)
            chosen.append((kh, t))
    return np.array(chosen, int).reshape(-1, 2)


def _score(
    X: np.ndarray,
    x: np.ndarray,
    pose: np.ndarray,
    test: np.ndarray,
    cam: Camera,
    threshold_px: float,
) -> np.ndarray | None:
    """Pose from ``pose`` rows, reprojection error (native pixels) of ``test`` rows."""
    feat.seed_opencv()
    ok, rvec, tvec, inl = cv2.solvePnPRansac(
        X[pose],
        x[pose],
        cam.K,
        cam.dist,
        iterationsCount=5000,
        reprojectionError=threshold_px,
        confidence=0.9999,
        flags=cv2.SOLVEPNP_SQPNP,
    )
    if not ok or inl is None or len(inl) < 20:
        return None
    # Pose inliers alone refine the nuisance pose; test rows never enter this fit.
    sel = pose[inl.ravel()]
    rvec, tvec = cv2.solvePnPRefineLM(X[sel], x[sel], cam.K, cam.dist, rvec, tvec)
    proj, _ = cv2.projectPoints(X[test], rvec, tvec, cam.K, cam.dist)
    return np.linalg.norm(proj.reshape(-1, 2) - x[test], axis=1)


def evaluate(
    feats,
    final_rules: list[str],
    params: SfMParams,
    pairs: dict,
    train_model: dict,
    heldout: list[int],
    c: HeldOutConfig,
) -> dict[str, Any]:
    """Score variants A and B on the held-out images (see module docstring)."""
    rng = np.random.default_rng(c.seed)
    per_image, err = {}, {"A": [], "B": []}
    image_of = []
    for h in heldout:
        f = feats[h]
        kp = io_utils.to_native_points(f.keypoints, f.displayed_shape, final_rules[h])
        assoc = associations(h, len(kp), train_model, pairs, c.min_votes)
        entry: dict[str, Any] = {"associations": int(len(assoc))}
        if len(assoc) < c.min_associations:
            entry["status"] = "too few associations"
            per_image[f.name] = entry
            continue
        order = rng.permutation(len(assoc))
        pose, test = order[: len(order) // 2], order[len(order) // 2 :]
        x = kp[assoc[:, 0]].astype(np.float64)
        scores = {}
        for v in ("A", "B"):
            rec, cam, _ = train_model[v]
            scores[v] = _score(
                rec.track_xyz[assoc[:, 1]].astype(np.float64),
                x,
                pose,
                test,
                cam,
                params.pnp_threshold_px,
            )
        if scores["A"] is None or scores["B"] is None:
            entry["status"] = "pose failed"
            per_image[f.name] = entry
            continue
        entry.update(
            status="ok",
            pose_associations=int(len(pose)),
            test_associations=int(len(test)),
            **{f"median_px_{v}": float(np.median(scores[v])) for v in ("A", "B")},
            **{f"p95_px_{v}": float(np.percentile(scores[v], 95)) for v in ("A", "B")},
        )
        per_image[f.name] = entry
        for v in ("A", "B"):
            err[v].append(scores[v])
        image_of.append(np.full(len(test), len(image_of)))
    n_ok = len(image_of)
    if n_ok == 0:
        return {
            "per_image": per_image,
            "decision": "A",
            "reason": "no held-out image could be evaluated",
        }

    eA, eB = np.concatenate(err["A"]), np.concatenate(err["B"])
    groups = [np.flatnonzero(np.concatenate(image_of) == k) for k in range(n_ok)]
    diffs = []
    # Resample whole images: their residuals share a PnP pose and are correlated.
    for _ in range(c.bootstrap):
        pick = np.concatenate([groups[k] for k in rng.integers(0, n_ok, n_ok)])
        diffs.append(np.median(eA[pick]) - np.median(eB[pick]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    point = float(np.median(eA) - np.median(eB))
    if lo > 0:
        decision, reason = (
            "B",
            "refined intrinsics predict held-out images better (95 % interval above zero)",
        )
    elif hi < 0:
        decision, reason = (
            "A",
            "checkerboard intrinsics predict held-out images better (95 % interval below zero)",
        )
    else:
        decision, reason = (
            "A",
            (
                "no clear difference on held-out images; the checkerboard intrinsics "
                "(fewer parameters fitted to the wound images) are kept"
            ),
        )
    summary = {
        v: {
            "median_px": float(np.median(e)),
            "p95_px": float(np.percentile(e, 95)),
            "rms_px": float(np.sqrt(np.mean(e**2))),
            "n": int(len(e)),
        }
        for v, e in (("A", eA), ("B", eB))
    }
    log.info(
        "Held-out images (%d evaluated): median A %.3f px, B %.3f px, A-B %+.3f px [%+.3f, %+.3f] -> %s",
        n_ok,
        summary["A"]["median_px"],
        summary["B"]["median_px"],
        point,
        lo,
        hi,
        decision,
    )
    return {
        "heldout_images": [feats[h].name for h in heldout],
        "images_evaluated": n_ok,
        "per_image": per_image,
        "pooled": summary,
        "median_difference_A_minus_B_px": point,
        "ci95_px": [float(lo), float(hi)],
        "bootstrap": f"{c.bootstrap} resamples of held-out images",
        "decision": decision,
        "reason": reason,
        "protocol": "training-only reconstruction; pose from one half of the associations, "
        "error on the other half; identical associations for A and B",
    }
