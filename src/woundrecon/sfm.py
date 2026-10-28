"""Incremental structure-from-motion.

Pipeline (classical, in the spirit of COLMAP's incremental mapper):

1. Keypoints are mapped from the displayed frame into the native sensor frame
   and undistorted with the checkerboard calibration. Keypoints outside the
   calibrated field of view (where distortion is only extrapolated) are dropped.
2. All image pairs are matched and verified with the essential matrix.
3. Pairwise matches are chained into multi-view tracks (union-find); tracks
   that would observe two different keypoints in one image are discarded.
4. Initialisation from the pair with many inliers, a low homography/essential
   inlier ratio (not a pure rotation / planar degeneracy) and enough parallax.
5. Images are registered one at a time by PnP + RANSAC (next image = most
   2D-3D correspondences). For every image both remaining rotations into the
   native frame are tried, because EXIF orientation was removed: the wrong one
   misplaces the principal point and distortion centre and fits worse. The
   signal is weak, so an image is only re-oriented with a clear margin, and a
   leave-one-image-out diagnostic is reported after reconstruction.
6. New tracks are triangulated after each registration, with checks on
   parallax angle, depth sign and reprojection error.
7. Global bundle adjustment after each registration and at the end, with
   outlier rejection between rounds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from . import bundle_adjustment as ba
from . import features as feat
from . import io_utils
from .camera import Camera, camera_centre

log = logging.getLogger(__name__)

# Alternative rotation that keeps an image's orientation class (portrait/landscape).
_FLIP = {"cw": "ccw", "ccw": "cw", "none": "180", "180": "none"}


@dataclass
class SfMParams:
    """Native-pixel geometry thresholds, parallax in degrees and feature/orientation settings."""

    ratio_test: float = 0.8
    min_inliers: int = 60
    ransac_threshold_px: float = 2.0
    pnp_threshold_px: float = 6.0
    min_triangulation_angle_deg: float = 2.0
    max_reprojection_error_px: float = 4.0
    min_pnp_inliers: int = 40
    valid_radius_scale: float = 1.25
    orientation_margin: float = 0.2  # alternative rotation must fit >20 % better

    @classmethod
    def from_config(cls, c: dict[str, Any]) -> SfMParams:
        """Construct geometry settings from matching YAML keys, retaining dataclass defaults."""
        return cls(**{k: c[k] for k in cls.__dataclass_fields__ if k in c})


@dataclass
class Reconstruction:
    """State of the incremental reconstruction."""

    names: list[str]
    camera: Camera
    rules: list[str]  # displayed -> native rotation per image
    registered: np.ndarray  # (N,) bool
    rvecs: np.ndarray  # (N, 3)
    tvecs: np.ndarray  # (N, 3)
    track_obs: list[list[tuple[int, int]]] = field(default_factory=list)  # (image, keypoint)
    track_xyz: np.ndarray | None = None  # (T, 3), NaN = not triangulated
    track_rgb: np.ndarray | None = None  # (T, 3) uint8


class IncrementalSfM:
    """Incremental SfM over a list of :class:`features.ImageFeatures`."""

    def __init__(
        self,
        features: list[feat.ImageFeatures],
        camera: Camera,
        default_rules: list[str],
        params: SfMParams,
    ):
        """Initialize shared-camera incremental reconstruction and native keypoint views.

        Args:
            features: Per-image displayed-frame features, descriptors and colours.
            camera: Shared native-sensor intrinsics, with focal/principal point in pixels.
            default_rules: One displayed-to-native orientation rule per image.
            params: Matching, parallax and native-pixel geometric thresholds.
        """
        self.f = features
        self.cam = camera
        self.p = params
        n = len(features)
        self.rec = Reconstruction(
            [x.name for x in features],
            camera,
            list(default_rules),
            np.zeros(n, bool),
            np.zeros((n, 3)),
            np.zeros((n, 3)),
        )
        self.kp: list[np.ndarray] = [np.zeros((0, 2))] * n  # native distorted pixels
        self.und: list[np.ndarray] = [np.zeros((0, 2))] * n  # native undistorted pixels
        self.valid: list[np.ndarray] = [np.zeros(0, bool)] * n
        for i in range(n):
            self._set_rule(i, default_rules[i])
        self.pairs: dict[tuple[int, int], feat.PairMatches] = {}
        self.kp_track: list[np.ndarray] = []
        self.log: dict[str, Any] = {"registration_order": [], "orientation_tests": {}, "ba": []}

    # ------------------------------------------------------------------ #
    # Keypoint coordinates under an orientation hypothesis
    # ------------------------------------------------------------------ #
    def _native_coords(self, i: int, rule: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return native distorted/undistorted (N, 2) pixels and their field-of-view mask."""
        fi = self.f[i]
        kp = io_utils.to_native_points(fi.keypoints, fi.displayed_shape, rule)
        valid = self.cam.in_valid_region(kp)
        und = np.full_like(kp, np.nan)
        und[valid] = self.cam.undistort_pixels(kp[valid])
        return kp, und, valid

    def _set_rule(self, i: int, rule: str) -> None:
        """Apply an image's orientation rule consistently to pixels, rays and valid-region mask."""
        self.kp[i], self.und[i], self.valid[i] = self._native_coords(i, rule)
        self.rec.rules[i] = rule

    # ------------------------------------------------------------------ #
    # Matching and tracks
    # ------------------------------------------------------------------ #
    def match_all(self) -> None:
        """Exhaustive matching and essential-matrix verification of all pairs."""
        n = len(self.f)
        for i in range(n):
            for j in range(i + 1, n):
                vi, vj = np.flatnonzero(self.valid[i]), np.flatnonzero(self.valid[j])
                a, b = feat.match_descriptors(
                    self.f[i].descriptors[vi], self.f[j].descriptors[vj], self.p.ratio_test
                )
                a, b = vi[a], vj[b]
                if len(a) < self.p.min_inliers:
                    continue
                inl, n_h = feat.verify(
                    self.und[i][a], self.und[j][b], self.cam.K, self.p.ransac_threshold_px
                )
                if inl.sum() >= self.p.min_inliers:
                    self.pairs[(i, j)] = feat.PairMatches(i, j, a[inl], b[inl], len(a), n_h)
            log.info(
                "Matched image %d/%d (%s): %d verified pairs so far",
                i + 1,
                n,
                self.f[i].name,
                len(self.pairs),
            )

    def build_tracks(self) -> None:
        """Chain verified pairwise matches into tracks with a union-find structure."""
        offsets = np.cumsum([0] + [len(x.keypoints) for x in self.f])
        parent = np.arange(offsets[-1])

        def find(a: int) -> int:
            """Find the canonical observation-set representative with path compression."""
            root = a
            while parent[root] != root:
                root = parent[root]
            while parent[a] != root:  # path compression
                parent[a], a = root, parent[a]
            return root

        for (i, j), m in self.pairs.items():
            for a, b in zip(offsets[i] + m.idx_i, offsets[j] + m.idx_j, strict=True):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra

        groups: dict[int, list[int]] = {}
        for node in np.unique(
            np.concatenate(
                [offsets[i] + m.idx_i for (i, _), m in self.pairs.items()]
                + [offsets[j] + m.idx_j for (_, j), m in self.pairs.items()]
            )
        ):
            groups.setdefault(find(int(node)), []).append(int(node))

        image_of = np.searchsorted(offsets, np.arange(offsets[-1]), side="right") - 1
        tracks, n_conflict = [], 0
        for nodes in groups.values():
            imgs = image_of[nodes]
            if len(nodes) < 2:
                continue
            if len(set(imgs.tolist())) != len(nodes):  # same image twice -> ambiguous
                n_conflict += 1
                continue
            tracks.append(
                [(int(im), int(nd - offsets[im])) for im, nd in zip(imgs, nodes, strict=True)]
            )

        self.rec.track_obs = tracks
        self.rec.track_xyz = np.full((len(tracks), 3), np.nan)
        self.rec.track_rgb = np.zeros((len(tracks), 3), np.uint8)
        self.kp_track = [np.full(len(x.keypoints), -1, int) for x in self.f]
        for t, obs in enumerate(tracks):
            for im, k in obs:
                self.kp_track[im][k] = t
            im, k = obs[0]
            self.rec.track_rgb[t] = self.f[im].colors[k]
        lengths = np.array([len(t) for t in tracks])
        log.info(
            "Tracks: %d (conflicting dropped: %d), mean length %.2f",
            len(tracks),
            n_conflict,
            lengths.mean(),
        )
        self.log["tracks"] = {
            "count": len(tracks),
            "conflicts_dropped": n_conflict,
            "mean_length": float(lengths.mean()),
        }

    # ------------------------------------------------------------------ #
    # Geometry helpers
    # ------------------------------------------------------------------ #
    def _pose(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """Return one image's world-to-camera rotation matrix and SfM-unit translation."""
        R, _ = cv2.Rodrigues(self.rec.rvecs[i])
        return R, self.rec.tvecs[i]

    def _rays(self, i: int, kps: np.ndarray) -> np.ndarray:
        """Normalised coordinates of keypoints (from the undistorted pixels)."""
        u = self.und[i][kps]
        return np.c_[(u[:, 0] - self.cam.cx) / self.cam.fx, (u[:, 1] - self.cam.cy) / self.cam.fy]

    def _reproj_error(self, i: int, X: np.ndarray, kps: np.ndarray) -> np.ndarray:
        """Return (N,) native-pixel errors for world points and observed keypoint indices."""
        R, t = self._pose(i)
        Xc = X @ R.T + t
        proj = self.cam.project(Xc)
        err = np.linalg.norm(proj - self.kp[i][kps], axis=1)
        err[Xc[:, 2] <= 0] = np.inf
        return err

    @staticmethod
    def _angle_deg(c1: np.ndarray, c2: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Parallax angle at each point X between camera centres c1 and c2."""
        a, b = X - c1, X - c2
        cosang = np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
        return np.degrees(np.arccos(np.clip(cosang, -1, 1)))

    # ------------------------------------------------------------------ #
    # Initialisation
    # ------------------------------------------------------------------ #
    def initialise(self) -> tuple[int, int]:
        """Pick the initial pair and triangulate its tracks."""
        candidates = []
        for (i, j), m in self.pairs.items():
            h_ratio = m.n_homography_inliers / len(m.idx_i)
            if h_ratio > 0.85:
                continue
            feat.seed_opencv()
            E, mask = cv2.findEssentialMat(
                self.und[i][m.idx_i],
                self.und[j][m.idx_j],
                self.cam.K,
                cv2.USAC_MAGSAC,
                0.9999,
                self.p.ransac_threshold_px,
            )
            if E is None or E.shape != (3, 3):
                continue
            _, R, t, mask_pose = cv2.recoverPose(
                E, self.und[i][m.idx_i], self.und[j][m.idx_j], self.cam.K, mask=mask.copy()
            )
            good = mask_pose.ravel() > 0
            if good.sum() < self.p.min_inliers:
                continue
            P1, P2 = np.eye(3, 4), np.c_[R, t]
            X = cv2.triangulatePoints(
                P1, P2, self._rays(i, m.idx_i[good]).T, self._rays(j, m.idx_j[good]).T
            )
            X = (X[:3] / X[3]).T
            angle = np.median(self._angle_deg(np.zeros(3), camera_centre(R, t.ravel()), X))
            candidates.append(
                (good.sum() * min(angle, 8.0) / 8.0, i, j, R, t.ravel(), angle, h_ratio)
            )
        if not candidates:
            raise RuntimeError("No valid initial pair found")
        score, i, j, R, t, angle, h_ratio = max(candidates, key=lambda c: c[0])
        log.info(
            "Initial pair %s - %s: score %.0f, median parallax %.1f deg, H/E ratio %.2f",
            self.f[i].name,
            self.f[j].name,
            score,
            angle,
            h_ratio,
        )
        self.log["initial_pair"] = {
            "images": [self.f[i].name, self.f[j].name],
            "median_parallax_deg": float(angle),
            "h_e_ratio": float(h_ratio),
        }
        self.rec.rvecs[j] = cv2.Rodrigues(R)[0].ravel()
        self.rec.tvecs[j] = t
        self.rec.registered[[i, j]] = True
        self.log["registration_order"] += [self.f[i].name, self.f[j].name]
        self._triangulate_new(j)
        return i, j

    # ------------------------------------------------------------------ #
    # Triangulation
    # ------------------------------------------------------------------ #
    def _triangulate_new(self, new: int) -> int:
        """Triangulate tracks seen by image ``new`` and at least one other registered image."""
        reg = np.flatnonzero(self.rec.registered)
        centres = {r: camera_centre(*self._pose(r)) for r in reg}
        kps = np.flatnonzero(self.kp_track[new] >= 0)
        tracks = self.kp_track[new][kps]
        todo = np.isnan(self.rec.track_xyz[tracks, 0])
        kps, tracks = kps[todo], tracks[todo]
        # For each track pick the registered partner view with the largest baseline.
        by_partner: dict[int, list[tuple[int, int, int]]] = {}
        for k, t in zip(kps, tracks, strict=True):
            partners = [
                (im, kk)
                for im, kk in self.rec.track_obs[t]
                if im != new and self.rec.registered[im]
            ]
            if not partners:
                continue
            im, kk = max(partners, key=lambda p: np.linalg.norm(centres[p[0]] - centres[new]))
            by_partner.setdefault(im, []).append((k, kk, t))

        added = 0
        Rn, tn = self._pose(new)
        for im, items in by_partner.items():
            k_new, k_old, ts = (np.array(v) for v in zip(*items, strict=True))
            Ro, to = self._pose(im)
            X = cv2.triangulatePoints(
                np.c_[Rn, tn], np.c_[Ro, to], self._rays(new, k_new).T, self._rays(im, k_old).T
            )
            X = (X[:3] / X[3]).T
            ok = self._angle_deg(centres[new], centres[im], X) >= self.p.min_triangulation_angle_deg
            ok &= self._reproj_error(new, X, k_new) < self.p.max_reprojection_error_px
            ok &= self._reproj_error(im, X, k_old) < self.p.max_reprojection_error_px
            self.rec.track_xyz[ts[ok]] = X[ok]
            added += int(ok.sum())
            self._prune_track_observations(ts[ok])
        return added

    def _prune_track_observations(self, tracks: np.ndarray) -> None:
        """Detach observations (in any registered image) that disagree with a new 3D point.

        Tracks are built from pairwise matches, so a track can contain a wrong
        keypoint in an image that took no part in triangulation. Such
        observations would otherwise enter bundle adjustment as gross outliers.
        """
        by_image: dict[int, list[tuple[int, int]]] = {}
        for t in tracks:
            for im, k in self.rec.track_obs[t]:
                if self.rec.registered[im] and self.kp_track[im][k] == t:
                    by_image.setdefault(im, []).append((t, k))
        for im, items in by_image.items():
            ts, ks = (np.array(v) for v in zip(*items, strict=True))
            err = self._reproj_error(im, self.rec.track_xyz[ts], ks)
            bad = err > self.p.max_reprojection_error_px
            self.kp_track[im][ks[bad]] = -1

    # ------------------------------------------------------------------ #
    # Registration
    # ------------------------------------------------------------------ #
    def _pnp(self, i: int, rule: str) -> dict[str, Any] | None:
        """Estimate a world-to-camera pose from existing tracks under one displayed orientation."""
        kp, und, valid = self._native_coords(i, rule)
        kps = np.flatnonzero((self.kp_track[i] >= 0) & valid)
        tracks = self.kp_track[i][kps]
        has3d = ~np.isnan(self.rec.track_xyz[tracks, 0])
        kps, tracks = kps[has3d], tracks[has3d]
        if len(kps) < self.p.min_pnp_inliers:
            return None
        obj = self.rec.track_xyz[tracks]
        feat.seed_opencv()
        ok, rvec, tvec, inl = cv2.solvePnPRansac(
            obj,
            und[kps],
            self.cam.K,
            None,
            iterationsCount=10000,
            reprojectionError=self.p.pnp_threshold_px,
            confidence=0.9999,
            flags=cv2.SOLVEPNP_SQPNP,
        )
        if not ok or inl is None or len(inl) < self.p.min_pnp_inliers:
            return None
        inl = inl.ravel()
        rvec, tvec = cv2.solvePnPRefineLM(obj[inl], und[kps][inl], self.cam.K, None, rvec, tvec)
        R, _ = cv2.Rodrigues(rvec)
        proj = self.cam.project(obj[inl] @ R.T + tvec.ravel())
        err = np.linalg.norm(proj - kp[kps][inl], axis=1)
        return {
            "rvec": rvec.ravel(),
            "tvec": tvec.ravel(),
            "n_inliers": len(inl),
            "n_corr": len(kps),
            "median_err_px": float(np.median(err)),
            "rule": rule,
        }

    def register_next(self) -> int | None:
        """Register the unregistered image with most 2D-3D correspondences."""
        best, best_count = None, 0
        for i in np.flatnonzero(~self.rec.registered):
            t = self.kp_track[i][self.kp_track[i] >= 0]
            count = int((~np.isnan(self.rec.track_xyz[t, 0])).sum())
            if count > best_count:
                best, best_count = i, count
        if best is None or best_count < self.p.min_pnp_inliers:
            return None

        trials = [
            r
            for r in (
                self._pnp(best, self.rec.rules[best]),
                self._pnp(best, _FLIP[self.rec.rules[best]]),
            )
            if r
        ]
        if not trials:
            log.warning("Registration failed: %s", self.f[best].name)
            self.rec.registered[best] = False
            self.kp_track[best][:] = -1  # do not try again
            return -1
        self.log["orientation_tests"][self.f[best].name] = {
            r["rule"]: r["median_err_px"] for r in trials
        }
        # Keep the default rotation unless the alternative is clearly better. The
        # signal is weak (principal point only ~33 px off-centre), so the margin is large.
        chosen = trials[0] if trials[0]["rule"] == self.rec.rules[best] else trials[-1]
        for r in trials:
            if (
                r["rule"] != self.rec.rules[best]
                and r["median_err_px"] < (1 - self.p.orientation_margin) * chosen["median_err_px"]
            ):
                chosen = r
        if chosen["rule"] != self.rec.rules[best]:
            log.info(
                "  %s: orientation hypothesis '%s' fits better than '%s'",
                self.f[best].name,
                chosen["rule"],
                self.rec.rules[best],
            )
            self._set_rule(best, chosen["rule"])
        self.rec.rvecs[best], self.rec.tvecs[best] = chosen["rvec"], chosen["tvec"]
        self.rec.registered[best] = True
        # Detach PnP outliers: keypoints of the new image whose track's 3D point
        # does not reproject onto them (wrong match inside the track).
        kps = np.flatnonzero(self.kp_track[best] >= 0)
        tr = self.kp_track[best][kps]
        has3d = ~np.isnan(self.rec.track_xyz[tr, 0])
        err = self._reproj_error(best, self.rec.track_xyz[tr[has3d]], kps[has3d])
        self.kp_track[best][kps[has3d][err > self.p.max_reprojection_error_px]] = -1
        added = self._triangulate_new(best)
        self.log["registration_order"].append(self.f[best].name)
        log.info(
            "Registered %s: %d/%d PnP inliers, median error %.2f px, +%d points",
            self.f[best].name,
            chosen["n_inliers"],
            chosen["n_corr"],
            chosen["median_err_px"],
            added,
        )
        return int(best)

    # ------------------------------------------------------------------ #
    # Bundle adjustment
    # ------------------------------------------------------------------ #
    def _ba_problem(self) -> tuple[ba.BAProblem, np.ndarray, np.ndarray, np.ndarray]:
        """Collect all observations of triangulated tracks in registered images."""
        tracks = np.flatnonzero(~np.isnan(self.rec.track_xyz[:, 0]))
        reg_idx = np.flatnonzero(self.rec.registered)
        cam_map = -np.ones(len(self.f), int)
        cam_map[reg_idx] = np.arange(len(reg_idx))
        cam_i, pt_i, obs, obs_kp = [], [], [], []
        for p, t in enumerate(tracks):
            for im, k in self.rec.track_obs[t]:
                if self.rec.registered[im] and self.kp_track[im][k] == t:
                    cam_i.append(cam_map[im])
                    pt_i.append(p)
                    obs.append(self.kp[im][k])
                    obs_kp.append((im, k))
        problem = ba.BAProblem(
            self.rec.rvecs[reg_idx].copy(),
            self.rec.tvecs[reg_idx].copy(),
            self.rec.track_xyz[tracks].copy(),
            np.array(cam_i),
            np.array(pt_i),
            np.array(obs),
            self.cam,
            fixed_cameras=(0,),
        )
        return problem, tracks, reg_idx, np.array(obs_kp)

    def bundle_adjust(
        self, free: tuple[str, ...] = (), max_nfev: int = 50, reject: bool = True
    ) -> dict:
        """Global BA followed by rejection of observations with large residuals."""
        problem, tracks, reg_idx, obs_kp = self._ba_problem()
        stats = ba.solve(problem, free=free, max_nfev=max_nfev)
        self.rec.rvecs[reg_idx], self.rec.tvecs[reg_idx] = problem.rvecs, problem.tvecs
        self.rec.track_xyz[tracks] = problem.points
        if reject:
            err = np.linalg.norm(ba.residuals(problem, ()), axis=1)
            bad = err > self.p.max_reprojection_error_px
            for im, k in obs_kp[bad]:
                self.kp_track[im][k] = -1  # detach the observation
            # Points left with fewer than two observations are discarded.
            counts = np.bincount(problem.pt_idx[~bad], minlength=len(tracks))
            self.rec.track_xyz[tracks[counts < 2]] = np.nan
            stats["rejected_observations"] = int(bad.sum())
            stats["removed_points"] = int((counts < 2).sum())
        self.log["ba"].append({k: v for k, v in stats.items() if k != "dist"})
        return stats

    def save_pairs(self, path, fingerprint: str = "") -> None:
        """Save verified keypoint associations with feature counts and a dependency fingerprint."""
        keys = list(self.pairs)

        def ragged(arrays):  # 1-D object array, one entry per pair
            """Preserve variable match counts as one array entry per verified image pair."""
            out = np.empty(len(arrays), dtype=object)
            out[:] = arrays
            return out

        np.savez(
            path,
            keys=np.array(keys),
            idx_i=ragged([self.pairs[k].idx_i for k in keys]),
            idx_j=ragged([self.pairs[k].idx_j for k in keys]),
            n_putative=np.array([self.pairs[k].n_putative for k in keys]),
            n_h=np.array([self.pairs[k].n_homography_inliers for k in keys]),
            n_keypoints=np.array([len(f.keypoints) for f in self.f]),
            fingerprint=np.array(fingerprint),
        )

    def load_pairs(self, path) -> bool:
        """Load cached pairs; returns False (and loads nothing) if they do not fit the current features."""
        d = np.load(path, allow_pickle=True)
        n_kp = [len(f.keypoints) for f in self.f]
        pairs = {}
        for (i, j), a, b, p, h in zip(
            d["keys"], d["idx_i"], d["idx_j"], d["n_putative"], d["n_h"], strict=True
        ):
            i, j = int(i), int(j)
            if (
                not (0 <= i < len(n_kp) and 0 <= j < len(n_kp))
                or (len(a) and a.max() >= n_kp[i])
                or (len(b) and b.max() >= n_kp[j])
            ):
                log.warning(
                    "Cached pair (%d, %d) refers to keypoints that do not exist; cache rejected",
                    i,
                    j,
                )
                return False
            pairs[(i, j)] = feat.PairMatches(i, j, a, b, int(p), int(h))
        self.pairs = pairs
        log.info("Loaded %d cached verified pairs", len(self.pairs))
        return True

    def run(self) -> Reconstruction:
        """Build tracks, register views and refine an arbitrary-scale SfM reconstruction.

        Returns:
            Reconstruction with world points/translations in a common arbitrary length
            unit and shared intrinsics/observations in full-resolution native pixels.

        Raises:
            RuntimeError: No valid initial camera pair can be reconstructed.
            ValueError: Geometry or solver inputs violate their numerical contracts.
        """
        if not self.pairs:
            self.match_all()
        self.build_tracks()
        self.initialise()
        self.bundle_adjust(max_nfev=50)
        while True:
            new = self.register_next()
            if new is None:
                break
            if new >= 0:
                self.bundle_adjust(max_nfev=30)
        for _ in range(2):
            self.bundle_adjust(max_nfev=200)
        self._retriangulate()
        self.bundle_adjust(max_nfev=200)
        return self.rec

    def orientation_check_loo(self) -> dict[str, dict[str, float]]:
        """Orientation diagnostic after reconstruction (leave-one-image-out re-triangulation).

        For every registered image, its tracks are re-triangulated from the
        *other* images only (points refined with the image's own observations
        would favour whichever hypothesis was used), the observations are split
        into two disjoint halves, the image is re-posed from one half under both
        rotation hypotheses and the median reprojection error is measured on the
        other half. The poses of the other images and the intrinsics still come
        from the full reconstruction, so this is a diagnostic, not a fully
        held-out test. Returns the median error per hypothesis.
        """
        from .camera import triangulate_dlt

        out = {}
        rng = np.random.default_rng(0)
        poses = {i: self._pose(i) for i in np.flatnonzero(self.rec.registered)}
        for i in poses:
            X, kps = [], []
            for k in np.flatnonzero(self.kp_track[i] >= 0):
                t = self.kp_track[i][k]
                others = [
                    (im, kk)
                    for im, kk in self.rec.track_obs[t]
                    if im != i and im in poses and self.kp_track[im][kk] == t
                ]
                if len(others) < 2 or np.isnan(self.rec.track_xyz[t, 0]):
                    continue
                rays = np.array(
                    [self.cam.normalise(self.kp[im][kk : kk + 1])[0] for im, kk in others]
                )
                X.append(triangulate_dlt(rays, [poses[im] for im, _ in others]))
                kps.append(k)
            if len(X) < self.p.min_pnp_inliers:
                continue
            X, kps = np.array(X), np.array(kps)
            order = rng.permutation(len(kps))
            fit, test = order[: len(order) // 2], order[len(order) // 2 :]  # disjoint halves
            res = {}
            for rule in (self.rec.rules[i], _FLIP[self.rec.rules[i]]):
                kp, und, _ = self._native_coords(i, rule)
                feat.seed_opencv()
                ok, rvec, tvec, inl = cv2.solvePnPRansac(
                    X[fit],
                    und[kps[fit]],
                    self.cam.K,
                    None,
                    iterationsCount=5000,
                    reprojectionError=self.p.pnp_threshold_px,
                    confidence=0.9999,
                    flags=cv2.SOLVEPNP_SQPNP,
                )
                if not ok or inl is None or len(inl) < 6:
                    res[rule] = float("inf")
                    continue
                sel = fit[inl.ravel()]
                rvec, tvec = cv2.solvePnPRefineLM(
                    X[sel], und[kps[sel]], self.cam.K, None, rvec, tvec
                )
                R, _ = cv2.Rodrigues(rvec)
                err = np.linalg.norm(
                    self.cam.project(X[test] @ R.T + tvec.ravel()) - kp[kps[test]], axis=1
                )
                res[rule] = float(np.median(err))
            out[self.f[i].name] = res
        return out

    def _retriangulate(self) -> None:
        """Second chance for tracks that are not triangulated yet (all registered views)."""
        added = 0
        for i in np.flatnonzero(self.rec.registered):
            added += self._triangulate_new(i)
        log.info("Re-triangulation added %d points", added)

    # ------------------------------------------------------------------ #
    # Outputs
    # ------------------------------------------------------------------ #
    def observations(self) -> dict[str, np.ndarray]:
        """Arrays describing the final reconstruction (for saving and later stages)."""
        problem, tracks, reg_idx, obs_kp = self._ba_problem()
        err = np.linalg.norm(ba.residuals(problem, ()), axis=1)
        return {
            "points": problem.points,
            "colors": self.rec.track_rgb[tracks],
            "obs_image": reg_idx[problem.cam_idx],
            "obs_point": problem.pt_idx,
            "obs_xy": problem.obs,
            "obs_error_px": err,
            "obs_keypoint": obs_kp[:, 1],
        }
