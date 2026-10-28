"""Dense matching must replace infeasible partners before dropping a reference view."""

from types import SimpleNamespace

import numpy as np
import pytest

from woundrecon import dense
from woundrecon.config import load_config
from woundrecon.stages import dense as stage


def _views(count=5):
    return [SimpleNamespace(name=f"view_{i}.png") for i in range(count)]


def test_dense_stage_keeps_reserve_candidates_until_enough_pairs_match(tmp_path, monkeypatch):
    cfg = load_config(data_root=tmp_path, output_dir=tmp_path / "outputs")
    cfg["dense"]["neighbours_per_view"] = 2
    views = _views()
    points = np.arange(60, dtype=float).reshape(20, 3)
    sfm = {
        "points": points,
        "obs_image": np.repeat(np.arange(5), 20),
        "obs_point": np.tile(np.arange(20), 5),
    }
    monkeypatch.setattr(stage, "load_sfm", lambda cfg: sfm)
    monkeypatch.setattr(stage, "load_views", lambda *args: (views, list(range(5))))

    def ranked(views, shared, k, *args):
        assert k == len(views) - 1
        return {0: [1, 2, 3, 4]}

    calls = []

    def match(ref, partner, common, params):
        calls.append(partner.name)
        np.testing.assert_array_equal(common, points)
        if partner.name == "view_1.png":
            return np.full((2, 2), np.nan), {"skipped": "disparity range too large"}
        return np.ones((2, 2), np.float32), {"valid_px": 4}

    class ReachedFusion(Exception):
        pass

    def fuse(depths, params):
        assert len(depths) == 2
        raise ReachedFusion

    monkeypatch.setattr(dense, "select_neighbours", ranked)
    monkeypatch.setattr(dense, "pair_depth", match)
    monkeypatch.setattr(dense, "fuse_reference", fuse)
    with pytest.raises(ReachedFusion):
        stage.run(cfg)
    assert calls == ["view_1.png", "view_2.png", "view_3.png"]


@pytest.mark.parametrize("invalid", [np.nan, np.inf])
def test_empty_depth_result_does_not_consume_a_partner_slot(monkeypatch, invalid):
    views = _views(4)

    def match(ref, partner, common, params):
        value = invalid if partner is views[1] else 1.0
        return np.full((2, 2), value), {"valid_px": 0 if partner is views[1] else 4}

    monkeypatch.setattr(dense, "pair_depth", match)
    depths, stats, partners = stage._match_reference(
        0, views, [1, 2, 3], [{0}] * 4, np.zeros((1, 3)), dense.StereoParams(), 2
    )
    assert len(depths) == 2
    assert partners == ["view_2.png", "view_3.png"]
    assert stats["view_1.png"]["skipped"] == "no finite depth samples"


def test_exhausted_candidates_do_not_fabricate_required_support(monkeypatch):
    views = _views(3)

    def match(ref, partner, common, params):
        if partner is views[1]:
            return np.ones((2, 2)), {"valid_px": 4}
        return np.full((2, 2), np.nan), {"skipped": "too few shared points"}

    monkeypatch.setattr(dense, "pair_depth", match)
    depths, stats, partners = stage._match_reference(
        0, views, [1, 2], [{0}] * 3, np.zeros((1, 3)), dense.StereoParams(), 2
    )
    assert len(depths) == 1
    assert partners == ["view_1.png"]
    assert len(stats) == 2
