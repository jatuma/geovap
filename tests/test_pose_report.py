"""S7: unit tests for `mapping.pose_report`'s pure/synthetic pieces -- no CloudStore, no real cache
data, so these run in the fast (non-slow) suite. The Pool-based pieces that need the real cache
(`compare_pose_sources`, `colour_de_comparison`, `interpolation_benefit`, `invariance_check`) are
exercised for real by `uv run python -m mapping.cli.pose_report run` (see `mapping/README.md`), not
re-tested here against a mocked store.
"""
from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from mapping import pose_report as pr
from geovap.domain.model.poses import Poses


# ------------------------------------------------------------------------------------- baseline (S2)
def test_baseline_summary_tiny_synthetic_table(tmp_path):
    """`baseline_summary`/`to_markdown` on a tiny hand-built `frame_quality.csv`-like table (4 rows:
    2 clean, 1 usable, 1 reject; one turning)."""
    csv_path = tmp_path / "frame_quality.csv"
    fields = ["du", "dv", "du_mad", "dv_mad", "inlier8", "yaw_rate", "cls"]
    rows = [
        {"du": "1.0", "dv": "2.0", "du_mad": "3.0", "dv_mad": "4.0", "inlier8": "0.5", "yaw_rate": "1.0", "cls": "clean"},
        {"du": "-2.0", "dv": "-1.0", "du_mad": "5.0", "dv_mad": "6.0", "inlier8": "0.3", "yaw_rate": "20.0", "cls": "clean"},  # turning
        {"du": "", "dv": "", "du_mad": "", "dv_mad": "", "inlier8": "", "yaw_rate": "0.5", "cls": "usable"},  # no edge fit
        {"du": "10.0", "dv": "10.0", "du_mad": "9.0", "dv_mad": "9.0", "inlier8": "0.05", "yaw_rate": "0.0", "cls": "reject"},
    ]
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    summary = pr.baseline_summary(csv_path)
    assert summary["overall"]["n"] == 4
    assert summary["overall"]["n_with_edge_fit"] == 3
    assert summary["overall"]["median_abs_du_px"] == pytest.approx(2.0)  # median(|1|,|-2|,|10|) = 2
    assert summary["n_turning"] == 1
    assert summary["n_straight"] == 3
    assert summary["turning"]["median_abs_du_px"] == pytest.approx(2.0)
    assert summary["by_class"]["clean"]["n"] == 2
    assert summary["by_class"]["reject"]["median_inlier_fraction"] == pytest.approx(0.05)

    md = pr.to_markdown(summary)
    assert "n = 4" in md
    assert "class=clean" in md and "class=reject" in md


# ------------------------------------------------------------------------------------- S7 task 1 agg
def _row(du_a, dv_a, du_b, dv_b, inl_a=0.2, inl_b=0.2, n=500):
    return {"du_a": du_a, "dv_a": dv_a, "dum_a": abs(du_a) + 1, "dvm_a": abs(dv_a) + 1, "inl_a": inl_a, "n_a": n,
            "du_b": du_b, "dv_b": dv_b, "dum_b": abs(du_b) + 1, "dvm_b": abs(dv_b) + 1, "inl_b": inl_b, "n_b": n,
            "frame": 0, "pass_a": 0, "pass_b": 0}


def test_aggregate_compare_rows_empty():
    assert pr.aggregate_compare_rows([]) == {"n": 0}


def test_aggregate_compare_rows_medians_and_flags():
    rows = [
        _row(0.0, 0.0, 0.0, 0.0),      # unchanged
        _row(10.0, 0.0, 1.0, 0.0),     # improved (mag 10 -> 1, > 2px better)
        _row(1.0, 0.0, 10.0, 0.0),     # worsened (mag 1 -> 10, > 2px worse)
        _row(3.0, 4.0, 3.0, 4.0),      # unchanged (mag 5 -> 5)
    ]
    out = pr.aggregate_compare_rows(rows)
    assert out["n"] == 4
    assert out["n_fit_a"] == 4 and out["n_fit_b"] == 4
    # |du|: a = [0, 10, 1, 3] -> median 2.0; b = [0, 1, 10, 3] -> median 2.0
    assert out["abs_du_a"]["median"] == pytest.approx(2.0)
    assert out["abs_du_b"]["median"] == pytest.approx(2.0)
    assert out["n_both_fit"] == 4
    assert out["frac_improved_gt2px"] == pytest.approx(0.25)  # 1/4
    assert out["frac_worsened_gt2px"] == pytest.approx(0.25)  # 1/4


def test_aggregate_compare_rows_ignores_nan_side():
    rows = [_row(1.0, 1.0, 2.0, 2.0)]
    rows[0]["du_a"] = None
    out = pr.aggregate_compare_rows(rows)
    assert out["abs_du_a"]["n"] == 0
    assert out["n_both_fit"] == 0  # a-side missing -> not counted as "both fit"
    assert out["frac_improved_gt2px"] is None


def test_stat_helper_nan_and_empty():
    assert pr._stat(np.array([np.nan, np.nan])) == {"median": None, "p95": None, "n": 0}
    s = pr._stat(np.array([1.0, 2.0, 3.0, np.nan]))
    assert s["n"] == 3 and s["median"] == pytest.approx(2.0)


# ------------------------------------------------------------------------------ S7 task 3 (conflict)
def test_conflict_summary_missing_file(tmp_path):
    out = pr.conflict_summary(tmp_path / "no_such.json")
    assert out == {"available": False, "path": str(tmp_path / "no_such.json")}


def test_conflict_summary_reads_existing(tmp_path):
    path = tmp_path / "conflict_reg.json"
    path.write_text(json.dumps({
        "overall_true_cross_pass": {"n_frames": 10, "n_conflict_before": 3, "n_conflict_after": 1},
        "overall_all_n2gt0_frames": {"n_frames": 12},
        "cloud_icp_pairs_summary": {"n_pairs": 5, "median_rms_before": 0.1, "median_rms_after": 0.02},
    }))
    out = pr.conflict_summary(path)
    assert out["available"] is True
    assert out["true_cross_pass"]["n_conflict_before"] == 3
    assert out["true_cross_pass"]["n_conflict_after"] == 1
    assert out["cloud_icp_pairs_summary"]["n_pairs"] == 5


# --------------------------------------------------------------------------- S7 task 4 (interp bench)
def _straight_line_poses(n: int, dt: float = 0.6, pass_id: int = 0) -> Poses:
    t = 1000.0 + np.arange(n) * dt
    origin = np.stack([100.0 * np.arange(n), np.zeros(n), 50.0 * np.ones(n)], axis=1)
    return Poses(
        filename=np.array([f"f{k}.jpg" for k in range(n)], dtype=object), t=t, origin=origin,
        roll=np.zeros(n), pitch=np.zeros(n), yaw=np.linspace(0.0, 90.0, n),
        pass_id=np.full(n, pass_id, dtype=np.int32), speed=np.zeros(n), source="test",
    )


def test_neighbor_linear_yaw_matches_hand_computation():
    poses = _straight_line_poses(5)  # yaw = 0, 22.5, 45, 67.5, 90
    pred = pr._neighbor_linear_yaw(poses, 2)  # neighbours k=1 (22.5) and k=3 (67.5) -> exact midpoint
    assert pred == pytest.approx(45.0)


def test_neighbor_linear_yaw_nan_at_pass_boundary():
    poses = _straight_line_poses(3)
    assert np.isnan(pr._neighbor_linear_yaw(poses, 0))  # no left neighbour
    assert np.isnan(pr._neighbor_linear_yaw(poses, 2))  # no right neighbour


def test_neighbor_linear_yaw_wraps_across_180():
    poses = _straight_line_poses(3)
    poses.yaw = np.array([170.0, 180.0, -170.0])  # true unwrapped progression: 170 -> 180 -> 190(=-170)
    pred = pr._neighbor_linear_yaw(poses, 1)
    assert pred == pytest.approx(-180.0)  # the (-180, 180] wrap maps exactly 180 deg to -180


def test_ang_diff_wraps():
    assert pr._ang_diff(np.array([179.0]), np.array([-179.0]))[0] == pytest.approx(2.0)
    assert pr._ang_diff(np.array([10.0]), np.array([5.0]))[0] == pytest.approx(5.0)


# ---------------------------------------------------------------------------------- S7 group selection
def test_passes_with_big_transform(tmp_path):
    path = tmp_path / "pass_transforms.json"
    path.write_text(json.dumps({"passes": {
        "0": {"t": [0.1, 0.0, 0.0], "yaw_deg": 0.0},
        "1": {"t": [0.5, 0.2, 0.0], "yaw_deg": 1.0},  # |t| = 0.54 > 0.3
        "2": {"t": [0.0, 0.0, 0.0], "yaw_deg": 0.0},
        "bad": {"t": [1.0, 0.0, 0.0], "yaw_deg": 0.0},  # non-integer key, ignored
    }}))
    assert pr.passes_with_big_transform(path, thresh_m=0.3) == {1}


def test_status_column_export_is_none():
    assert pr._status_column("export") is None


def test_status_column_reads_csv(tmp_path, monkeypatch):
    path = tmp_path / "poses_x.csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "status"])
        w.writerow([1, "traj_rot+reg"])
        w.writerow([0, "kept+refined+reg"])
    col = pr._status_column(path)
    assert col[0] == "kept+refined+reg"
    assert col[1] == "traj_rot+reg"


# ---------------------------------------------------- S7 task 6 (report markdown, no CloudStore needed)
def _tiny_final_summary(slow: dict) -> dict:
    """A hand-built stand-in for `run_final_report`'s `summary` dict -- exercises `_final_markdown`,
    `_summary_bullets` and `_slow_regression_markdown` (the code path the reviewer found missing) on a
    tiny synthetic table, with no CloudStore / real cache dependency."""
    group = {"n": 2, "abs_du_a": {"median": 1.0}, "abs_du_b": {"median": 0.5}, "abs_dv_a": {"median": 1.0},
             "abs_dv_b": {"median": 0.5}, "mad_du_a": {"median": 8.0}, "mad_du_b": {"median": 7.0},
             "inlier8_a": {"median": 0.2}, "inlier8_b": {"median": 0.3}, "frac_improved_gt2px": 0.5, "frac_worsened_gt2px": 0.1}
    return {
        "a": "export", "b": "corrected",
        "identity": {"hash_a": "aaa", "hash_b": "bbb", "len_a": 2, "len_b": 2,
                     "b_traj_attached": True, "b_traj_mode": "rot_only", "pose_at_idx_0_exact": True},
        "compare_groups": {"turning": group, "straight": group, "refined_or_reg": group},
        "colour_de_turning": {"n": 2, "de_a": {"median": 10.0}, "de_b": {"median": 9.0}, "n_improved": 1, "n_worsened": 1},
        "cross_pass_conflict": {"available": False, "path": "nope.json"},
        "interpolation_benefit": {"n_turning_covered": 2, "dense_model_resid_deg": {"median": 0.2, "p95": 0.4},
                                   "old_model_resid_deg": {"median": 5.0, "p95": 10.0}},
        "invariance": {"worst_px": 2.76e-6, "n_passes": 1, "per_pass_worst_px": {"0": 2.76e-6}},
        "slow_regression_037": slow,
        "assemble_provenance": {
            "s4_overlay": {"n_overlaid": 1, "n_pass_changed": 0, "dpos_median_m": 0.1, "dpos_p95_m": 0.2, "dpos_max_m": 0.3,
                           "dyaw_median_deg": 0.1, "dyaw_p95_deg": 0.2, "dyaw_max_deg": 0.3},
            "status_counts": {"traj_rot+reg": 1, "kept+reg": 1},
        },
    }


def test_final_markdown_has_summary_and_slow_sections_skipped():
    s = _tiny_final_summary({"skipped": "store not built"})
    md = pr._final_markdown(s)
    assert "## Summary" in md
    assert "## 6. Slow regression" in md
    assert "skipped: store not built" in md
    # section 1-5 headings all present
    for i in range(1, 6):
        assert f"## {i}." in md


def test_final_markdown_slow_regression_populated():
    slow = {
        "median_de76_export": 6.076, "n_export": 100,
        "median_de76_corrected_registered": 6.088, "n_corrected_registered": 100,
        "median_de76_corrected_unregistered": 7.8, "n_corrected_unregistered": 100,
        "delta_registered": 0.012, "delta_unregistered": 1.724, "tolerance": 0.1, "passes_registered": True,
        "lever_arm_example": {"pass_id": 0, "frame_idx": 0, "yaw_deg": 0.1, "t_m": 0.06,
                               "dist_from_centre_m": 470.0, "shift_m": 0.86},
    }
    s = _tiny_final_summary(slow)
    md = pr._final_markdown(s)
    assert "6.076" in md and "6.088" in md and "7.8" in md
    assert "470" in md and "0.86" in md  # lever-arm example numbers surface in both Summary and section 6
    assert md.count("6.088") >= 2  # cited once in Summary, once in section 6


def test_final_markdown_invariance_wording_reflects_target_miss():
    s = _tiny_final_summary({"error": "boom"})
    md = pr._final_markdown(s)
    assert "not computed: boom" in md
    assert "above the 1e-6 px aspirational target" in md
    assert "2.76e-06 px" in md


# ------------------------------------------------------------------------ S7 task 5/6 (slow regression)
def test_slow_regression_037_skips_when_store_missing(tmp_path):
    out = pr.slow_regression_037(transforms_path=tmp_path / "no_such.json")
    assert "skipped" in out


def test_lever_arm_example_missing_pass(tmp_path):
    path = tmp_path / "pass_transforms.json"
    path.write_text(json.dumps({"passes": {"1": {"centre": [0.0, 0.0], "yaw_deg": 0.0, "t": [0.0, 0.0, 0.0]}}}))
    out = pr.lever_arm_example(transforms_path=path, pass_id=99)
    assert "error" in out
