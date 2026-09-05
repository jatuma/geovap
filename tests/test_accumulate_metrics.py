import numpy as np

from mapping import metrics
from mapping.accumulate import ColourTopK, NearestInTime
from mapping.sample import linear_to_srgb_u8, srgb_to_linear


def test_srgb_roundtrip():
    x = np.arange(256, dtype=np.uint8).reshape(-1, 1).repeat(3, 1)
    assert np.array_equal(linear_to_srgb_u8(srgb_to_linear(x)), x)


def test_topk_median_rejects_outlier():
    acc = ColourTopK(3, k=5)
    grey = srgb_to_linear(np.array([[120, 120, 120]], np.uint8)).repeat(3, 0)
    for f in range(4):
        acc.update(np.arange(3), grey + f * 0.001, np.full(3, 0.5 - 0.01 * f, np.float32), f)
    # one outlier with the best score (would win a 'best single frame' policy)
    acc.update(np.array([0]), srgb_to_linear(np.array([[255, 0, 0]], np.uint8)), np.array([0.9], np.float32), 9)
    out = acc.finalize()
    assert np.all(out["n_views"] == [5, 4, 4])
    assert abs(int(out["rgb"][0, 0]) - 120) <= 2 and abs(int(out["rgb"][0, 2]) - 120) <= 2
    assert out["src_image"][0] == 9  # best score is still reported as source


def test_topk_keeps_best_scores_only():
    acc = ColourTopK(1, k=2)
    c = srgb_to_linear(np.array([[10, 10, 10]], np.uint8))
    for f, s in enumerate([0.1, 0.5, 0.3, 0.9]):
        acc.update(np.array([0]), c, np.array([s], np.float32), f)
    assert np.allclose(sorted(acc.score[0].tolist()), [0.5, 0.9])


def test_nearest_in_time_slots():
    acc = NearestInTime(4)
    acc.update(np.array([1, 3]), np.array([[1, 2, 3], [4, 5, 6]], np.uint8), 7, np.array([3.0, 4.0]))
    assert acc.filled.tolist() == [False, True, False, True] and acc.frame[3] == 7


def test_hist_stats_percentiles():
    de = np.concatenate([np.full(900, 4.0), np.full(100, 30.0)]).astype(np.float32)
    h = metrics.StrataHist()
    h.add(de, np.ones(len(de), bool), **{k: np.zeros(len(de), np.int64) for k in metrics.STRATA})
    s = metrics.hist_stats(h.total_hist())
    assert abs(s["median"] - 4.0) < 0.05 and abs(s["pct_gt_20"] - 10.0) < 1e-6 and abs(s["p95"] - 30.0) < 0.05


def test_delta_e_zero_for_identical():
    rgb = np.random.default_rng(0).integers(0, 256, (1000, 3), dtype=np.uint8)
    d76, d00 = metrics.delta_e(rgb, rgb)
    assert d76.max() < 1e-4 and d00.max() < 1e-4
