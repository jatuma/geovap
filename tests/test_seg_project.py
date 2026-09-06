import numpy as np

from mapping.seg import taxonomy as T
from mapping.seg.project import VoteHist, edge_factor


def test_votehist_weighted_argmax_and_unlabelled():
    acc = VoteHist(3)
    road, terrain = T.COMMON_ID["road"], T.COMMON_ID["terrain"]
    # point 0: two weak road votes vs one strong terrain vote -> road wins by summed weight
    acc.update(np.array([0, 1]), np.array([road, road]), np.array([0.4, 0.9], np.float32), frame=5)
    acc.update(np.array([0]), np.array([road]), np.array([0.4], np.float32), frame=6)
    acc.update(np.array([0]), np.array([terrain]), np.array([0.7], np.float32), frame=7)
    out = acc.finalize()
    assert out["label"].tolist() == [road, road, T.IGNORE]
    assert out["n_views"].tolist() == [3, 1, 0]
    assert out["src_frame"][0] == 7  # strongest single vote is the provenance frame
    assert abs(out["conf"][0] / 255 - 0.8 / 1.5) < 0.01 and out["conf"][1] == 255 and out["conf"][2] == 0


def test_votehist_float32_keeps_small_votes():
    acc = VoteHist(1)
    for _ in range(200):
        acc.update(np.array([0]), np.array([0]), np.array([1.0], np.float32), 0)
    for _ in range(300):  # 300 x 0.05 = 15 < 200: class 0 still wins, but the votes must be counted, not lost
        acc.update(np.array([0]), np.array([1]), np.array([0.05], np.float32), 1)
    assert abs(acc.wsum[0, 1] - 15.0) < 0.01


def test_edge_factor_zero_on_boundary_one_inside():
    m = np.zeros((20, 40), np.uint8)
    m[:, 20:] = 1
    ef = edge_factor(m, 3.0)
    assert ef[10, 19] == 0.0 and ef[10, 20] == 0.0
    assert ef[10, 5] == 1.0 and ef[10, 35] == 1.0
    assert 0.0 < ef[10, 17] < 1.0
