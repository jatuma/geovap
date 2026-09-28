import numpy as np

from mapping.seg import evaluate as E
from mapping.seg import taxonomy as T


def test_confusion_and_iou():
    gt = np.array([[0, 0, 1, 1], [0, 0, 1, 255]], np.uint8)
    pred = np.array([[0, 1, 1, 1], [0, 0, 0, 0]], np.uint8)
    mask = gt != 255
    cm = E.confusion(gt, pred, mask)
    assert cm[0, 0] == 3 and cm[0, 1] == 1 and cm[1, 1] == 2 and cm[1, 0] == 1
    iou = E.iou_from_conf(cm)
    assert abs(iou[0] - 3 / 5) < 1e-9 and abs(iou[1] - 2 / 4) < 1e-9 and np.isnan(iou[2])


def test_cos_weights_and_boundary_iou():
    W = E.lat_weights()
    assert W.shape == (1000, 2000) and W[500, 0] > 0.999 and W[0, 0] < 0.01
    gt = np.zeros((40, 40), np.uint8)
    gt[10:30, 10:30] = 2
    pred = gt.copy()
    valid = np.ones_like(gt, bool)
    i, u = E.boundary_iou(gt, pred, valid, 2, 2)
    assert i == u and i > 0
    pred2 = np.zeros_like(gt)
    pred2[12:32, 12:32] = 2  # shifted by 2 px
    i2, u2 = E.boundary_iou(gt, pred2, valid, 2, 2)
    assert 0 < i2 / u2 < 1


def test_transition_distance():
    pred = np.zeros((20, 40), np.uint8)
    pred[:, 20:] = 1
    band = np.zeros_like(pred, bool)
    band[:, 18] = True
    d = E.transition_distance(pred, band, (1,), np.ones_like(pred, bool))
    assert np.allclose(d, 1.0) or np.allclose(d, 2.0)


def test_nearfield_band_edge_distance():
    from mapping.seg import nearfield

    photo = np.full((4000, 8000, 3), 120, np.uint8)
    photo[:, 4000:] = 30  # one vertical edge at u = 4000 (-> col 1000 at 2000 px)
    bands = np.zeros((1000, 2000), np.uint8)
    bands[600:800, 1010:1016] = 1  # band 10 px right of the edge
    r = nearfield.band_edge_distance(photo, bands)
    assert r["n"] > 100 and 9 <= r["median_px"] <= 13
    bands2 = np.zeros_like(bands)
    bands2[600:800, 998:1004] = 1
    assert nearfield.band_edge_distance(photo, bands2)["median_px"] <= 2
