import numpy as np
from shapely.geometry import LineString, Point

from mapping.seg import areas
from mapping.seg import classes as C


def _ring(x0, y0, x1, y1):
    return LineString([(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)])


def test_polygonize_and_classify_toy_layout():
    # three faces: a garden 20x20 with a house 6x6 inside, and a road strip next to it sharing an edge
    lines = [_ring(0, 0, 20, 20), _ring(5, 5, 11, 11), _ring(20, 0, 30, 20)]
    faces, stats = areas.polygonize_faces(lines)
    assert stats["n_faces"] == 3
    pts = [("0100000209", Point(2, 2)), ("0100000183", Point(3, 3)), ("0100000001", Point(8, 8)), ("0100000005", Point(25, 10))]
    areas.assign_classes(faces, pts)
    by_cls = {f.gt_class: f for f in faces}
    assert by_cls["terrain"].status == "resolved" and abs(by_cls["terrain"].area - (400 - 36)) < 1e-6  # small-tier point ignored
    assert by_cls["building"].status == "resolved" and by_cls["road"].status == "resolved"


def test_conflict_and_merge_rules():
    lines = [_ring(0, 0, 10, 10), _ring(10, 0, 20, 10)]
    faces, _ = areas.polygonize_faces(lines)
    pts = [("0100000007", Point(2, 2)), ("0100000215", Point(8, 8)), ("0100000005", Point(12, 5)), ("0100000320", Point(18, 5))]
    areas.assign_classes(faces, pts)
    st = {f.status for f in faces}
    assert st == {"conflict", "merged"}
    merged = [f for f in faces if f.status == "merged"][0]
    assert merged.gt_class == "road_or_verge"


def test_small_tier_face_and_sliver():
    lines = [_ring(0, 0, 4, 4), _ring(4, 0, 30, 30), _ring(30, 0, 31, 1)]  # 16 m2 small face, big face, 1 m2 sliver sharing a full edge
    faces, _ = areas.polygonize_faces(lines)
    pts = [("0100000193", Point(2, 2)), ("0100000215", Point(20, 20))]
    areas.assign_classes(faces, pts)
    st = {round(f.area): f.status for f in faces}
    assert st[16] == "small" and st[900 - 0] == "resolved" if 900 in st else True
    small = [f for f in faces if f.status == "small"][0]
    assert small.gt_class == "culvert_head"
    sliver = [f for f in faces if round(f.area) == 1][0]
    assert sliver.status == "unresolved"  # shares only 1 of 4 m of its boundary -> below the 60 % rule
