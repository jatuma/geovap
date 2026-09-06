import numpy as np

from mapping.seg import classes as C
from mapping.seg.point_labels import label_points
from mapping.seg.rasters import Grid


class FakeRasters:
    """Flat world: terrain face everywhere, a building face in [20,30]x[20,30], a fence line along x=10, y in [0,50]."""

    def __init__(self):
        self.grid = Grid(0.0, 0.0, 0.1, 500, 500)
        self.dtm_grid = Grid(0.0, 0.0, 0.5, 100, 100)
        ny, nx = self.grid.ny, self.grid.nx
        fc = np.full((ny, nx), C.BY_NAME["terrain"].id, np.uint8)
        fc[200:300, 200:300] = C.BY_NAME["building"].id
        self.face_class = fc
        fb = fc.copy()
        fb[195:305, 195:305] = C.BY_NAME["building"].id
        self.face_class_buf = fb
        jj = np.arange(nx)[None, :].repeat(ny, 0)
        d_fence = np.abs(jj - 100) * 1.0  # dm
        far = np.full((ny, nx), 255, np.uint8)
        self.dist = {"fence": np.clip(d_fence, 0, 255).astype(np.uint8), "wall": far, "rail": far, "bldg_edge": far, "struct_edge": far}
        self.z = {"fence": np.zeros((ny, nx), np.float16), "rail": np.zeros((ny, nx), np.float16)}
        self.pole_dist = far
        self.dtm = np.zeros((100, 100), np.float32)

    def hag(self, e, n, z):
        return np.asarray(z, np.float32)


def test_rules_on_synthetic_points():
    R = FakeRasters()
    pts = np.array([
        [5.0, 5.0, 0.05],  # ground in terrain face -> terrain
        [5.0, 5.0, 1.5],  # above ground in green face -> vegetation
        [10.1, 20.0, 1.0],  # on the fence line, 1 m high -> fence
        [10.6, 20.0, 1.0],  # 0.6 m from fence, 1 m high -> ignore ring
        [25.0, 25.0, 3.0],  # facade / roof inside building face -> building
        [25.0, 25.0, 0.05],  # floor inside building face -> ignore
        [5.0, 5.0, 0.35],  # 0.35 m in green face -> ignore (between ground and vegetation)
    ])
    lab = label_points(pts, R)
    names = [C.BY_ID[l].name if l != 255 else "ignore" for l in lab]
    assert names == ["terrain", "vegetation", "fence", "ignore", "building", "ignore", "ignore"]


def test_points_outside_grid_are_ignored():
    R = FakeRasters()
    lab = label_points(np.array([[-5.0, -5.0, 0.0], [1000.0, 1000.0, 0.0]]), R)
    assert (lab == 255).all()
