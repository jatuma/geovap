def test_imports():
    import numba  # noqa: F401
    import laspy  # noqa: F401
    import cv2  # noqa: F401
    import skimage  # noqa: F401

    from mapping import config
    # replaces the pre-refactor `compat.ensure_experiments_on_path(); from common import io_data`
    # smoke check: the vendor-export reader now lives here, reachable without sys.path surgery.
    from geovap.io.adapters.poses import ladybug_export_csv  # noqa: F401

    assert config.PANO_W == 8000
