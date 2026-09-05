def test_imports():
    import numba  # noqa: F401
    import laspy  # noqa: F401
    import cv2  # noqa: F401
    import skimage  # noqa: F401

    from mapping import config, compat

    compat.ensure_experiments_on_path()
    from common import io_data  # noqa: F401

    assert config.PANO_W == 8000
