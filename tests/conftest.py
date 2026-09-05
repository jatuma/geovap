import pytest

from mapping import config


@pytest.fixture(scope="session")
def has_data() -> bool:
    return config.EXPORT_CSV.exists()


@pytest.fixture(scope="session")
def poses(has_data):
    if not has_data:
        pytest.skip("dataset not available")
    from mapping.poses import load_poses

    return load_poses()
