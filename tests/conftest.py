import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.data.synthetic import FixtureSpec, make_fixture  # noqa: E402


@pytest.fixture
def smoke_cfg(tmp_path):
    """Smoke config with all storage redirected into a temporary directory."""
    return load_config(ROOT / "configs" / "colab_smoke_test.yaml", [f"paths.drive_root={tmp_path.as_posix()}/drive"])


@pytest.fixture
def fixture_root(tmp_path):
    root = tmp_path / "fixture"
    make_fixture(root, FixtureSpec(n_items=5, views_per_item=3, seed=1))
    return root
