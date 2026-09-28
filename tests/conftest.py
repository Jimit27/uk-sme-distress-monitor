import sys
from pathlib import Path

import pytest

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
FIXTURES = HERE / "fixtures"


@pytest.fixture(scope="session", autouse=True)
def build_fixture_zips():
    from fixtures.make_fixtures import build

    build()


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES
