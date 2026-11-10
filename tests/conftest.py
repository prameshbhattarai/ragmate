import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))  # lets tests `import fakes`

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample"


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Every test gets its own RAGMATE_HOME, so nothing touches the project's .ragmate/."""
    home = tmp_path / "ragmate-home"
    monkeypatch.setenv("RAGMATE_HOME", str(home))
    return home


@pytest.fixture
def settings():
    from ragmate.config import get_settings

    return get_settings()


@pytest.fixture
def store(settings):
    """A real Chroma store on disk (in tmp), with the model-free HashingEmbeddings."""
    from ragmate.store import HashingEmbeddings, open_store

    return open_store(settings, HashingEmbeddings())


@pytest.fixture
def sample_store(store, settings):
    from ragmate.ingest import ingest_paths

    ingest_paths([SAMPLE_DIR], store, settings)
    return store
