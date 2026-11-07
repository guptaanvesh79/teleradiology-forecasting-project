import json
from pathlib import Path

import pytest

from forecasting.config import specification
from forecasting.data import sha256_file
from forecasting.pipeline import FROZEN_FILES, fingerprint, frozen_selection, runtime_versions


@pytest.fixture
def frozen_root(tmp_path):
    for filename in FROZEN_FILES:
        path = tmp_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"Placeholder for {filename}\n")
    data = tmp_path / "data/studies.csv"
    data.parent.mkdir()
    data.write_text("synthetic input placeholder\n")
    development = tmp_path / "artifacts/development"
    development.mkdir(parents=True)
    (development / "metrics.csv").write_text("metrics\n")
    (development / "ranking.csv").write_text("ranking\n")
    selection = {
        "specification": specification(), "implementation_sha256": fingerprint(tmp_path),
        "data_sha256": sha256_file(data), "runtime_versions": runtime_versions(),
        "development_metrics_sha256": sha256_file(development / "metrics.csv"),
        "development_ranking_sha256": sha256_file(development / "ranking.csv"),
    }
    (tmp_path / "artifacts/selection.json").write_text(json.dumps(selection))
    return tmp_path


def test_uncommitted_selection_cannot_open_holdout(frozen_root):
    with pytest.raises(ValueError, match="Commit the modelling code"):
        frozen_selection(frozen_root)


@pytest.mark.parametrize("filename,error", [
    ("forecasting/config.py", "code/configuration changed"),
    ("data/studies.csv", "Input data changed"),
    ("artifacts/development/metrics.csv", "development results changed"),
])
def test_freeze_detects_changes(frozen_root, filename, error):
    path = frozen_root / filename
    path.write_text(path.read_text() + "changed\n")
    with pytest.raises(ValueError, match=error):
        frozen_selection(frozen_root)
