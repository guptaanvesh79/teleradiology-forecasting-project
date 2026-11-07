import json

from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest

from forecasting.api import create_app
from forecasting.artifacts import validate_daily, validate_hourly
from forecasting.pipeline import export_forecast


def test_submission_and_daily_artifacts(artifact_root):
    hourly = validate_hourly(pd.read_csv(artifact_root / "forecast.csv"))
    daily = validate_daily(pd.read_csv(artifact_root / "artifacts/daily_forecast.csv"), hourly)
    assert len(hourly) == 744 and len(daily) == 31
    assert hourly.index[0] == pd.Timestamp("2026-10-01")
    assert hourly.index[-1] == pd.Timestamp("2026-10-31 23:00:00")


@pytest.mark.parametrize("problem", ["missing_hour", "duplicate_hour", "wrong_format", "negative", "nan", "crossing", "extra_column"])
def test_invalid_exports_rejected(artifact_root, problem):
    frame = pd.read_csv(artifact_root / "forecast.csv")
    if problem == "missing_hour":
        frame = frame.iloc[:-1]
    elif problem == "duplicate_hour":
        frame.loc[1, "hour_start"] = frame.loc[0, "hour_start"]
    elif problem == "wrong_format":
        frame.loc[0, "hour_start"] = "2026-10-01T00:00:00Z"
    elif problem == "negative":
        frame.loc[0, "p10"] = -1
    elif problem == "nan":
        frame.loc[0, "forecast"] = np.nan
    elif problem == "crossing":
        frame.loc[0, "p90"] = 1
    else:
        frame["extra"] = 1
    with pytest.raises(ValueError):
        validate_hourly(frame)


def test_export_round_trip_preserves_daily_sums(artifact_root):
    hourly = validate_hourly(pd.read_csv(artifact_root / "forecast.csv")) / 7
    daily = hourly.resample("D").sum().rename_axis("date")
    saved, saved_daily = export_forecast(hourly, daily, artifact_root)
    assert np.allclose(saved_daily.forecast, saved.forecast.resample("D").sum(), rtol=0, atol=1e-9)


def test_endpoints_match_saved_artifacts(artifact_root):
    with TestClient(create_app(artifact_root)) as client:
        assert client.get("/health").json() == {"status": "ready", "hours": 744, "days": 31}
        hourly = client.get("/forecast/hourly")
        daily = client.get("/forecast/daily")
        assert hourly.status_code == daily.status_code == 200
        assert len(hourly.json()["data"]) == 744 and len(daily.json()["data"]) == 31
        assert hourly.json()["metadata"]["timezone"] == "unspecified"
        assert sum(row["forecast"] for row in hourly.json()["data"]) == sum(row["forecast"] for row in daily.json()["data"])
        for endpoint, count in (("hourly", 24), ("daily", 1)):
            response = client.get(f"/forecast/{endpoint}?start_date=2026-10-31&end_date=2026-10-31")
            assert response.status_code == 200 and len(response.json()["data"]) == count
        assert len(client.get("/forecast/daily?end_date=2026-10-02").json()["data"]) == 2


@pytest.mark.parametrize("endpoint", ["hourly", "daily"])
@pytest.mark.parametrize("query,status", [
    ("start_date=bad", 422), ("start_date=2026-10-32", 422),
    ("start_date=2026-10-20&end_date=2026-10-10", 422),
    ("start_date=2026-09-30", 404), ("end_date=2026-11-01", 404),
])
def test_api_date_errors(artifact_root, endpoint, query, status):
    with TestClient(create_app(artifact_root)) as client:
        assert client.get(f"/forecast/{endpoint}?{query}").status_code == status


def test_missing_and_modified_artifacts_fail_startup(tmp_path, artifact_root):
    with pytest.raises(RuntimeError, match="missing or invalid"):
        with TestClient(create_app(tmp_path / "missing")):
            pass
    frame = pd.read_csv(artifact_root / "forecast.csv")
    frame.loc[0, "p10"] = 0.5
    frame.to_csv(artifact_root / "forecast.csv", index=False)
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        with TestClient(create_app(artifact_root)):
            pass


def test_daily_bounds_are_independent_and_totals_are_checked(artifact_root):
    hourly = validate_hourly(pd.read_csv(artifact_root / "forecast.csv"))
    daily = pd.read_csv(artifact_root / "artifacts/daily_forecast.csv")
    daily["p10"] = 60.0
    daily["p90"] = 80.0
    validate_daily(daily, hourly)
    daily.loc[0, "forecast"] += 1
    with pytest.raises(ValueError, match="must equal sums"):
        validate_daily(daily, hourly)
