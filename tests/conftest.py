import json

import numpy as np
import pandas as pd
import pytest

from forecasting.data import sha256_file


@pytest.fixture
def study_frame():
    return pd.DataFrame({
        "study_id": ["S1", "S2", "S3"],
        "arrival_time": ["2026-06-01 01:10:00", "2026-06-01 01:20:00", "2026-06-02 23:59:00"],
        "hospital_id": ["H01"] * 3, "modality": ["CT"] * 3,
        "specialty_required": ["GENERAL"] * 3, "criticality": ["ROUTINE"] * 3,
        "patient_location": ["ED"] * 3, "day_total_studies": [2, 2, 1],
    })


@pytest.fixture
def series():
    index = pd.date_range("2026-06-01", "2026-09-01", freq="h", inclusive="left", name="hour_start")
    counts = 10 + index.hour.to_numpy() * 2 + (index.day_of_week.to_numpy() < 5) * 7
    counts = np.where((index.hour == 0) | (index.hour >= 18), 0, counts)
    return pd.Series(counts, index=index, name="actual")


@pytest.fixture
def fast_models(monkeypatch):
    # Full production backtests use 200 iterations; synthetic unit tests need only 8.
    from forecasting.models import PARAMETERS
    monkeypatch.setitem(PARAMETERS, "max_iter", 8)


@pytest.fixture
def artifact_root(tmp_path):
    index = pd.date_range("2026-10-01", "2026-11-01", freq="h", inclusive="left", name="hour_start")
    hourly = pd.DataFrame({"forecast": 3.0, "p10": 1.0, "p90": 5.0}, index=index)
    hourly.to_csv(tmp_path / "forecast.csv", date_format="%Y-%m-%d %H:%M:%S")
    output = tmp_path / "artifacts"
    output.mkdir()
    daily = hourly.resample("D").sum().rename_axis("date")
    daily.to_csv(output / "daily_forecast.csv", date_format="%Y-%m-%d")
    metadata = {
        "model": "seasonal_naive", "forecast_origin": "2026-10-01", "timezone": "unspecified",
        "point_estimate": "hourly_point_forecast", "daily_point_estimate": "sum_of_hourly_point_forecasts",
        "hourly_interval": "calibration residuals", "daily_interval": "whole-day residuals",
        "forecast_csv_sha256": sha256_file(tmp_path / "forecast.csv"),
        "daily_csv_sha256": sha256_file(output / "daily_forecast.csv"),
    }
    (output / "forecast_metadata.json").write_text(json.dumps(metadata))
    return tmp_path
