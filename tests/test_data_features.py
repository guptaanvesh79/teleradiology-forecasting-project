import numpy as np
import pandas as pd
import pytest

from forecasting.config import CANDIDATES, FEATURES
from forecasting.data import day_audit, hourly_counts, load_studies, source_consistency
from forecasting.features import calendar_features
from forecasting.models import training_mask


def test_hourly_grid_preserves_empty_boundaries(study_frame, tmp_path):
    path = tmp_path / "studies.csv"
    study_frame.to_csv(path, index=False)
    frame = load_studies(path)
    hourly = hourly_counts(frame)
    assert len(hourly) == 2928
    assert hourly.index[0] == pd.Timestamp("2026-06-01 00:00:00")
    assert hourly.index[-1] == pd.Timestamp("2026-09-30 23:00:00")
    assert hourly.iloc[0] == 0 and hourly.iloc[1] == 2 and hourly.iloc[-1] == 0
    assert hourly.sum() == 3
    assert source_consistency(frame) == {
        "days_with_inconsistent_source_totals": 0, "days_source_total_differs_from_rows": 0,
    }


@pytest.mark.parametrize("problem", ["missing_column", "duplicate", "timestamp", "out_of_range", "missing_value", "negative_total", "fractional_total"])
def test_invalid_studies_fail(study_frame, tmp_path, problem):
    frame = study_frame.copy()
    if problem == "missing_column":
        frame = frame.drop(columns="modality")
    elif problem == "duplicate":
        frame.loc[1, "study_id"] = frame.loc[0, "study_id"]
    elif problem == "timestamp":
        frame.loc[0, "arrival_time"] = "not-a-date"
    elif problem == "out_of_range":
        frame.loc[0, "arrival_time"] = "2026-10-01 00:00:00"
    elif problem == "missing_value":
        frame.loc[0, "hospital_id"] = None
    else:
        frame["day_total_studies"] = frame.day_total_studies.astype(float)
        frame.loc[0, "day_total_studies"] = -1 if problem == "negative_total" else 1.5
    path = tmp_path / "bad.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_studies(path)


def test_calendar_features_are_known_future_values():
    times = pd.DatetimeIndex(["2026-06-01 00:00:00", "2026-06-01 23:00:00", "2026-10-03 06:00:00"])
    features = calendar_features(times)
    assert list(features.columns) == list(FEATURES)
    assert features.hour.tolist() == [0, 23, 6]
    assert features.day_of_week.tolist() == [0, 0, 5]
    assert features.is_weekend.tolist() == [0, 0, 1]
    assert features.days_since_start.tolist() == [0, 0, 124]
    assert np.allclose(features.hour_sin ** 2 + features.hour_cos ** 2, 1)
    assert np.allclose(features.weekday_sin ** 2 + features.weekday_cos ** 2, 1)
    assert features.iloc[2].hour_sin == pytest.approx(1)
    with pytest.raises(ValueError):
        calendar_features(pd.date_range("2026-10-01", periods=1, tz="UTC"))


def test_sparse_threshold_and_mask_use_training_days(series):
    history = series.loc[:"2026-07-20 23:00:00"].copy()
    history.loc["2026-07-08"] = 0
    history.loc["2026-07-09"] = 1
    audit = day_audit(history)
    assert audit.loc["2026-07-08", "near_empty"]
    assert audit.loc["2026-07-09", "near_empty"]
    assert audit.loc["2026-07-08", "sparse_threshold"] == pytest.approx(
        0.1 * history.resample("D").sum().groupby(lambda date: date.day_of_week).median().loc[2]
    )
    filtered = training_mask(history, CANDIDATES[2])
    assert not filtered.loc["2026-07-08"].any()
    assert filtered.loc["2026-07-07"].all()
    assert training_mask(history, CANDIDATES[1]).all()


def test_zero_weekday_medians_do_not_exclude_everything(series):
    history = series.iloc[:24 * 35] * 0
    assert not day_audit(history).near_empty.any()
    assert training_mask(history, CANDIDATES[2]).all()
