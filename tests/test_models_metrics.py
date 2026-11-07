import numpy as np
import pandas as pd
import pytest

from forecasting.config import CANDIDATES
from forecasting.metrics import evaluate, pinball, summarize
from forecasting.models import calibration_residuals, fit_predict, ordered_band, run_origin
from forecasting.pipeline import rank_candidates


def test_naive_repeats_seven_days_without_shifting(series):
    history = series.iloc[:24 * 35]
    future = pd.date_range(history.index[-1] + pd.Timedelta(hours=1), periods=744, freq="h")
    prediction = fit_predict(history, future, CANDIDATES[0])
    expected = np.resize(history.iloc[-168:].to_numpy(), 744)
    assert np.array_equal(prediction.forecast.to_numpy(), expected)
    assert prediction.forecast.iloc[168] == history.iloc[-168]


@pytest.mark.parametrize("candidate", [CANDIDATES[1], CANDIDATES[4]])
def test_later_observations_cannot_change_predictions(series, fast_models, candidate):
    origin = pd.Timestamp("2026-07-07")
    changed = series.copy()
    changed.loc[changed.index >= origin] = 100000
    first, first_daily = run_origin(series, origin, 28, candidate)
    second, second_daily = run_origin(changed, origin, 28, candidate)
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(first_daily, second_daily)
    assert (first.p10 <= first.forecast).all() and (first.forecast <= first.p90).all()
    assert (first[["forecast", "p10", "p90"]] >= 0).all().all()
    assert np.allclose(first_daily.forecast, first.forecast.resample("D").sum())


def test_calibration_is_three_prior_weeks(series):
    history = series.loc[:"2026-07-06 23:00:00"]
    residuals = calibration_residuals(history, CANDIDATES[0])
    assert len(residuals) == 504
    assert residuals.index.min() == pd.Timestamp("2026-06-16")
    assert residuals.index.max() == history.index.max()
    assert residuals.index.is_unique and not residuals.isna().any()


def test_future_must_begin_after_history(series):
    with pytest.raises(ValueError, match="Future"):
        fit_predict(series.iloc[:24 * 35], series.index[:24], CANDIDATES[0])


def test_zero_count_training_and_band_order(series, fast_models):
    history = series.iloc[:24 * 35] * 0
    future = pd.date_range(history.index[-1] + pd.Timedelta(hours=1), periods=24, freq="h")
    assert fit_predict(history, future, CANDIDATES[1]).forecast.eq(0).all()
    low, high = ordered_band(np.array([0, 5]), np.array([-2, 7]), np.array([-1, 3]))
    assert np.array_equal(low, [0, 5]) and np.array_equal(high, [0, 5])


def test_metrics_and_last_week():
    index = pd.date_range("2026-07-07", periods=240, freq="h")
    actual = pd.Series(2.0, index=index)
    prediction = pd.DataFrame({"forecast": 3.0, "p10": 1.0, "p90": 4.0, "crossed_before_sort": False}, index=index)
    daily = prediction[["forecast", "p10", "p90"]].resample("D").sum()
    scores = evaluate(actual, prediction, daily, True)
    assert scores["all"]["hourly_wape"] == 0.5
    assert scores["all"]["daily_wape"] == 0.5
    assert scores["all"]["bias"] == 0.5
    assert scores["all"]["total_error"] == 240
    assert scores["all"]["coverage"] == 1
    assert scores["all"]["pinball_10"] == pytest.approx(0.1)
    assert scores["last_7_days"]["hours"] == 168
    assert scores["last_7_days"]["actual_total"] == 336
    empty = summarize(actual * 0, prediction, daily)
    assert empty["hourly_wape"] is None and empty["bias"] is None
    assert pinball([10], [12], 0.9) == pytest.approx(0.2)


def test_candidate_selection_ignores_last_week_rows():
    records = []
    for origin in ("2026-07-07", "2026-07-21", "2026-08-04"):
        for name in ("seasonal_naive", "poisson_raw"):
            records.append(dict(origin=origin, candidate=name, scope="all", hourly_wape=0.2, daily_wape=0.1))
            records.append(dict(origin=origin, candidate=name, scope="last_7_days", hourly_wape=0.99, daily_wape=0.99))
    assert rank_candidates(pd.DataFrame(records)).iloc[0].candidate == "seasonal_naive"
