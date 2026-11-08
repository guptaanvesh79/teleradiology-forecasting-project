# Teleradiology forecasting — Part 1

Classical hourly forecasts for October 2026 using only the supplied synthetic
June–September data. The simulated forecast origin is **2026-10-01**, regardless
of the current date. No LLM produces forecasts or explanations in Part 1.
AI coding assistance is disclosed in [AI_USE_LOG.md](AI_USE_LOG.md).

## Run the service

With Docker Desktop running:

```bash
docker compose up --build
```

API documentation: <http://localhost:8000/docs>.

- `GET /forecast/hourly`: all 744 hourly point forecasts and p10/p90 ranges.
- `GET /forecast/daily`: 31 daily point totals and separately calibrated ranges.
- `GET /health`: readiness and artifact row counts.

Both forecast endpoints accept inclusive `start_date` and `end_date`, for example
`/forecast/hourly?start_date=2026-10-25&end_date=2026-10-31`. Malformed/reversed dates
return 422; ranges extending outside October return 404. Response metadata names
the model, estimate definition, and unspecified timezone. Missing, modified, or
invalid artifacts fail startup rather than silently serving inconsistent data.
The service reads saved artifacts and does not train during requests.

## Run the tests

The single Docker test command builds an isolated test target and exits with
pytest's result:

```bash
docker compose run --build --rm tests
```

The test service uses a Compose profile, so `docker compose up --build` starts
only the API. The service image excludes pytest and notebook tooling; the test
image adds pytest and the permanent `tests/` folder.

For local testing after installing the development requirements:

```bash
python -B -m pytest -q -p no:cacheprovider tests
```

Tests cover all eight features, input validation, hourly aggregation, sparse-day
handling, post-origin leakage, saved forecast contracts, and both API endpoints.
They use synthetic fixtures and pytest-managed temporary files without external
services or LLM calls. Small model fits use eight iterations within isolated
tests; production remains at 200. Submitted forecast files are read-only.

## Project structure

- `forecasting/`: calendar features, models, metrics, pipeline, validation, API,
  and notebook generator.
- `tests/`: shared synthetic fixtures plus feature/data and API/forecast-contract tests.
- `data/`: unchanged source CSV, supplied dictionary, and checksums.
- `artifacts/development/`: all three development backtests, metrics, and ranking.
- `artifacts/final/`: September results, manifest, and training audits.
- `artifacts/selection.json`: frozen model selection, specification, and hashes.
- `artifacts/daily_forecast.csv` and `artifacts/forecast_metadata.json`: service data.
- [forecast.csv](forecast.csv): exactly 744 October rows in the required format.
- [report.ipynb](report.ipynb): executed report, limitations, and embedded charts.

The notebook is the report and presentation. It embeds charts directly without
creating a separate Markdown report or image folder. Optional helper scripts,
environment snapshots, editor configuration, and caches are not part of this layout.

## Local environment and pipeline

Use Python **3.13.18** and the existing `main_env` environment:

```bash
conda activate main_env
python -m pip install --only-binary=:all: -r requirements-dev.txt
python -m pip check
```

Runtime/service dependencies are in `requirements.txt`; `requirements-dev.txt`
adds pytest, Matplotlib, JupyterLab, and ipykernel. The service image resolves its own Linux
dependencies and excludes notebook tooling. No API keys or user-supplied environment
variables are required. Docker sets `PYTHONDONTWRITEBYTECODE`, `PYTHONUNBUFFERED`,
`OMP_NUM_THREADS`, and `OPENBLAS_NUM_THREADS` to 1.

The completed checkout can regenerate October artifacts and the notebook:

```bash
python -m forecasting forecast
python -m forecasting report
```

The notebook generator uses kernel `main_env`. If needed, register it with:

```bash
python -m ipykernel install --sys-prefix --name main_env --display-name "Python (main_env)"
```

Development originally ran with `python -m forecasting develop`; code, selection,
and development results were committed at `a929ab4` before the first
`python -m forecasting evaluate-final`. The final manifest records that commit.
Reproduce original development from a separate checkout of the pre-evaluation
commit. In this completed checkout, `develop` refuses to overwrite the selection
and `evaluate-final` preserves the existing September result. Do not remove those
guards to tune on September.

The freeze validates the selected settings, data, dependencies, development
results, and the original pre-evaluation Git snapshot. Saved source hashes refer
to that archived snapshot, not later working-tree formatting or layout edits.
Later source changes are tracked through Git. No separate maintenance record is
required. Original selection and evaluation files are unchanged.

## Method and assumptions

The eight calendar features are hour, weekday, weekend, sine/cosine encodings of
hour and weekday, and days since June 1. All are available at forecast time.
The whole horizon is predicted directly without actual counts inside the window.
`day_total_studies` is only a retrospective consistency check, not a feature or
replacement target. Timestamps stay on the unspecified source clock, without
invented timezone/DST corrections or regional holidays.

The baseline repeats the final 168 observed hours. Histogram gradient boosting
uses Poisson loss for point predictions and three quantile objectives (0.1/0.5/0.9).
Shared parameters: 200 iterations, learning rate 0.05, 15 leaves, minimum 20 samples
per leaf, L2=1, seed=42, and `early_stopping=False`. No parameter search or random
validation split is used.

Histogram gradient boosting keeps this small calendar-feature problem in one
CPU-based library and learns nonlinear hour/weekday interactions without manual
feature scaling. The same estimator supports count-oriented Poisson loss and
direct quantile loss. Quantile regression allows asymmetric, time-varying hourly
ranges rather than imposing a fixed band around a point forecast.

Each learned family is trained raw and with days below 10% of their training
weekday median excluded, producing five candidates including baseline. Thresholds
use only each origin's history. Validation actuals are never removed. Missing rows
represent zero recorded arrivals, not established zero underlying demand.

Origins July 7, July 21, and August 4 forecast 28 days each. September 1 forecasts
all 30 September days from earlier history. Selection minimizes mean development
hourly WAPE, then daily WAPE, then simplicity. September cannot change selection.
WAPE and bias are fractions in result CSVs; total error is forecast minus actual.
Full-window and last-seven-day metrics are reported, including interval coverage.
Percentage metrics with zero actual totals are undefined.

Quantile hourly ranges are learned directly, clipped nonnegative, and sorted to
repair crossings before scoring. Baseline/point hourly ranges use residual
percentiles grouped by stored hour. Daily ranges use percentiles of whole-day
residual sums from three prior weekly forecasts, each trained only on earlier
history. They are not sums of hourly quantiles. A sum of hourly medians is not
necessarily the daily median or expectation. The notebook documents calibration,
missing-data, trend-extrapolation, and long-horizon limitations.

## Selected model and results

The frozen selection is **quantile_filtered**, with mean development hourly WAPE
**19.95%**, versus **24.11%** for seasonal naive.

| September metric | Selected quantile model | Seasonal naive |
|---|---:|---:|
| Hourly WAPE | 15.85% | 19.16% |
| Daily-total WAPE | 13.12% | 15.44% |
| Hourly p10–p90 coverage | 91.25% | 86.25% |

Selected-model final-week hourly WAPE is **15.32%** and hourly coverage **92.86%**.
Full-month daily coverage is **76.67%**; ranges do not guarantee 80% coverage.
Raw quantile training scores slightly better on September (15.71% hourly WAPE),
but the previously frozen selection is retained. October's point total is
**43,974.8** recorded studies. October actuals are unavailable.

The permanent pytest suite and Docker test target are included in this checkout.
The earlier 45-test results belong to the original Part 1 snapshot; verification
of the current suite is recorded below. Part 2 remains a separate phase.

## Current Part 1 verification

- Local permanent suite: **58 passed**.
- Linux Docker permanent suite: **58 passed**.
- Notebook: five executed code cells, four embedded charts, no execution errors.
- Original selection, September predictions/metrics, and October forecast files
  remain unchanged. Hourly coverage remains the measured 91.25%, not an asserted
  80% guarantee.

The installed FastAPI/Starlette release emits an HTTPX TestClient deprecation
warning; tests pass without suppressing it. The sandboxed macOS run also reports
a physical-core detection warning and uses logical-core fallback. Neither warning
changes the submitted forecasts.
