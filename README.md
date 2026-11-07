# Teleradiology forecasting — Part 1

Classical hourly forecasts for October 2026 using only the supplied synthetic
June–September studies. The forecast origin is simulated as **2026-10-01**,
regardless of the date on which this code is run. No LLM produces forecasts or
explanations in Part 1. AI coding assistance is disclosed in [AI_USE_LOG.md](AI_USE_LOG.md).

## Run the service

With Docker Desktop running, the single command is:

```bash
docker compose up --build
```

API documentation: <http://localhost:8000/docs>. Forecast endpoints:

- `GET /forecast/hourly`: point forecasts and p10/p90, all 744 hours by default.
- `GET /forecast/daily`: daily point totals and separately calibrated ranges.
- `GET /health`: readiness and artifact row counts.

Both forecast endpoints accept inclusive `start_date` and `end_date`, for example
`/forecast/hourly?start_date=2026-10-25&end_date=2026-10-31`. Invalid/reversed dates
return 422; dates extending outside October return 404. Hourly JSON timestamps
use ISO format without an offset; CSV timestamps use `YYYY-MM-DD HH:MM:SS`.
Responses explicitly identify the model, estimate definition, and unknown timezone.
Missing, modified, or invalid artifacts fail startup with a diagnostic rather
than silently training or serving inconsistent totals. The service trains nothing.

## Run the tests

The single Docker test command builds a separate test target:

```bash
docker compose run --build --rm tests
```

The test service is behind a Compose profile and is not launched by normal startup.
Locally, activate `main_env` and run `python -m pytest -q`.
Tests use synthetic fixtures, including small 8-iteration estimators for speed;
the real backtests and final forecast use the specified 200 iterations.

## Results and forecast artifacts

- [report.ipynb](report.ipynb): executed notebook, all backtests, and operations charts.
- [REPORT.md](REPORT.md): evidence-based limitations and concrete September failures.
- [forecast.csv](forecast.csv): exactly 744 October rows, fractional nonnegative
  `forecast`, `p10`, and `p90` with ordered bounds.
- `artifacts/daily_forecast.csv`: 31 daily totals with separately estimated bounds.
- `artifacts/development/`: three 28-day windows, all five candidates, predictions,
  metrics, training-day audits, and ranking.
- `artifacts/final/`: all September predictions/metrics and a provenance manifest.
- `artifacts/selection.json`: frozen specification, winning development candidate,
  implementation/data/dependency checksums, and selection rule.

The notebook is the presentation; no mobile UI or separate optional anomaly
detection deliverable is included. Part 2 remains a separate phase.

## Reproduce the experiment in order

On a clean checkout **before the September evaluation**, using the pinned local
environment and supplied `data/studies.csv`:

```bash
python -m forecasting develop
git add forecasting requirements.txt data artifacts/development artifacts/selection.json
git commit -m "Freeze Part 1 development selection before September evaluation"
python -m forecasting evaluate-final
python -m forecasting forecast
python -m forecasting report
```

The modelling code and selection must be committed before `evaluate-final`.
The final manifest records that pre-evaluation commit. The code also verifies
input data, dependency versions, implementation, and development-result hashes.
Once September has been evaluated, `develop` refuses to overwrite the selection;
`evaluate-final` preserves its existing result. Do not remove that guard to tune
against September. Use a separate clean copy of the pre-evaluation commit for a
full reproduction; any genuine implementation correction must be disclosed.
The saved completed checkout can regenerate October and the report without
changing model selection. Kernel `main_env` is needed to execute the notebook.

## Modelling choices and assumptions

The eight calendar features are hour, weekday, weekend, sine/cosine encodings of
hour and weekday, and days since June 1. All are known at forecast time. The
whole horizon is predicted directly; actual arrivals inside it never enter
features, training, or calibration. `day_total_studies` is a retrospective
consistency check only, never an input or a replacement target.

The seasonal-naive baseline repeats the last 168 observed hours. Histogram
gradient boosting uses Poisson loss for point predictions and three quantile
objectives (0.1/0.5/0.9) for direct uncertainty. Shared parameters: 200 iterations,
learning rate 0.05, 15 leaves, minimum 20 samples per leaf, L2=1, seed=42,
`early_stopping=False`. No parameter search or random temporal validation split.

Each learned family is trained raw and with days below 10% of the training
weekday median excluded, producing five candidates including baseline. High
days remain. Thresholds are learned separately at each origin. Validation
actuals are never filtered. Select the lowest mean development hourly WAPE,
then daily WAPE, then candidate-order simplicity; September cannot change it.

Origins July 7, July 21, and August 4 forecast 28 days each. September 1 forecasts
all 30 September days using only earlier data. WAPE and bias are fractions in
CSV files; total error is forecast minus actual in studies. Every method/window
reports full-window and last-seven-day metrics. Percentage metrics with zero
actual totals are undefined, not zero. The development windows overlap.

Quantile hourly ranges are learned directly, clipped nonnegative, and sorted to
repair crossings before evaluation; original crossing rates are reported. For
the baseline and point model, hourly residual percentiles are grouped by stored
hour. Daily ranges use percentiles of **whole-day residual sums**, added to the
daily point forecast. Residuals come from three disjoint prior weekly forecasts,
each trained only on its earlier history. Bounds are bracketed around point
estimates. Daily bounds are not sums of hourly quantiles. A sum of hourly medians
is not necessarily the daily median or mathematical expectation. Coverage and
the limited 21-day, seven-day-horizon calibration are explicitly evaluated.

Timestamps remain on the unspecified source clock. Every calendar day has 24
stored hours without timezone/DST conversion. Missing rows mean zero **recorded**
arrivals, not demonstrated zero demand. We do not invent regional holidays,
outage labels, October actuals, or future hospital metadata. See the notebook
and REPORT for measured weaknesses and unusual-day sensitivity.

No API keys or user-supplied environment variables are required. Docker sets
`PYTHONDONTWRITEBYTECODE=1`, `PYTHONUNBUFFERED=1`, `OMP_NUM_THREADS=1`, and
`OPENBLAS_NUM_THREADS=1`. The Linux image resolves its own dependencies from
the pinned runtime requirements and does not install the macOS lock file.

## Python environment

The local environment is `main_env`, using Python **3.13.18** on Apple Silicon
macOS. New interactive zsh terminals activate it automatically. VS Code's
workspace interpreter points to `/opt/anaconda3/envs/main_env/bin/python`.

From the repository directory, install the pinned dependencies:

```bash
conda activate main_env
python -m pip install --dry-run --only-binary=:all: -r requirements-dev.txt
python -m pip install --only-binary=:all: -r requirements-dev.txt
```

For scripts or shells that do not initialize Conda, use the absolute interpreter:

```bash
/opt/anaconda3/envs/main_env/bin/python -m pip install --only-binary=:all: -r requirements-dev.txt
```

- `requirements.txt` contains forecasting and service dependencies.
- `requirements-dev.txt` includes runtime dependencies plus pytest, charts, and
  notebook tools. SciPy and the other transitive dependencies install automatically.
- `requirements-macos.lock.txt` records all installed Python packages after
  verification. It is a Python 3.13/macOS snapshot, not a Linux Docker lock file.

The existing pip, setuptools, and wheel versions are retained. HTTPX will support
later LLM integration; setup does not require credentials or make paid API calls.

## Notebooks

Register the kernel inside the environment:

```bash
python -m ipykernel install --sys-prefix --name main_env --display-name "Python (main_env)"
```

In VS Code, select **Python (main_env)** as the notebook kernel. Python and
Jupyter extensions are already installed on the current machine.

## Verify the Python tools

```bash
python --version
python -c "import sys; print(sys.executable)"
python -m pip check
python scripts/verify_environment.py
python -m pytest --version
```

The smoke check verifies every direct package version and import, fits a small
point model and p10/p50/p90 models, checks a temporary FastAPI endpoint and invalid
inputs, renders a chart, and executes a notebook cell through `main_env`.
Temporary outputs are removed after the check. This is an environment check,
not the assignment's forecasting evaluation or endpoint test suite.

Refresh the macOS package snapshot after a deliberately verified dependency change:

```bash
python -m pip list --format=freeze > requirements-macos.lock.txt
```

## Docker Desktop

Docker Desktop is installed as a macOS application, outside the Python environment.
Download the Apple Silicon installer from the official installation page linked
below, mount it, and drag `Docker.app` into `/Applications`, then launch it:

```bash
open -a Docker
```

On this machine the Homebrew cask required administrator access to `/usr/local/bin`,
including with `--no-binaries`, so the downloaded application was installed directly.
CLI tools are linked in `~/.docker/bin`, Compose is also linked in
`~/.docker/cli-plugins`, and `.zshrc` adds the user CLI directory to `PATH`.
Docker Desktop supports installing its CLI tools in the user directory; retain
that option during setup.
Complete Docker Desktop's initial setup, wait until its engine is running, then check:

```bash
docker version
docker compose version
docker run --rm hello-world
```

Installation references: [Docker Desktop for Mac](https://docs.docker.com/desktop/setup/install/mac-install/)
and the [Homebrew Docker Desktop cask](https://formulae.brew.sh/cask/docker-desktop).
The service image and test target are defined in `Dockerfile` and `compose.yaml`.

## Verified installation

Verified on 2026-11-07:

- Python 3.13.18 in `main_env`; all 11 direct package imports and versions match.
- `pip check` reports no broken requirements; pytest 9.2.0 is available.
- Point and p10/p50/p90 models, API validation, chart rendering, and notebook
  execution pass `scripts/verify_environment.py`.
- The macOS snapshot matches all 122 installed Python packages, including the
  original pip, setuptools, and wheel versions.
- Docker Desktop 4.98.0, Docker Engine 29.9.2, and Compose v5.6.1 are available.
- `docker run --rm hello-world` successfully pulls and runs the ARM64 container.

Open a new terminal to pick up the Docker CLI PATH setting. Choose
**Python (main_env)** as the notebook kernel in VS Code.
