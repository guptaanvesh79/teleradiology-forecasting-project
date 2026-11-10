# Teleradiology forecasting — classical and local LLM flows

Classical hourly forecasts for October 2026 using only the supplied synthetic
June–September data. The simulated forecast origin is **2026-10-01**, regardless
of the current date. No LLM produces forecasts or explanations in Part 1.
AI coding assistance is disclosed in [AI_USE_LOG.md](AI_USE_LOG.md).

## Run the service

On macOS, start Docker only for an API session:

```bash
python docker_tasks.py up
```

The helper starts Docker Desktop if it is closed, waits up to 120 seconds for
the engine, builds the API, and displays its logs. Press Ctrl+C to stop the API.
It then closes Docker Desktop if it started Desktop and no other containers
are running. A previously running Desktop stays open. If Desktop is paused or
its initial status is uncertain, resume/check it manually before retrying.
Training, notebooks, local tests, and hosted LLM calls do not need Docker.
Automatic startup at login can remain disabled.

The helper uses only Python's standard library. From another directory, pass
its full path; it always uses this project's Compose file and directory.

The direct assignment command also works with Docker Desktop already running:

```bash
docker compose up --build
```

API documentation: <http://localhost:8000/docs>.

- `GET /forecast/hourly`: all 744 hourly point forecasts and p10/p90 ranges.
- `GET /forecast/daily`: 31 daily point totals and separately calibrated ranges.
- `GET /health`: readiness and artifact row counts.
- `GET /explain`: a checked LLM explanation of Part 1 numbers, or a fixed-template
  fallback if the provider or response is invalid. It accepts the same dates.

Both forecast endpoints accept inclusive `start_date` and `end_date`, for example
`/forecast/hourly?start_date=2026-10-25&end_date=2026-10-31`. Malformed/reversed dates
return 422; ranges extending outside October return 404. Response metadata names
the model, estimate definition, and unspecified timezone. Missing, modified, or
invalid artifacts fail startup rather than silently serving inconsistent data.
The service reads saved artifacts and does not train during requests.

## Run the tests

To start Docker only for the container test run:

```bash
python docker_tasks.py test
```

This has the same startup/shutdown rules as the API helper and returns the
container test exit code. Ctrl+C also cleans up the helper's test container.
Cleanup problems are reported without replacing the task's exit code. If another
container prevents Desktop shutdown, stop Docker manually when finished:
`docker desktop stop`. Images, volumes, and submitted forecasts are preserved.

The single Docker test command builds an isolated test target and exits with
pytest's result:

```bash
docker compose run --build --rm tests
```

The test service uses a Compose profile, so `docker compose up --build` starts
only the API. The service image excludes pytest and notebook tooling; the test
image adds pytest, Git for isolated protocol-freeze tests, nbformat for report-cell
checks, and the permanent `tests/` folder. Tests never call Ollama or hosted APIs.

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
- `llm_forecasting/`: separate LLM CLI, provider adapters, prompts and schemas,
  experiment stages, explanations, and saved-result report sections.
- `tests/`: shared synthetic fixtures, Part 1 tests, and two Part 2 test modules.
- `docker_tasks.py`: macOS Docker startup and cleanup for API sessions and tests.
- `data/`: unchanged source CSV, supplied dictionary, and checksums.
- `artifacts/development/`: all three development backtests, metrics, and ranking.
- `artifacts/final/`: September results, manifest, and training audits.
- `artifacts/selection.json`: frozen model selection, specification, and hashes.
- `artifacts/daily_forecast.csv` and `artifacts/forecast_metadata.json`: service data.
- [forecast.csv](forecast.csv): exactly 744 October rows in the required format.
- [report.ipynb](report.ipynb): executed report, limitations, and embedded charts.
- `artifacts/llm/`: created only when you manually run the LLM stages; contains
  the frozen protocol, request evidence, results and demonstrations.
- `llm_forecast.csv`: created only after the manual October LLM stage succeeds.

The notebook is the report and presentation. It embeds charts directly without
creating a separate Markdown report or image folder. Apart from the Docker task
helper, optional verification scripts, environment snapshots, editor configuration,
and caches are not part of this layout.

## Local environment and pipeline

Use Python **3.13.18** and the existing `main_env` environment:

```bash
conda activate main_env
python -m pip install --only-binary=:all: -r requirements-dev.txt
python -m pip check
```

Runtime/service dependencies are in `requirements.txt`; `requirements-dev.txt`
adds pytest, Matplotlib, JupyterLab, and ipykernel. The service image resolves its own Linux
dependencies and excludes notebook tooling. Part 1 and the local Ollama defaults
require no API key; optional provider variables are documented below. Docker sets
`PYTHONDONTWRITEBYTECODE`, `PYTHONUNBUFFERED`,
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

## Preserved Part 1 verification

- Local Part 1 test subset: **58 passed**.
- Linux Docker Part 1 test subset: **58 passed**.
- Notebook: five executed code cells, four embedded charts, no execution errors.
- Original selection, September predictions/metrics, and October forecast files
  remain unchanged. Hourly coverage remains the measured 91.25%, not an asserted
  80% guarantee.

The installed FastAPI/Starlette release emits an HTTPX TestClient deprecation
warning; tests pass without suppressing it. The sandboxed macOS run also reports
a physical-core detection warning and uses logical-core fallback. Neither warning
changes the submitted forecasts.

## Local model runtime for Part 2

Ollama **0.40.15** is installed natively through Homebrew, outside `main_env`.
The installed model is **`qwen3.5:4b-q4_K_M`**, approximately 3.32 GB on disk.
Keep this explicit tag: Homebrew's default MLX variant failed JSON-schema
requests because its grammar library was missing; the Q4_K_M backend passed
the numeric structured-response test. No Python dependencies were changed.

Start the local server only when needed, with Docker closed:

```bash
OLLAMA_HOST=127.0.0.1:11434 \
OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=1 \
OLLAMA_NO_CLOUD=1 \
OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 ollama serve
```

In another terminal, check the installation:

```bash
ollama list
ollama ps
```

If reproducing the installation on another Mac:

```bash
brew install ollama
# Start the server using the command above, then download the model:
ollama pull qwen3.5:4b-q4_K_M
```

The server listens only on localhost. Cloud inference is disabled for this
session. No login service was installed; Ctrl+C stops the server. API clients
should send `keep_alive: 0` as well to release model memory after each request.

A three-hour synthetic installation test validated numeric JSON, finite
nonnegative ordered quantiles, and model unloading. Ollama reported about
3.3 GB allocated at an 8K context. This is a small runtime check, not evidence of
month-long forecasting accuracy; a separate timestamp-contract check failed.
Part 2's implementation is now included, but real assignment experiments and
their combined executed notebook remain pending manual execution. Installation
checks do not establish forecasting accuracy. The retry policy is one initial
call plus three retries per request.

## Part 2 configuration

The fixed environment variable names are `LLM_PROVIDER`, `LLM_BASE_URL`,
`LLM_MODEL`, and `LLM_API_KEY`. Defaults select local Ollama and the installed
Q4_K_M model; there is no local API key. Existing Python packages are reused.

An optional `.env.example` documents the names. Python reads the process
environment, not `.env` automatically. If using a trusted local `.env` file:

```bash
cp .env.example .env
# Edit .env locally, then export its values in this terminal:
set -a
source .env
set +a
```

`.env` is excluded from Git and the Docker build context. Never commit keys.
OpenAI and Gemini adapters use HTTPX with the same names; configure their model,
API key and HTTPS API base URL together. Hosted adapters have mocked tests only;
the assignment CLI deliberately accepts only the frozen local Qwen setup.

For native Python, the default base URL is `http://127.0.0.1:11434`. Compose
defaults to `http://host.docker.internal:11434` for a host Ollama server. Leave
`LLM_BASE_URL` unset when switching between those defaults; container localhost
is not your Mac. A provider connection failure returns the explanation template.
No network/listener settings are automatically changed, and no Ollama image or
model weights are included in the service image.

## Part 2 manual execution sequence

Keep the native Ollama server running in its own terminal using the command
above, and Docker closed during the experiment stages. In another terminal:

```bash
conda activate main_env
python -m llm_forecasting develop
```

This runs one 28-day forecast at each Part 1 development origin: July 7, July 21,
and August 4. Each initial call predicts twelve hours: 00:00–11:00 or
12:00–23:00. Retries request only still-invalid or missing hours. Review output
validity, context use, latency and errors here. The
code saves `artifacts/llm/protocol.json`, which fixes the data, modelling source,
model digest, Ollama/Python versions, prompt, response schema, twelve-hour chunk
size, request budgets, generation settings,
development results and their request-evidence prefix.

Before September, manually commit the exact implementation and development:

```bash
git add llm_forecasting tests forecasting/api.py forecasting/reporting.py Dockerfile compose.yaml .env.example README.md AI_USE_LOG.md docker_tasks.py artifacts/llm
git commit -m "Freeze Part 2 protocol and development before September"
python -m llm_forecasting evaluate-final
python -m llm_forecasting forecast
python -m llm_forecasting explain-demo
python -m llm_forecasting report
```

`evaluate-final` runs September three times with identical prompts and settings
for corresponding chunks. The first evaluation attempt freezes development,
even if a call fails. Do not change prompts, source, versions or model settings
after seeing September results. `forecast` then generates all 744 October hours
using history before October; it does not choose a best September run or blend
its results. Existing completed stages are preserved without more inference.

`explain-demo` saves a genuine validated Qwen explanation of Part 1 numbers and
a clearly labelled simulated-provider-failure template. It reports failure if
the real explanation still falls back. `report` reads saved evidence only,
writes the comparison table, and executes the existing combined `report.ipynb`.
Until these stages run successfully, the notebook remains the original Part 1
submission and Part 2 is not complete. Commit the completed results afterward.

Valid hourly predictions are cached in the append-only `requests.jsonl` journal,
including valid neighbours in a partly invalid response. Each retry rebuilds its
prompt and schema for only the remaining hour keys. The historical summary,
date, weekday, original calendar context and generation settings stay unchanged;
previous predictions are never included in retry context. Accepted values are
retained from their first valid response and are never overwritten.

Rerunning an interrupted stage reconstructs that hourly cache from validated
raw evidence, checks each saved subset request and retries only unfinished
hours, preserving every previous attempt. After a chunk exhausts its
retries on invalid replies or transient request failures, the stage continues
through later chunks and all remaining origins or repeat runs. The final summary
lists every missing chunk and returns a nonzero exit status if any remain.
Rerun the same command to request only missing hours; there is no automatic
extra retry pass. Each chunk invocation allows at most four calls total, not
four per failed hour, with 1/2/4-second delays and a 120-second forecast read
timeout. This bounded loop is equivalent to recursively requesting the
remaining set, without an unbounded recursion chain. There are 410 successful forecast chunks when no
retries are needed: 168 development calls, 180 September calls and 62 October
calls. The fixed attempt budget is derived as 410 × 4 = 1,640; explanation calls
are separate. Morning and afternoon cache keys include their full start
timestamps. Do not delete evidence to hide failures or bypass the
freeze. A fresh development trial can archive incompatible failed-only attempts
automatically, as described below.

An incomplete stage publishes no forecast CSVs, metrics or repeatability
results. Development writes its frozen protocol only after every development
chunk succeeds. September keeps its pending freeze marker until all three runs
are complete; October exports only after all 62 chunks validate. Request
evidence is retained throughout. Incompatible evidence, exhausted global request
budgets, invalid context allocations and permanent HTTP configuration errors
still stop immediately instead of proceeding through the stage.

Protocol version 4 freezes the labelled-hour response contract, its
hr1–hr24 mapping, generic schema template, exact-key schema-generation policy
and first-valid-hour recovery rules. Each request journal entry retains its
specific one-to-twelve-key schema and per-hour validation results. Older protocols
or journals are rejected and preserved; their results cannot be mixed into this
experiment. No automatic deletion or migration is performed.

If code or settings change after failed development calls, rerun:

```bash
python -m llm_forecasting develop
```

When the journal belongs to one earlier experiment and contains only recognized
failed development attempts, `develop` preserves its bytes under
`artifacts/llm/failed_development/<unique-id>/requests.jsonl`, prints that path,
creates a fresh active journal and continues. It does this only if there is no
frozen protocol, September evaluation, October submission or saved development
results. Successful chunks, mixed experiments, unknown statuses, damaged
journals and legacy-incompatible records still require manual inspection.
Trials retaining any valid hours are also protected from automatic archiving,
even if no complete chunk has succeeded yet. Unchanged experiments keep their
existing journal and resume normally.

The previous positional trial and its archives were deleted once, at the user's
request, before starting the labelled trial. Part 1 files were preserved.
Positional responses are not converted, reused or carried through an execution
compatibility exception. This reset is not part of any normal stage command;
future runs retain their evidence and resume matching successful chunks.

Archived attempts remain evidence of the earlier trial; they are not relabelled
as valid responses or mixed into the revised protocol's statistics. A stable
`.pipeline.lock` file keeps concurrent stages excluded while the journal moves;
this runtime lock is ignored by Git. Archive failures stop before new forecast
requests. Normal stage commands do not delete or edit old responses.

The client currently requests a 16K context; its exact value is saved in the
frozen protocol. Context size and immediate unloading are sent by the Python
client with each request, so startup environment overrides are unnecessary.
The other frozen settings are temperature 0.2, thinking disabled, at most
8,192 output tokens, no fixed sampling seed, sequential requests and model
unloading after each call. The input-token budget is context length minus the
actual output allowance for that request: currently 16,384 − 8,192 = 8,192 tokens
for forecasts. An explicit smaller output allowance reserves less context for
output. Oversized inputs and truncated responses are rejected; investigate
these during development. Exhausted retries now display the final underlying
error as well as preserving every attempt in the journal.
API charges for these local runs are US$0, excluding electricity and hardware.
The configured output ceiling is an upper bound; the required response is much
smaller. Smaller chunks may reduce truncation or formatting failures; their
forecast accuracy still requires measurement. More calls and repeated model
loading increase execution time.

## Part 2 output and reliability definitions

Prompts contain only pre-origin weekday/hour profiles, recent daily totals,
weekday daily summaries and empty/sparse-day diagnostics. All recorded zeros
and unusual days are retained. Future calendar dates are known; study-level
fields and `day_total_studies` are not predictive inputs. Later chunks never
receive actual counts or earlier forecasts from inside the horizon.

Qwen returns a strict JSON object with exactly two fields: `chunk_total` and
`hours`. The total is the model-stated p50 sum for only the hours requested in
that call. `hours` is an object with the requested keys, each containing a
`[p10,p50,p90]` array. Initial requests contain twelve keys (36 hourly numeric
values); retries contain only remaining keys. Morning chunks use `hr1`–`hr12`, corresponding
to 00:00–11:00; afternoon requests require `hr13`–`hr24`, corresponding to
12:00–23:00. Python sends the exact keys in the prompt and request-specific
schema and maps by key rather than JSON order. Valid returned hours are kept;
missing keys and individual malformed or crossed triples remain pending.
Invalid JSON, duplicate keys, extra top-level fields, unrequested hour keys,
an invalid `chunk_total`, transport failures and truncation yield no new
accepted hours from that reply. Previous successes still remain cached.
Labels such as `hr01` and `hour1` are not accepted.

The existing historical summaries and calendar context remain unchanged:
requested date, weekday and actual hours 0–11 or 12–23. Date selection follows
the requested chunk, never the count of API calls. Python creates timestamps;
exported CSVs retain `hour_start,forecast,p10,p90`, without hr-key columns.
Raw labelled responses remain in the new trial's request journal. Invalid dimensions, extra
fields, strings/bools used as numbers, nonfinite/negative values, crossed
quantiles and truncation trigger retries; nothing is clipped, sorted or replaced
with classical predictions. An exhausted chunk remains missing while later
chunks are attempted; the stage can publish only when all required chunks are
valid. October CSV publication is atomic and uses the same
four-column contract as Part 1, with p50 stored as `forecast`.

Each model-stated request total is checked against its requested hourly p50 sum
at absolute tolerance `1e-6`, when every requested p50 is numerically available.
This arithmetic can be checked even if numeric quantile bounds cross; missing
or nonnumeric values make it undefined. Incoherence is reported without repair
or a special retry.
Chunk CSVs (`*_chunks.csv` and `october_chunks.csv`) preserve the start, exclusive
end, hourly p50 sum, stated total, signed discrepancy and coherence flag.

The final chunk's model-stated total is available only when one accepted reply
covers all twelve hours. If valid hours were stitched from several partial
replies, no model stated a total for that combined forecast: its stated total,
discrepancy and coherence are left undefined. Request-level arithmetic is
reported separately, including partial and failed attempts with numeric p50s.

Daily point totals sum all 24 hourly p50 values. Daily stated totals sum two
known full-chunk model totals; if either is undefined, daily arithmetic is also
undefined. Coherence tables show scorable and undefined counts, and exclude
undefined cases from their rates. Where both totals exist, opposite chunk
errors can cancel at the daily level. These are not independent
model-stated whole-day totals, and daily point sums are not necessarily daily
medians. Daily LLM interval
metrics are undefined because no daily quantiles were requested; hourly bounds
are never summed into daily intervals.

Every backtest reports full-window and final-seven-day WAPE, bias, totals, total
errors, hourly coverage/width and pinball loss. Invalid-reply share divides
invalid or partially valid received replies by all received replies. Partial
reply counts and requested/accepted hourly-value counts are reported separately.
Transport and HTTP failures
are separate. Latency includes failed calls and unloading; active-request time
also includes retry delays. Wall time can include manual pauses on resumed
runs. Token totals include available usage from failed replies, with missing
usage counts reported. Three September runs produce pairwise hourly MAE,
monthly-total range and per-day repeat variation; retries are not repeat runs.

`/explain` uses only the loaded Part 1 hourly and daily artifacts. It checks two
or three model-written sentences, known fact placeholders, numeric claims and
range ordering, then substitutes computed figures rounded to one decimal.
Unknown/missing facts, literal numeric claims, invalid content and provider
failure after three retries return a fixed template. Numeric checking does not
prove every prose interpretation correct. Existing forecast endpoints and
readiness do not require Ollama; explanations never train or forecast.

## Part 2 implementation verification

The two new test modules use synthetic data, temporary Git repositories and
HTTPX mocks. They exercise all manual stage transitions, all three September
runs, all 410 successful forecast calls, all 744 October rows, raw failure
evidence, retry limits, frozen commits,
post-origin leakage, provider adapters, atomic publication, explanation checks
and fallback, including recorded cancellation without automatic retries.
Automatic failed-trial archiving is checked for byte preservation, successful
result protection, archive failures and exclusion of another process after
journal rotation.
Recovery tests also cover later chunks, later development origins, all three
September runs and suppression of incomplete publications. Label tests cover
shuffled keys, both half-days, date boundaries, cache isolation, incorrect
schemas and one-time reset isolation. Per-hour recovery tests cover shrinking
requests, first-valid-value retention, partial-cache replay after exhaustion or
Ctrl+C, global-budget stops, rejected overwrites and unavailable stitched totals.
Real
Qwen experiment results, live explanation verification and
the combined executed notebook must be checked during your manual run.

Current implementation checks:

- Combined local suite: **225 passed** (58 Part 1 and 167 Part 2).
- Combined Linux Docker suite: **225 passed**; no live model/provider calls.
- Running Docker API: all 744 hourly and 31 daily values and metadata match
  the original artifacts; `/explain` returns a checked template with Ollama off.
- Docker-to-host loopback connectivity passed using a temporary mock server;
  the service image contains neither notebook/test tooling nor model weights.
- New report sections executed in a temporary notebook with mocked results:
  three code cells, one embedded figure, no execution errors.
- Original modelling files, Part 1 results, `forecast.csv`, and the submitted
  Part 1 notebook remain unchanged. Earlier positional responses and archives
  were deleted for the requested fresh start; only the stable runtime lock
  remains until manual execution. No real labelled-model calls, submission CSV
  or experiment commits were created by implementation checks.
