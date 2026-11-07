# AI-use log — Part 1

I used ChatGPT/Codex for planning, explaining calendar features and forecasting
objectives, environment setup, drafting Python code, tests, Docker configuration,
and the notebook/report. The classical predictions themselves are calculated by
local scikit-learn estimators and a seasonal-naive algorithm. No history was sent
to an LLM to obtain forecasts; no LLM API or explanation endpoint is used in Part 1.

One concrete error was in an AI-written startup test: it treated pytest's
`tmp_path` as an empty directory even though another fixture had already populated
that same directory with valid forecasts. The first run reported **40 passing
tests and one failure** because startup correctly succeeded. I inspected the
fixture paths, changed the missing-artifact test to use a separate nonexistent
subdirectory, and reran the suite. This fixed the test instead of changing correct
service behaviour to satisfy it.

During setup, the smoke script initially used pandas' deprecated `dayofweek`
property; runtime warnings prompted changing it to `day_of_week`. Homebrew's
Docker cask also required administrator-only paths even with binary linking
disabled; the official downloaded app was installed directly and user-level CLI
links were verified with Docker's hello-world container.

Original checks included synthetic feature/metric examples, tests that altered future counts
without changing earlier forecasts, file/checksum and Git freeze checks, local
and container tests, notebook execution, and comparisons of API results with the
saved CSVs. The development selection is committed before September evaluation.
Part 2 usage will be appended only after the Part 1 freeze.

On 2026-11-08, the user requested removal of the pytest files and Docker test
target. The test results above describe the original Part 1 commit; the current
verification scripts were subsequently removed as part of the minimum-layout
cleanup. The earlier counts are historical results, not a current test suite.

Codex also resolved Ruff diagnostics and formatting on 2026-11-08. All saved
backtest and October predictions and API responses were compared with the
pre-cleanup version. The original selection and results remain unchanged;
checksum-protected source changes are recorded in the separate Ruff maintenance
record rather than replacing the original pre-September provenance.

The user also requested the minimum project layout. Optional helper scripts,
environment snapshots, duplicate Markdown report, separate chart files, and
editor/cache files were removed. The notebook retains the report and embeds its
charts directly; the final training audit was moved under `artifacts/final/`.
