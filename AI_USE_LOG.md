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

Checks include synthetic feature/metric examples, tests that alter future counts
without changing earlier forecasts, file/checksum and Git freeze checks, local
and container tests, notebook execution, and comparisons of API results with the
saved CSVs. The development selection is committed before September evaluation.
Part 2 usage will be appended only after the Part 1 freeze.
