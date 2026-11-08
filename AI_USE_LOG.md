# AI-use log — Part 1

I used ChatGPT/Codex to help plan the approach, explain calendar features, and
draft data preparation, forecasting, API, Docker, tests, and report code. All
Part 1 forecast numbers are computed locally by classical algorithms. No history
was sent to an LLM to obtain forecasts, and no LLM API is used in this phase.

One AI-written test incorrectly assumed pytest's `tmp_path` was empty, although
another fixture had already created valid forecast files there. Pytest caught
the mistake: startup correctly succeeded when the test expected failure. The
test was corrected to use a separate nonexistent directory; correct service
behaviour was preserved.

Verification includes hand-checked feature values, synthetic input cases, a
regression check that changing future counts cannot change earlier forecasts,
comparisons between API responses and saved CSVs, local and Docker pytest runs,
and execution of the notebook. Model selection was committed before September
evaluation, and that holdout was not used for subsequent tuning.

Part 2 has not started. Its AI usage and checks will be added to this log after
the completed Part 1 commit.
