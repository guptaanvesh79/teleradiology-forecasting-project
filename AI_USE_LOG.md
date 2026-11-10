# AI-use log

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

For Part 2, AI helped implement schemas, backtests, retries and explanations.
Installation exposed Homebrew's missing MLX grammar library, resolved with
Q4_K_M. A timestamp failure led to Python-generated dates.
Pytest caught a numeric checker misreading date punctuation. Live development
exposed a stale 4,096-token input limit rejecting a 5,518-token prompt; the guard
now uses the configured budget. Positional replies also crossed quantiles.
At my request, positional responses and archives were deleted for a fresh
hr1–hr24 trial. Per-hour retries preserve accepted values without replacing them.
Experiments and the combined notebook remain pending
manual execution. No paid calls were made.
