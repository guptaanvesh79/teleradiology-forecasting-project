# AI-use log

ChatGPT/Codex helped plan calendar features, data preparation, forecasting, API,
Docker, tests and reporting. Classical algorithms produce Part 1's numbers.

An AI-written test assumed pytest's `tmp_path` was empty despite existing fixture
files. Pytest caught the mistaken startup-failure expectation. The fixture was
corrected to use a separate nonexistent directory, preserving service behaviour.

Verification includes hand-checked features, synthetic inputs, future-count
leakage checks, API/artifact comparisons, local and Docker pytest, and notebook
execution. Part 1 selection was committed before September evaluation.

For Part 2, AI helped implement schemas, backtests, retries and explanations.
Installation exposed Homebrew's missing MLX grammar library, resolved with
Q4_K_M. A timestamp failure led to Python-generated dates.
Pytest caught a numeric checker misreading date punctuation. Live development
exposed a stale 4,096-token input limit rejecting a 5,518-token prompt; the guard
now uses the configured budget. Positional replies also crossed quantiles.
At my request, positional responses and archives were deleted for a fresh
hr1–hr24 trial. Per-hour retries preserve accepted values without replacing them.
The direct development and three September runs were then executed manually.

AI helped build daily-volume/hourly-allocation forecasting and rolling calibration.
Pytest corrected a retry-count expectation that ignored one-day calibration tails.
After development, I requested promotion into the main LLM package and deletion
of the direct trial and temporary folders. Its daily development CSVs and raw
journal were retained unchanged with migration provenance. This is an exploratory
hybrid, differing from L1's direct hourly-quantile requirement. Further experiments,
notebook regeneration and commits remain manual. Part 1 is preserved; no paid
calls were made.
