# Part 1 results

The executed report is [report.ipynb](report.ipynb).

Selected method: **quantile_filtered**. Selection and modelling code were committed at `a929ab4eb675156753dddcfbccb1f8e9fbf2fc7d` before September was evaluated.

October point total: **43,974.8** recorded studies.

## Drawbacks and limitations

The submitted method was frozen as **quantile_filtered** using development windows only.
September is an evaluation result, not a reason to change that selection.

1. **Recorded arrivals are not established demand.** In June–August there are no
   rows at stored hour 00:00 or 18:00–23:00. July 8 and 9 contain only 36 and 1
   studies. We preserve zero recorded counts and do not infer true closures,
   outages, timezone corrections, or unobserved patients. Future capture changes
   could invalidate the learned clock pattern.
2. **Unusual-day treatment matters.** Mean development hourly WAPE is
   20.34% for raw quantile training and
   19.95% with the specified
   near-empty-day exclusion. Validation counts are never removed. Excluding
   plausible capture failures can also remove genuinely low-demand days.
   On September, raw quantile training has 15.71%
   hourly WAPE versus 15.85% for the frozen filtered choice;
   this holdout difference does not change the previously selected method.
3. **Long-horizon performance is measured, not assumed.** September hourly WAPE
   is 15.85% overall and 15.32% in the
   last seven days; bias is 4.57% and 4.52%.
   The three development windows overlap, so they are not independent trials.
4. **An 80% interval is a target, not a guarantee.** September hourly coverage is
   91.25% overall and 92.86% in the final week.
   Coverage on hours with positive recorded arrivals is 87.65%;
   predictable zero hours can make overall coverage look stronger. Daily coverage
   is 76.67% overall and 71.43% in
   the final week. Only 21 complete prior days calibrate daily ranges, using
   seven-day horizons rather than a full month. No post-September recalibration
   or widening is performed to improve the reported holdout.
5. **The feature set has limited information.** Trees can fit historical level
   changes but cannot extrapolate a continuing growth trend beyond training.
   Four months cannot establish annual seasonality or October-specific holidays.
   No future hospital staffing, onboarding, hospital volumes, or outages are
   known. Hourly quantiles are marginal estimates, not joint sample paths.
   A sum of hourly medians is a daily point forecast, not a mathematical
   expectation or necessarily the daily median.
6. **Errors are concentrated.** The largest September daily absolute errors are:

- 2026-09-14: 1,085 recorded studies, 1,521.4 forecast (+436.4); daily range 1,191.3–1,820.6.
- 2026-09-04: 1,945 recorded studies, 1,532.5 forecast (-412.5); daily range 1,202.3–1,831.7.
- 2026-09-10: 1,128 recorded studies, 1,513.0 forecast (+385.0); daily range 1,182.8–1,812.2.

   The largest hourly MAEs by stored hour are 04:00 (27.9 studies MAE), 05:00 (24.1 studies MAE), 06:00 (22.5 studies MAE).

Next steps would be a documented timezone, source-system completeness and outage
logs, more historical months, hospital onboarding schedules, and additional
rolling month-long tests of calibration. These data were not invented for this
submission. October actuals are unavailable; no October accuracy is claimed.
