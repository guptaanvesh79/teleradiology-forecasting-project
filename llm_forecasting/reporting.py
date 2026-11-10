"""Read saved evidence into the existing notebook; never call an LLM."""

import pandas as pd

from forecasting.api import load_artifacts
from .explain import TEMPLATE, explanation_facts, render, validate_explanation
from .pipeline import (
    ROOT,
    frozen_protocol,
    output,
    read_json,
    records,
    verify_manifest,
    write_frame,
)


def coherence_summary(errors):
    values = pd.Series(errors, dtype="float64")
    known = values.dropna()
    return {
        "periods": len(values),
        "scorable_periods": len(known),
        "undefined_periods": int(values.isna().sum()),
        "coherent_share": float((known.abs() <= 1e-6).mean()) if len(known) else None,
        "mean_absolute_discrepancy": float(known.abs().mean()) if len(known) else None,
        "maximum_absolute_discrepancy": (
            float(known.abs().max()) if len(known) else None
        ),
    }


def saved_report_data(root=ROOT):
    root = root.resolve()
    base = output(root)
    protocol = frozen_protocol(root)
    final_manifest = read_json(base / "final/manifest.json")
    if not final_manifest.get("complete"):
        raise ValueError(
            "Three complete September runs are required for the combined report"
        )
    verify_manifest(root, final_manifest)
    verify_manifest(root, read_json(base / "forecast_manifest.json"))
    examples = read_json(base / "explanations.json")
    hourly, daily, _ = load_artifacts(root)
    facts = explanation_facts(hourly, daily)
    live, fallback = examples["live"], examples["simulated_provider_failure"]
    if live["source"] != "llm" or live["facts"] != facts:
        raise ValueError(
            "The live explanation is missing or does not match Part 1 facts"
        )
    accepted = [record for record in live["records"] if record["status"] == "valid"]
    if (
        not accepted
        or validate_explanation(accepted[-1]["text"], facts) != live["explanation"]
    ):
        raise ValueError(
            "The saved LLM explanation differs from its validated response"
        )
    if (
        fallback["source"] != "template"
        or fallback["facts"] != facts
        or fallback["explanation"] != render(TEMPLATE, facts)
    ):
        raise ValueError("Saved fallback demonstration does not match Part 1 facts")

    part1 = pd.read_csv(root / "artifacts/final/metrics.csv").assign(
        method_source="Part 1"
    )
    llm = pd.read_csv(base / "final/metrics.csv").assign(method_source="LLM")
    comparison = pd.concat([part1, llm], ignore_index=True)
    selection = read_json(root / "artifacts/selection.json")["selected_candidate"]
    selected = part1.loc[(part1.candidate == selection) & (part1.scope == "all")].iloc[
        0
    ]
    mean_wape = float(llm.loc[llm.scope == "all", "hourly_wape"].mean())
    direction = "lower" if mean_wape < selected.hourly_wape else "higher or equal"
    recommendation = (
        f"The LLM's mean September hourly WAPE is {mean_wape:.2%}, {direction} than "
        f"the frozen Part 1 model's {selected.hourly_wape:.2%}. "
        "I would keep the classical forecast as the production baseline and use the LLM "
        "for validated explanations. A single holdout, three stochastic draws, and "
        "uncalibrated LLM quantiles do not establish a reliable production replacement. "
        "Any numeric ensemble needs additional origin-safe development and evaluation; "
        "this submission does not tune or blend using September outcomes."
    )
    run1 = pd.read_csv(
        base / "final/2026-09-01_run1_hourly.csv", parse_dates=["hour_start"]
    ).set_index("hour_start")
    classical = pd.read_csv(
        root / f"artifacts/final/2026-09-01_{selection}_hourly.csv",
        parse_dates=["hour_start"],
    ).set_index("hour_start")
    if (
        not classical.index.equals(run1.index)
        or not (classical.actual.to_numpy() == run1.actual.to_numpy()).all()
    ):
        raise ValueError("Comparison methods do not use the same September actuals")
    relative = (
        pd.DataFrame(
            {
                "actual": run1.actual,
                "part1_forecast": classical.forecast,
                "llm_run1_forecast": run1.forecast,
            }
        )
        .resample("D")
        .sum()
    )
    relative["part1_absolute_error"] = (relative.part1_forecast - relative.actual).abs()
    relative["llm_absolute_error"] = (
        relative.llm_run1_forecast - relative.actual
    ).abs()
    relative["llm_improvement"] = (
        relative.part1_absolute_error - relative.llm_absolute_error
    )
    llm_wins = relative.loc[relative.llm_improvement > 1e-9]
    part1_wins = relative.loc[relative.llm_improvement < -1e-9]
    where_models_win = pd.concat(
        [
            llm_wins.nlargest(2, "llm_improvement"),
            part1_wins.nsmallest(2, "llm_improvement"),
        ]
    )
    daily_comparison_summary = (
        f"Using the first fixed LLM run: LLM wins on {len(llm_wins)} days, "
        f"Part 1 wins on {len(part1_wins)} days, and "
        f"{len(relative)-len(llm_wins)-len(part1_wins)} days tie on daily-total absolute error. "
        "Examples show the largest daily differences in each direction when present; "
        "they are descriptive and do not change the frozen forecast selection."
    )
    days = run1[["actual", "forecast"]].resample("D").sum()
    day = (days.forecast - days.actual).abs().idxmax()
    hour = (run1.forecast - run1.actual).abs().idxmax()
    failures = []
    for label, when, row in (
        ("Largest daily absolute error", day, days.loc[day]),
        ("Largest hourly absolute error", hour, run1.loc[hour]),
    ):
        if abs(row.forecast - row.actual) > 1e-9:
            failures.append(
                f"{label}, {when}: actual {row.actual:.1f}, forecast {row.forecast:.1f}, signed error {row.forecast-row.actual:+.1f} studies (September run 1)."
            )
    misses = run1.loc[(run1.actual < run1.p10) | (run1.actual > run1.p90)]
    if not misses.empty:
        when = (misses.forecast - misses.actual).abs().idxmax()
        row = misses.loc[when]
        failures.append(
            f"Interval miss, {when}: actual {row.actual:.1f} outside [{row.p10:.1f}, {row.p90:.1f}] (run 1)."
        )
    if len(failures) < 3:
        raw = pd.read_csv(base / "final/2026-09-01_run1_daily.csv")
        known = raw.dropna(subset=["coherence_error"])
        row = (
            known.loc[known.coherence_error.abs().idxmax()] if not known.empty else None
        )
        if row is not None and abs(row.coherence_error) > 1e-6:
            failures.append(
                f"Daily arithmetic discrepancy, {row['date']}: sum of stated chunk totals {row.stated_total:.1f}, hourly p50 sum {row.forecast:.1f}."
            )
    coherence = []
    for stage in ("development", "final"):
        for path in sorted(
            [*(base / stage).glob("*_daily.csv"), *(base / stage).glob("*_chunks.csv")]
        ):
            frame = pd.read_csv(path)
            coherence.append(
                {
                    "window": path.stem.rsplit("_", 1)[0],
                    "level": "chunk" if path.stem.endswith("_chunks") else "daily",
                    **coherence_summary(frame.coherence_error),
                }
            )
    for suffix, level in (("daily", "daily"), ("chunks", "chunk")):
        frame = pd.read_csv(base / f"october_{suffix}.csv")
        coherence.append(
            {
                "window": "October",
                "level": level,
                **coherence_summary(frame.coherence_error),
            }
        )
    events = records(root)
    request_coherence = []
    for stage, origin, run in sorted(
        {(row["stage"], row["origin"], row["run"]) for row in events}
    ):
        rows = [
            row
            for row in events
            if (row["stage"], row["origin"], row["run"]) == (stage, origin, run)
            and row["received_reply"]
        ]
        request_coherence.append(
            {
                "stage": stage,
                "origin": origin,
                "run": run,
                **coherence_summary(
                    [
                        (row.get("hour_validation") or {}).get("coherence_error")
                        for row in rows
                    ]
                ),
            }
        )
    if len(failures) < 3:
        for event in events:
            failed = (event.get("hour_validation") or {}).get("failed_hours", {})
            if failed:
                key = next(iter(failed))
                failures.append(
                    f"Invalid hourly reply in {event['key']}, {key}: {failed[key]} This hour required retry recovery."
                )
                break
    return {
        "protocol": protocol,
        "comparison": comparison,
        "where_models_win": where_models_win,
        "daily_comparison_summary": daily_comparison_summary,
        "recommendation": recommendation,
        "failure_examples": failures,
        "coherence": pd.DataFrame(coherence),
        "request_coherence": pd.DataFrame(request_coherence),
        "examples": examples,
    }


def notebook_cells(root=ROOT):
    import nbformat

    base = output(root)
    required = (
        "protocol.json",
        "final/manifest.json",
        "forecast_manifest.json",
        "explanations.json",
    )
    if not all((base / name).exists() for name in required) or not read_json(
        base / "final/manifest.json"
    ).get("complete"):
        return [
            nbformat.v4.new_markdown_cell(
                "## Part 2 status\n\nThe separate LLM implementation is available. Real experiments and their report are pending manual execution; no LLM accuracy is claimed."
            )
        ]
    markdown, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    return [
        markdown("""# Part 2 — local Qwen forecasting

The Q4_K_M Qwen3.5-4B model runs natively through Ollama with thinking disabled,
temperature 0.2, and no fixed sampling seed. The exact context size is shown in
the frozen generation settings below. Initial requests predict twelve hours;
retries request only still-missing or invalid hours. First-valid hourly values
are retained across retries and manual resumes, without replacement.
All chunks use the
same pre-origin summaries; no actual counts or generated forecasts from inside
the horizon are fed to later chunks or retry prompts. Each response maps requested
hr1–hr12 or hr13–hr24 keys
to quantile arrays; Python maps those labels to requested timestamps regardless
of JSON key order. Dates and weekdays remain in the input context. Saved CSVs
use ordinary timestamps, with no hr-label column. There is one development run
per origin and three September runs. Earlier positional trial responses were
discarded at the user's request before this labelled protocol; results here
describe the labelled trial only.

All failed attempts remain in the request journal. Three retries are distinct
from the three independent September runs. Input and output token counts are
provider-reported. Local API charges are zero; electricity and hardware are not
priced. Cold model loading, unloading and retries contribute to measured time.
The Part 2 prompt was frozen before its own September scoring; the Part 1
September results had already been seen, so this is not a pristine blind study.
"""),
        code("""from llm_forecasting.reporting import saved_report_data
llm_report = saved_report_data(ROOT)
llm_root = ROOT / "artifacts/llm"
print("Frozen identity and generation settings")
display(llm_report["protocol"]["identity"])
display(llm_report["protocol"]["generation"])
print("Every LLM development and September run, including final-week scores")
display(pd.read_csv(llm_root / "development/metrics.csv"))
display(pd.read_csv(llm_root / "final/metrics.csv"))
print("September comparison: all Part 1 methods and each LLM run; errors are fractions")
display(llm_report["comparison"])
print(llm_report["daily_comparison_summary"])
display(llm_report["where_models_win"])
print(llm_report["recommendation"])
print("Measured failures; no examples are invented if absent")
for example in llm_report["failure_examples"]:
    print(example)
"""),
        markdown("""## Reliability, repeatability and arithmetic

Invalid-reply share uses received replies as its denominator. HTTP and transport
failures are reported separately; all attempts contribute to time and usage.
Request arithmetic compares the stated total with p50 values for that call's
requested hours, when all those numeric values are available. Invalid bounds
can still have numeric p50 values; missing or nonnumeric values make this check
undefined. Partial replies count as invalid received replies for reliability.

A twelve-hour chunk assembled from multiple partial replies has no model-stated
total for the combined result. Its chunk-level coherence is undefined, and so
is its daily-level coherence if either half-day lacks a full-chunk stated total.
The separate request table retains each individual call's arithmetic evidence.
Undefined cases are counted separately, excluded from coherence-rate denominators,
and never repaired or reported as arithmetic failures. Where both full-chunk
totals exist, opposite errors can cancel at the daily level. A sum of hourly medians is not necessarily
a daily median. No LLM daily quantiles were requested: daily interval metrics
are undefined, and hourly bounds are never summed into daily ranges.
"""),
        code("""for stage in ("development", "final"):
    stats = json.loads((llm_root / stage / "reliability.json").read_text())
    print(stage, "— every attempt, including retries")
    display(pd.Series({k: v for k, v in stats.items() if k != "windows"}))
    display(pd.DataFrame(stats["windows"]))
display(pd.Series(json.loads((llm_root / "october_reliability.json").read_text())))
print("Twelve-hour chunk and assembled daily total consistency")
display(llm_report["coherence"])
print("Arithmetic among received request replies, including partial attempts")
display(llm_report["request_coherence"])
repeat = json.loads((llm_root / "final/repeatability.json").read_text())
display(pd.Series({k: v for k, v in repeat.items() if k != "daily_total_std"}))
fig, axes = plt.subplots(2, 1, figsize=(12, 7))
backtest_daily.actual.plot(ax=axes[0], label="Actual", color="black")
backtest_daily.forecast.plot(ax=axes[0], label="Frozen Part 1", color="#2563eb")
for run in (1, 2, 3):
    rows = pd.read_csv(llm_root / f"final/2026-09-01_run{run}_daily.csv", parse_dates=["date"]).set_index("date")
    rows.forecast.plot(ax=axes[0], label=f"Qwen run {run}", alpha=.7)
axes[0].set(title="September daily forecasts: actuals, classical model, and repeat LLM runs", ylabel="Recorded studies")
axes[0].legend()
pd.Series(repeat["daily_total_std"]).plot.bar(ax=axes[1], color="#7c3aed")
axes[1].set(title="Variation across three LLM runs", ylabel="Daily point-total standard deviation", xlabel="September date")
show_chart(fig)
oct_llm = pd.read_csv(ROOT / "llm_forecast.csv", parse_dates=["hour_start"]).set_index("hour_start")
print("October LLM daily point totals; October actuals are unavailable")
display(oct_llm.forecast.resample("D").sum().to_frame())
"""),
        markdown("""## Validated explanations and limitations

The explanation uses Part 1 numbers only. Model-written sentences contain fact
placeholders; unsupported figures, unknown facts, invalid sentence counts and
provider failures lead to a fixed template. Numeric checking does not prove
that all prose semantics are correct, so causal claims are not requested.

Quantization may change model quality; no unquantized comparison was run.
The summaries discard study-level and some temporal detail. Historical profile
quantiles are not calibrated forecast intervals. Four months cannot reveal
annual changes, undocumented outages or future staffing. Three stochastic runs
give limited evidence about variability. Late-horizon WAPE and coverage above
must be read separately; valid JSON does not establish useful predictions.
"""),
        code("""print("Genuine local-model explanation")
display(llm_report["examples"]["live"]["explanation"])
print("Deliberately simulated provider failure — fixed-template fallback")
display(llm_report["examples"]["simulated_provider_failure"]["explanation"])
print("Both use these verified Part 1 facts")
display(llm_report["examples"]["live"]["facts"])
"""),
    ]


def build_combined_report(root=ROOT):
    data = saved_report_data(root)
    write_frame(
        output(root) / "comparison.csv",
        data["comparison"].set_index(["method_source", "candidate", "scope"]),
    )
    from forecasting.reporting import build_report

    build_report(root)
