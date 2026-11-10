"""Manual stages with frozen protocols, append-only evidence and atomic exports."""

from contextlib import contextmanager
from datetime import datetime, timedelta
import fcntl
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile

import numpy as np
import pandas as pd

from forecasting.artifacts import validate_hourly
from forecasting.config import DEVELOPMENT_ORIGINS
from forecasting.data import hourly_counts, load_studies, sha256_file
from forecasting.metrics import evaluate, fraction
from forecasting.pipeline import FROZEN_FILES, frozen_selection

from .client import GENERATION, GenerationError, LLMClient, PartialResponseError, digest
from .prompts import (
    CHUNK_HOURS,
    COHERENCE_TOLERANCE,
    HOUR_LABEL_MAPPING,
    SCHEMA_POLICY,
    RecoveredForecast,
    SYSTEM_PROMPT,
    daily_from_chunks,
    forecast_chunks,
    forecast_messages,
    forecast_hour_keys,
    forecast_schema,
    frames_from_reply,
    history_summary,
    inspect_forecast,
    response_schema_template,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = tuple(
    f"llm_forecasting/{name}.py"
    for name in ("__init__", "__main__", "client", "prompts", "pipeline")
)
PROTOCOL_VERSION = 4
RETRY_POLICY = {
    "retain_first_valid_hour": True,
    "request_only_remaining_hours": True,
    "maximum_calls_per_chunk_invocation": GENERATION["retries"] + 1,
    "stitched_chunk_stated_total": "undefined",
}
DEVELOPMENT_DAYS, FINAL_DAYS, FINAL_RUNS, FORECAST_DAYS = 28, 30, 3, 31
REQUEST_COUNTS = {
    "development": len(DEVELOPMENT_ORIGINS) * DEVELOPMENT_DAYS * 24 // CHUNK_HOURS,
    "final": FINAL_RUNS * FINAL_DAYS * 24 // CHUNK_HOURS,
    "october": FORECAST_DAYS * 24 // CHUNK_HOURS,
}
EXPECTED_FORECAST_REQUESTS = sum(REQUEST_COUNTS.values())
MAX_FORECAST_ATTEMPTS = EXPECTED_FORECAST_REQUESTS * (GENERATION["retries"] + 1)


class IncompleteForecastError(ValueError):
    def __init__(self, failed_chunks):
        self.failed_chunks = tuple(failed_chunks)
        super().__init__(
            f"Stage incomplete: {len(self.failed_chunks)} chunk(s) still missing:\n"
            + "\n".join(self.failed_chunks)
            + "\nSuccessful chunks are cached. Rerun the same command to request only missing chunks; no incomplete forecasts or metrics were published."
        )


def output(root):
    return Path(root) / "artifacts/llm"


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_bytes(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=".pending-", delete=False
        ) as file:
            temporary = Path(file.name)
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_json(path, value):
    atomic_bytes(path, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def write_frame(path, frame):
    atomic_bytes(
        path,
        frame.to_csv(
            index=True, date_format="%Y-%m-%d %H:%M:%S", float_format="%.10f"
        ).encode(),
    )


@contextmanager
def exclusive(root):
    base = output(root)
    base.mkdir(parents=True, exist_ok=True)
    with (base / ".pipeline.lock").open("a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(
                "Another LLM experiment is running; use one at a time"
            ) from error
        try:
            # The lock survives journal rotation and is shared by every stage.
            with (base / "requests.jsonl").open("a", encoding="utf-8"):
                pass
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def records(root):
    path = output(root) / "requests.jsonl"
    if not path.exists():
        return []
    try:
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        required = {
            "key",
            "stage",
            "origin",
            "run",
            "experiment_sha256",
            "request_sha256",
            "status",
            "received_reply",
            "latency_seconds",
        }
        if any(not isinstance(row, dict) or not required <= row.keys() for row in rows):
            raise ValueError("Incomplete request record")
    except (ValueError, TypeError) as error:
        raise ValueError(
            "Request journal is damaged; preserve it for investigation"
        ) from error
    if any(
        row.get("protocol_version") != PROTOCOL_VERSION
        or row.get("chunk_hours") != CHUNK_HOURS
        for row in rows
    ):
        raise ValueError(
            "Incompatible LLM request journal; preserve it and start a separate 12-hour experiment"
        )
    return rows


def append_record(root, event):
    path = output(root) / "requests.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(event, allow_nan=False) + "\n")
        file.flush()
        os.fsync(file.fileno())


def prepare_development_journal(root, experiment_sha):
    """Under exclusive(), rotate only a failed-only, unfrozen older trial."""
    base = output(root)
    previous = records(root)
    if not previous:
        return
    if any(
        not isinstance(row["experiment_sha256"], str) or not row["experiment_sha256"]
        for row in previous
    ):
        raise ValueError(
            "Request journal has invalid experiment hashes; preserve it for investigation"
        )
    previous_hashes = {row["experiment_sha256"] for row in previous}
    if previous_hashes == {experiment_sha}:
        return

    protected_paths = [
        base / "protocol.json",
        base / "forecast_manifest.json",
        root / "llm_forecast.csv",
    ]
    protected_paths.extend((base / "development").rglob("*"))
    protected_paths.extend((base / "final").rglob("*"))
    protected_paths.extend(base.glob("october_*"))
    failed_statuses = ("invalid", "http_error", "transport_error", "interrupted")
    if (
        len(previous_hashes) != 1
        or any(
            row["stage"] != "development" or row["status"] not in failed_statuses
            for row in previous
        )
        or any(path.is_file() for path in protected_paths)
    ):
        raise ValueError(
            "Existing evidence belongs to another protocol and is not a failed-only development trial; preserve it and archive the experiment manually"
        )

    journal = base / "requests.jsonl"
    try:
        archive_root = base / "failed_development"
        archive_root.mkdir(parents=True, exist_ok=True)
        archive_dir = Path(
            tempfile.mkdtemp(
                prefix=datetime.now().strftime("%Y%m%dT%H%M%S-"), dir=archive_root
            )
        )
        destination = archive_dir / "requests.jsonl"
        os.replace(journal, destination)
        journal.touch(exist_ok=False)
    except OSError as error:
        raise OSError(
            f"Could not finish archiving the previous development journal under {base / 'failed_development'}; no new forecast was requested. Inspect the preserved files before retrying: {error}"
        ) from error
    print(
        f"Archived {len(previous)} failed development attempts to {destination}. Starting a fresh journal for the updated experiment.",
        flush=True,
    )


def reset_llm_trial(root=ROOT):
    """Explicit destructive reset; never called automatically by a CLI stage."""
    root = Path(root).resolve()
    base = output(root)
    if not base.resolve().is_relative_to(root):
        raise ValueError(
            "Refusing to reset an LLM artifact directory outside the project"
        )
    with exclusive(root):
        for path in base.iterdir():
            if path.name == ".pipeline.lock":
                continue
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()
        (root / "llm_forecast.csv").unlink(missing_ok=True)
    print(
        "Cleared the previous Part 2 trial. The next develop run starts fresh with labelled hours.",
        flush=True,
    )


def hashes(root, paths):
    return {
        str(Path(path).relative_to(root)): sha256_file(path)
        for path in sorted(paths)
        if path.is_file()
    }


def part1_hashes(root):
    paths = [root / "forecast.csv", *(root / "artifacts").rglob("*")]
    return hashes(root, [p for p in paths if output(root) not in p.parents])


def verify_hashes(root, expected):
    for name, checksum in expected.items():
        path = root / name
        if not path.is_file() or sha256_file(path) != checksum:
            raise ValueError(f"Frozen file changed or is missing: {name}")


def journal_snapshot(root):
    content = (output(root) / "requests.jsonl").read_bytes()
    return {
        "journal_bytes": len(content),
        "journal_sha256": hashlib.sha256(content).hexdigest(),
    }


def verify_journal(root, saved):
    content = (output(root) / "requests.jsonl").read_bytes()[: saved["journal_bytes"]]
    if (
        len(content) != saved["journal_bytes"]
        or hashlib.sha256(content).hexdigest() != saved["journal_sha256"]
    ):
        raise ValueError("Saved request evidence changed")


def runtime():
    return {
        "python": platform.python_version(),
        **{
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scikit-learn", "httpx", "pydantic")
        },
    }


def protocol_base(root, client):
    frozen_selection(root)  # Part 1 must already have committed, frozen results.
    if platform.python_version() != "3.13.18":
        raise ValueError("Run assignment experiments using main_env Python 3.13.18")
    return {
        "version": PROTOCOL_VERSION,
        "chunk_hours": CHUNK_HOURS,
        "request_counts": REQUEST_COUNTS,
        "expected_forecast_requests": EXPECTED_FORECAST_REQUESTS,
        "provider": client.settings.public(),
        "generation": GENERATION,
        "identity": client.identity(),
        "runtime": runtime(),
        "data_sha256": sha256_file(root / "data/studies.csv"),
        "sources": hashes(root, [root / name for name in SOURCE_FILES]),
        "part1_sources": hashes(root, [root / name for name in FROZEN_FILES]),
        "part1_results": part1_hashes(root),
        "system_prompt": SYSTEM_PROMPT,
        "schema_template": response_schema_template(),
        "hour_label_mapping": HOUR_LABEL_MAPPING,
        "schema_policy": SCHEMA_POLICY,
        "retry_policy": RETRY_POLICY,
        "development_origins": list(DEVELOPMENT_ORIGINS),
        "development_days": DEVELOPMENT_DAYS,
        "final_origin": "2026-09-01",
        "final_days": FINAL_DAYS,
        "final_runs": FINAL_RUNS,
        "forecast_origin": "2026-10-01",
        "forecast_days": FORECAST_DAYS,
        "max_forecast_attempts": MAX_FORECAST_ATTEMPTS,
        "coherence_tolerance": COHERENCE_TOLERANCE,
        "daily_stated_total_definition": "sum of two known full-chunk model totals; undefined when either chunk is stitched from partial replies",
        "api_budget_usd": 0,
        "daily_point_definition": "sum_of_hourly_p50; not necessarily a daily median",
    }


def validate_protocol_contract(saved):
    if (
        saved.get("version") != PROTOCOL_VERSION
        or saved.get("chunk_hours") != CHUNK_HOURS
        or saved.get("schema_template") != response_schema_template()
        or saved.get("hour_label_mapping") != HOUR_LABEL_MAPPING
        or saved.get("schema_policy") != SCHEMA_POLICY
        or saved.get("retry_policy") != RETRY_POLICY
        or saved.get("request_counts") != REQUEST_COUNTS
        or saved.get("expected_forecast_requests") != EXPECTED_FORECAST_REQUESTS
        or saved.get("max_forecast_attempts") != MAX_FORECAST_ATTEMPTS
    ):
        raise ValueError(
            "Incompatible LLM protocol; preserve it and start a separate 12-hour experiment"
        )


def frozen_protocol(root=ROOT):
    root = Path(root)
    path = output(root) / "protocol.json"
    saved = read_json(path)
    validate_protocol_contract(saved)
    for field in ("sources", "part1_sources", "part1_results", "development_files"):
        verify_hashes(root, saved[field])
    verify_journal(root, saved)
    if saved["generation"] != GENERATION or saved["runtime"] != runtime():
        raise ValueError("Frozen generation settings or Python dependencies changed")
    if sha256_file(root / "data/studies.csv") != saved["data_sha256"]:
        raise ValueError("Input data changed after development")
    try:
        committed = subprocess.check_output(
            ["git", "show", "HEAD:artifacts/llm/protocol.json"],
            cwd=root,
            stderr=subprocess.PIPE,
        )
        if committed != path.read_bytes():
            raise ValueError(
                "Commit the exact LLM protocol before September evaluation"
            )
        for name, checksum in {
            **saved["sources"],
            **saved["development_files"],
        }.items():
            content = subprocess.check_output(
                ["git", "show", f"HEAD:{name}"], cwd=root, stderr=subprocess.PIPE
            )
            if hashlib.sha256(content).hexdigest() != checksum:
                raise ValueError(
                    "Commit the exact LLM implementation and development files before September evaluation"
                )
        journal = subprocess.check_output(
            ["git", "show", "HEAD:artifacts/llm/requests.jsonl"],
            cwd=root,
            stderr=subprocess.PIPE,
        )
        if (
            hashlib.sha256(journal[: saved["journal_bytes"]]).hexdigest()
            != saved["journal_sha256"]
        ):
            raise ValueError(
                "Commit the development request evidence before September evaluation"
            )
    except subprocess.CalledProcessError as error:
        raise ValueError(
            "Commit the LLM implementation, protocol and development results first"
        ) from error
    return saved


def check_client(client, protocol):
    if (
        client.settings.public() != protocol["provider"]
        or client.identity() != protocol["identity"]
    ):
        raise ValueError(
            "Provider, installed model or Ollama changed after development"
        )


def reliability(events):
    replies = [row for row in events if row["received_reply"]]
    invalid = sum(row["status"] in ("invalid", "partial") for row in replies)
    requested = sum(len(row.get("requested_hour_keys", [])) for row in events)
    accepted = sum(
        len((row.get("hour_validation") or {}).get("accepted_hours", {}))
        for row in events
    )
    latencies = [row["latency_seconds"] for row in events]
    return {
        "attempts": len(events),
        "received_replies": len(replies),
        "invalid_replies": invalid,
        "invalid_reply_share": fraction(invalid, len(replies)),
        "partial_replies": sum(row["status"] == "partial" for row in replies),
        "requested_hour_values": requested,
        "accepted_hour_values": accepted,
        "hour_acceptance_share": fraction(accepted, requested),
        "transport_failures": sum(row["status"] == "transport_error" for row in events),
        "interrupted_attempts": sum(row["status"] == "interrupted" for row in events),
        "http_failures": sum(row["status"] == "http_error" for row in events),
        "mean_call_seconds": float(np.mean(latencies)) if latencies else None,
        "active_request_seconds": float(
            sum(latencies) + sum(row.get("retry_delay_seconds", 0) for row in events)
        ),
        "input_tokens": sum(row.get("input_tokens") or 0 for row in events),
        "output_tokens": sum(row.get("output_tokens") or 0 for row in events),
        "usage_missing_attempts": sum(
            row.get("input_tokens") is None or row.get("output_tokens") is None
            for row in events
        ),
        "api_cost_usd": 0.0,
        "cost_excludes": "electricity and hardware",
    }


def pending_hours(hours, accepted):
    return hours[[key not in accepted for key in forecast_hour_keys(hours)]]


def replay_hour_cache(matching, summary, hours, client):
    accepted, stated_total = {}, None
    for row in matching:
        remaining = pending_hours(hours, accepted)
        if remaining.empty:
            raise ValueError(
                "Saved requests exist after this chunk was already complete"
            )
        messages = forecast_messages(summary, hours, remaining)
        schema = forecast_schema(remaining)
        _, payload = client.payload(messages, schema, GENERATION["max_output_tokens"])
        if row["request_sha256"] != digest(payload) or row.get("request") != payload:
            raise ValueError("A saved chunk has a different prompt or settings")
        inspection = row.get("hour_validation")
        if inspection is None:
            if row["status"] in ("valid", "partial"):
                raise ValueError("Cached forecast lacks per-hour validation evidence")
            continue
        checked = inspect_forecast(row["text"], remaining)
        if checked != inspection:
            raise ValueError("Cached hourly validation differs from the raw response")
        accepted.update(checked["accepted_hours"])
        if len(remaining) == CHUNK_HOURS and not checked["failed_hours"]:
            stated_total = checked["stated_total"]
    return accepted, stated_total


def recover_chunk(
    summary, hours, matching, root, client, key, stage, origin, run, experiment_sha
):
    accepted, stated_total = replay_hour_cache(matching, summary, hours, client)
    cached = len(accepted)
    if cached == CHUNK_HOURS:
        return RecoveredForecast(accepted, stated_total), cached
    state = {}

    def request_factory():
        remaining = pending_hours(hours, accepted)
        state.clear()
        state.update(hours=remaining, inspection=None)
        print(
            f"{key}: requesting {len(remaining)} remaining hours {forecast_hour_keys(remaining)}",
            flush=True,
        )
        return forecast_messages(summary, hours, remaining), forecast_schema(remaining)

    def validate(text):
        nonlocal stated_total
        remaining = state["hours"]
        inspection = inspect_forecast(text, remaining)
        state["inspection"] = inspection
        accepted.update(inspection["accepted_hours"])
        if len(remaining) == CHUNK_HOURS and not inspection["failed_hours"]:
            stated_total = inspection["stated_total"]
        if inspection["failed_hours"]:
            print(
                f"{key}: retained {len(accepted)}/{CHUNK_HOURS} valid hours; remaining {list(inspection['failed_hours'])}",
                flush=True,
            )
            error_type = (
                PartialResponseError if inspection["accepted_hours"] else ValueError
            )
            raise error_type(
                "Invalid or missing hours: " + ", ".join(inspection["failed_hours"])
            )
        return RecoveredForecast(accepted, stated_total)

    def before_attempt():
        if len(records(root)) >= MAX_FORECAST_ATTEMPTS:
            raise ValueError(
                f"The fixed {MAX_FORECAST_ATTEMPTS}-attempt experiment budget is exhausted"
            )

    def record(event):
        append_record(
            root,
            {
                **event,
                "protocol_version": PROTOCOL_VERSION,
                "chunk_hours": CHUNK_HOURS,
                "key": key,
                "stage": stage,
                "origin": str(origin),
                "run": run,
                "experiment_sha256": experiment_sha,
                "requested_hour_keys": forecast_hour_keys(state["hours"]),
                "hour_validation": state["inspection"],
            },
        )

    reply = client.generate(
        [],
        {},
        validate,
        record=record,
        before_attempt=before_attempt,
        max_output_tokens=GENERATION["max_output_tokens"],
        request_factory=request_factory,
    )
    return reply, cached


def forecast_window(series, origin, days, run, stage, root, client, experiment_sha):
    summary = history_summary(series, origin)
    hourly_parts, chunk_parts = [], []
    failed_chunks, reused = [], 0
    for hours in forecast_chunks(origin, days):
        key = f"{stage}/{origin}/{run}/{hours[0].isoformat()}"
        previous = records(root)
        if any(row["experiment_sha256"] != experiment_sha for row in previous):
            raise ValueError(
                "Existing evidence belongs to another protocol; archive this LLM experiment before new development"
            )
        matching = [row for row in previous if row["key"] == key]
        try:
            reply, cached = recover_chunk(
                summary,
                hours,
                matching,
                root,
                client,
                key,
                stage,
                origin,
                run,
                experiment_sha,
            )
            reused += cached == CHUNK_HOURS
        except GenerationError as error:
            # Configuration/context errors are systemic, rather than a
            # missing prediction that another calendar chunk can resolve.
            if any(
                attempt.get("http_status") in (400, 401, 403, 404, 405, 422)
                or (
                    attempt.get("input_tokens") is not None
                    and attempt.get("input_token_budget") is not None
                    and attempt["input_tokens"] > attempt["input_token_budget"]
                )
                for attempt in error.records
            ):
                raise
            failed_chunks.append(key)
            print(
                f"{key}: failed after four attempts; continuing with later chunks. {error}",
                flush=True,
            )
            continue
        hourly, chunks = frames_from_reply(reply, hours)
        hourly_parts.append(hourly)
        chunk_parts.append(chunks)
    print(
        f"{stage}/{origin}/{run}: {len(hourly_parts)}/{days * 24 // CHUNK_HOURS} chunks ready ({reused} reused, {len(failed_chunks)} missing).",
        flush=True,
    )
    if failed_chunks:
        raise IncompleteForecastError(failed_chunks)
    hourly, chunks = pd.concat(hourly_parts), pd.concat(chunk_parts)
    return hourly, daily_from_chunks(hourly, chunks), chunks


def score_window(actual, hourly, daily):
    # Daily intervals are unavailable. NaN placeholders adapt the existing metric
    # interface; replace their coverage/width with undefined before publication.
    scores = evaluate(
        actual,
        hourly.assign(crossed_before_sort=False),
        daily.assign(p10=np.nan, p90=np.nan),
        quantile_model=True,
    )
    for metrics in scores.values():
        metrics["daily_coverage"] = None
        metrics["mean_daily_interval_width"] = None
    return scores


def collect_windows(series, windows, stage, root, client, experiment_sha):
    """Try all independent windows before deciding whether a stage is complete."""
    completed, failed_chunks = [], []
    for origin, days, run in windows:
        try:
            frames = forecast_window(
                series, origin, days, run, stage, root, client, experiment_sha
            )
        except IncompleteForecastError as error:
            failed_chunks.extend(error.failed_chunks)
        else:
            completed.append((origin, days, run, frames))
    if failed_chunks:
        raise IncompleteForecastError(failed_chunks)
    return completed


def save_window(
    series, origin, days, run, stage, root, client, experiment_sha, *, frames=None
):
    hourly, daily, chunks = (
        frames
        if frames is not None
        else forecast_window(
            series, origin, days, run, stage, root, client, experiment_sha
        )
    )
    base = output(root) / stage
    label = f"{origin}_run{run}"
    actual = series.reindex(hourly.index)
    if actual.isna().any():
        raise ValueError("Backtest extends beyond observed history")
    write_frame(base / f"{label}_hourly.csv", hourly.assign(actual=actual))
    write_frame(base / f"{label}_chunks.csv", chunks)
    write_frame(
        base / f"{label}_daily.csv", daily.assign(actual=actual.resample("D").sum())
    )
    scores = score_window(actual, hourly, daily)
    return [
        {
            "origin": origin,
            "candidate": f"qwen_run{run}",
            "run": run,
            "scope": scope,
            **values,
        }
        for scope, values in scores.items()
    ]


def stage_reliability(root, stage):
    events = [row for row in records(root) if row["stage"] == stage]
    result = reliability(events)
    groups = sorted({(row["origin"], row["run"]) for row in events})
    result["windows"] = []
    for origin, run in groups:
        rows = [row for row in events if (row["origin"], row["run"]) == (origin, run)]
        stats = reliability(rows)
        starts = [datetime.fromisoformat(row["started_at"]) for row in rows]
        ends = [
            start + timedelta(seconds=row["latency_seconds"])
            for start, row in zip(starts, rows, strict=True)
        ]
        result["windows"].append(
            {
                "origin": origin,
                "run": run,
                "elapsed_wall_seconds": (max(ends) - min(starts)).total_seconds(),
                **stats,
            }
        )
    result["elapsed_note"] = (
        "Wall time includes manual pauses if a failed stage is resumed; active time includes call latency and retry delays."
    )
    return result


def develop(root=ROOT, client=None):
    root = Path(root)
    base = output(root)
    with exclusive(root):
        if (base / "final/manifest.json").exists():
            raise ValueError("September evaluation has started; development is frozen")
        if (base / "protocol.json").exists():
            validate_protocol_contract(read_json(base / "protocol.json"))
            raise ValueError(
                "Development protocol already exists; commit it and run evaluate-final"
            )
        records(root)  # Reject legacy evidence before contacting the provider.
        client = client or LLMClient()
        protocol = protocol_base(root, client)
        experiment_sha = digest(protocol)
        prepare_development_journal(root, experiment_sha)
        series = hourly_counts(load_studies(root / "data/studies.csv"))
        completed = collect_windows(
            series,
            [(origin, DEVELOPMENT_DAYS, 1) for origin in DEVELOPMENT_ORIGINS],
            "development",
            root,
            client,
            experiment_sha,
        )
        metrics = []
        for origin, days, run, frames in completed:
            metrics.extend(
                save_window(
                    series,
                    origin,
                    days,
                    run,
                    "development",
                    root,
                    client,
                    experiment_sha,
                    frames=frames,
                )
            )
        write_frame(
            base / "development/metrics.csv",
            pd.DataFrame(metrics).set_index(["origin", "candidate", "scope"]),
        )
        write_json(
            base / "development/reliability.json",
            stage_reliability(root, "development"),
        )
        protocol.update(
            experiment_sha256=experiment_sha,
            development_files=hashes(root, (base / "development").rglob("*")),
            **journal_snapshot(root),
        )
        write_json(base / "protocol.json", protocol)
        print(
            "Saved development results. Manually commit llm_forecasting, tests and artifacts/llm before evaluate-final.",
            flush=True,
        )


def verify_manifest(root, manifest):
    if manifest["protocol_sha256"] != sha256_file(output(root) / "protocol.json"):
        raise ValueError("Results belong to a different frozen LLM protocol")
    if manifest.get("complete"):
        verify_hashes(root, manifest["files"])
        verify_journal(root, manifest)


def repeatability(frames):
    if len(frames) != 3 or any(
        not frame.index.equals(frames[0].index) for frame in frames
    ):
        raise ValueError("Repeatability requires three complete aligned runs")
    matrix = pd.concat(
        [frame.forecast.rename(f"run{i}") for i, frame in enumerate(frames, 1)], axis=1
    )
    daily = matrix.resample("D").sum()
    totals = matrix.sum()
    return {
        "monthly_totals": totals.to_dict(),
        "monthly_total_range": float(totals.max() - totals.min()),
        "pairwise_hourly_mae": [
            {
                "runs": [f"run{i+1}", f"run{j+1}"],
                "mae": float((matrix.iloc[:, i] - matrix.iloc[:, j]).abs().mean()),
            }
            for i in range(3)
            for j in range(i + 1, 3)
        ],
        "mean_hourly_std": float(matrix.std(axis=1, ddof=0).mean()),
        "max_hourly_std": float(matrix.std(axis=1, ddof=0).max()),
        "daily_total_std": {
            str(day.date()): float(value)
            for day, value in daily.std(axis=1, ddof=0).items()
        },
        "std_definition": "population standard deviation across the three realized runs",
    }


def evaluate_final(root=ROOT, client=None):
    root = Path(root)
    with exclusive(root):
        protocol = frozen_protocol(root)
        path = output(root) / "final/manifest.json"
        if path.exists():
            manifest = read_json(path)
            verify_manifest(root, manifest)
            if manifest.get("complete"):
                print("Preserving completed September results; no LLM calls made.")
                return
        else:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip()
            manifest = {
                "protocol_sha256": sha256_file(output(root) / "protocol.json"),
                "pre_evaluation_commit": commit,
                "complete": False,
            }
            write_json(
                path, manifest
            )  # Freeze development even if the first call fails.
        client = client or LLMClient()
        check_client(client, protocol)
        series = hourly_counts(load_studies(root / "data/studies.csv"))
        completed = collect_windows(
            series,
            [("2026-09-01", FINAL_DAYS, run) for run in range(1, FINAL_RUNS + 1)],
            "final",
            root,
            client,
            protocol["experiment_sha256"],
        )
        metrics, frames = [], []
        for origin, days, run, window_frames in completed:
            metrics.extend(
                save_window(
                    series,
                    origin,
                    days,
                    run,
                    "final",
                    root,
                    client,
                    protocol["experiment_sha256"],
                    frames=window_frames,
                )
            )
            frames.append(
                pd.read_csv(
                    output(root) / f"final/2026-09-01_run{run}_hourly.csv",
                    parse_dates=["hour_start"],
                ).set_index("hour_start")
            )
        base = output(root) / "final"
        write_frame(
            base / "metrics.csv",
            pd.DataFrame(metrics).set_index(["origin", "candidate", "scope"]),
        )
        write_json(base / "repeatability.json", repeatability(frames))
        write_json(base / "reliability.json", stage_reliability(root, "final"))
        manifest.update(
            complete=True,
            files=hashes(root, [p for p in base.rglob("*") if p != path]),
            **journal_snapshot(root),
        )
        write_json(path, manifest)
        print(
            "Saved all three frozen September runs; do not tune against these results."
        )


def export_october(hourly, path):
    frame = hourly[["forecast", "p10", "p90"]].rename_axis("hour_start")
    content = frame.to_csv(date_format="%Y-%m-%d %H:%M:%S", float_format="%.10f")
    validate_hourly(pd.read_csv(io.StringIO(content)))
    if Path(path).exists():
        raise ValueError("Refusing to overwrite an existing llm_forecast.csv")
    atomic_bytes(path, content.encode())


def forecast(root=ROOT, client=None):
    root = Path(root)
    with exclusive(root):
        protocol = frozen_protocol(root)
        final = read_json(output(root) / "final/manifest.json")
        verify_manifest(root, final)
        if not final.get("complete"):
            raise ValueError(
                "Complete all three September runs before October forecasting"
            )
        path = output(root) / "forecast_manifest.json"
        if path.exists():
            verify_manifest(root, read_json(path))
            print("Preserving validated October forecast; no LLM calls made.")
            return
        if (root / "llm_forecast.csv").exists():
            raise ValueError(
                "Existing llm_forecast.csv has no verified manifest; preserve and inspect it"
            )
        client = client or LLMClient()
        check_client(client, protocol)
        series = hourly_counts(load_studies(root / "data/studies.csv"))
        hourly, daily, chunks = forecast_window(
            series,
            "2026-10-01",
            FORECAST_DAYS,
            1,
            "october",
            root,
            client,
            protocol["experiment_sha256"],
        )
        write_frame(output(root) / "october_daily.csv", daily)
        write_frame(output(root) / "october_chunks.csv", chunks)
        write_json(
            output(root) / "october_reliability.json",
            stage_reliability(root, "october"),
        )
        export_october(hourly, root / "llm_forecast.csv")
        paths = [
            root / "llm_forecast.csv",
            output(root) / "october_daily.csv",
            output(root) / "october_chunks.csv",
            output(root) / "october_reliability.json",
        ]
        write_json(
            path,
            {
                "complete": True,
                "protocol_sha256": sha256_file(output(root) / "protocol.json"),
                "files": hashes(root, paths),
                **journal_snapshot(root),
            },
        )
        print("Saved validated llm_forecast.csv with 744 October hours.")
