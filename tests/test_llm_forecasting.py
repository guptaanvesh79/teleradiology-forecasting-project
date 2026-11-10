"""LLM contracts and full stage transitions using temporary data and HTTP mocks."""

import json
from pathlib import Path
import shutil
import subprocess
import sys

import httpx
import numpy as np
import pandas as pd
import pytest

from llm_forecasting.client import (
    DEFAULT_MODEL,
    EXPECTED_DIGEST,
    EXPECTED_OLLAMA,
    GENERATION,
    GenerationError,
    LLMClient,
    Settings,
)
from llm_forecasting import pipeline
from llm_forecasting.prompts import (
    daily_from_chunks,
    forecast_chunks,
    forecast_hour_keys,
    forecast_messages,
    forecast_schema,
    frames_from_reply,
    history_summary,
    inspect_forecast,
    parse_forecast,
)

MORNING_HOURS = pd.date_range("2026-07-21", periods=12, freq="h", name="hour_start")


def reply(point=2.0, discrepancy=0.0, *, hour_keys=None):
    keys = forecast_hour_keys(MORNING_HOURS) if hour_keys is None else hour_keys
    return {
        "chunk_total": len(keys) * point + discrepancy,
        "hours": {key: [point / 2, point, point * 2] for key in keys},
    }


def reply_for_request(request, **kwargs):
    body = json.loads(request.content)
    for message in body.get("messages", []):
        if message.get("role") == "user":
            try:
                user = json.loads(message["content"])
            except (ValueError, TypeError):
                continue
            if isinstance(user, dict) and "response_hour_keys" in user:
                return reply(hour_keys=user["response_hour_keys"], **kwargs)
    return reply(**kwargs)


def parse_morning(text):
    return parse_forecast(text, MORNING_HOURS)


def ollama_result(value, **overrides):
    return {
        "message": {"content": json.dumps(value)},
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 100,
        "eval_count": 200,
        **overrides,
    }


def mock_client(chat, *, provider="ollama", api_key="", sleep=lambda _: None):
    calls = []

    def handler(request):
        calls.append(request)
        if request.url.path.endswith("/api/generate"):
            return httpx.Response(200, json={"done": True})
        return chat(request)

    http = httpx.Client(transport=httpx.MockTransport(handler))
    settings = Settings(
        provider,
        (
            "http://127.0.0.1:11434"
            if provider == "ollama"
            else "https://provider.invalid/v1"
        ),
        DEFAULT_MODEL if provider == "ollama" else "test-model",
        api_key,
    )
    return LLMClient(settings, http=http, sleep=sleep), calls


def test_origin_safe_summary_and_prompt_leakage(hourly_history):
    origin = "2026-07-21"
    changed = hourly_history.copy()
    changed.loc[origin:] = 10**9
    first = history_summary(hourly_history, origin)
    second = history_summary(changed, origin)
    assert first == second
    assert first["history_end"] == "2026-07-20 23:00:00"
    assert len(first["weekday_hour_profiles"]) == 168
    assert len(first["recent_28_daily_totals"]) == 28
    assert first["unusual_days_removed"] is False
    assert first["sparse_days"]
    dates = next(forecast_chunks(origin, 28))
    messages = forecast_messages(first, dates)
    assert messages == forecast_messages(second, dates)
    client, _ = mock_client(
        lambda request: httpx.Response(
            200, json=ollama_result(reply_for_request(request))
        )
    )
    validator = lambda text: parse_forecast(text, MORNING_HOURS)
    a = client.generate(messages, forecast_schema(MORNING_HOURS), validator)
    b = client.generate(
        forecast_messages(second, dates), forecast_schema(MORNING_HOURS), validator
    )
    for frame_a, frame_b in zip(
        frames_from_reply(a, dates), frames_from_reply(b, dates), strict=True
    ):
        pd.testing.assert_frame_equal(frame_a, frame_b)


@pytest.mark.parametrize(
    "problem",
    ["timezone", "missing", "unordered", "negative", "nan", "short", "bad_origin"],
)
def test_invalid_summary_inputs(hourly_history, problem):
    series, origin = hourly_history.copy(), "2026-07-21"
    if problem == "timezone":
        series.index = series.index.tz_localize("UTC")
    elif problem == "missing":
        series = series.drop(series.index[0])
    elif problem == "unordered":
        series = series.iloc[::-1]
    elif problem == "negative":
        series.iloc[0] = -1
    elif problem == "nan":
        series = series.astype(float)
        series.iloc[0] = np.nan
    elif problem == "short":
        series = series.iloc[:100]
    else:
        origin += " 01:00:00"
    with pytest.raises(ValueError):
        history_summary(series, origin)


@pytest.mark.parametrize(
    "origin,days,hours,count",
    [
        ("2026-07-07", 28, 672, 56),
        ("2026-09-01", 30, 720, 60),
        ("2026-10-01", 31, 744, 62),
    ],
)
def test_chunk_order_hour_grid_and_coherence(origin, days, hours, count):
    chunks = list(forecast_chunks(origin, days))
    assert len(chunks) == count and all(len(chunk) == 12 for chunk in chunks)
    frames, totals = [], []
    for dates in chunks:
        hourly, daily = frames_from_reply(
            parse_forecast(
                json.dumps(reply(discrepancy=1, hour_keys=forecast_hour_keys(dates))),
                dates,
            ),
            dates,
        )
        assert not daily.coherent.any()
        assert (daily.coherence_error == 1).all()
        frames.append(hourly)
        totals.append(daily)
    frame = pd.concat(frames)
    assert len(frame) == hours
    assert frame.index.equals(
        pd.date_range(origin, periods=hours, freq="h", name="hour_start")
    )
    np.testing.assert_array_equal(
        daily_from_chunks(frame, pd.concat(totals)).forecast,
        frame.forecast.resample("D").sum(),
    )


def test_twelve_hour_calendar_and_labelled_hour_mapping():
    morning, afternoon = list(forecast_chunks("2026-10-01", 1))
    for index, start in ((morning, 0), (afternoon, 12)):
        calendar = json.loads(forecast_messages({}, index)[1]["content"])["calendar"]
        assert calendar == {
            "date": "2026-10-01",
            "weekday": 3,
            "hours": list(range(start, start + 12)),
        }
        value = {
            "chunk_total": sum(range(start, start + 12)),
            "hours": {
                f"hr{hour + 1}": [hour, hour, hour + 1]
                for hour in range(start, start + 12)
            },
        }
        hourly, chunks = frames_from_reply(
            parse_forecast(json.dumps(value), index), index
        )
        assert hourly.forecast.tolist() == list(range(start, start + 12))
        assert chunks.coherent.all()
        assert chunks.chunk_end_exclusive.iloc[0] == index[0] + pd.Timedelta(hours=12)


@pytest.mark.parametrize(
    "start",
    ["2026-07-31 12:00", "2026-08-01 00:00", "2026-09-30 12:00", "2026-10-01 00:00"],
)
@pytest.mark.parametrize("order", ["reverse", "lexical"])
def test_shuffled_hour_keys_map_to_chronological_timestamps(start, order):
    index = pd.date_range(start, periods=12, freq="h", name="hour_start")
    values = {
        f"hr{hour + 1}": [hour * 5, hour * 5 + 2, hour * 5 + 4] for hour in index.hour
    }
    keys = list(reversed(values)) if order == "reverse" else sorted(values)
    value = {
        "chunk_total": sum(triple[1] for triple in values.values()),
        "hours": {key: values[key] for key in keys},
    }
    hourly, chunks = frames_from_reply(parse_forecast(json.dumps(value), index), index)
    assert hourly.index.equals(index)
    assert hourly.forecast.tolist() == [hour * 5 + 2 for hour in index.hour]
    assert chunks.coherent.all()


@pytest.mark.parametrize("half", [0, 1])
def test_prompt_and_schema_require_the_same_exact_hour_keys(half):
    index = list(forecast_chunks("2026-10-01", 1))[half]
    expected = [f"hr{number}" for number in range(half * 12 + 1, half * 12 + 13)]
    summary = {"unchanged": "history"}
    message = json.loads(forecast_messages(summary, index)[1]["content"])
    assert message["history"] == summary
    assert message["calendar"] == {
        "date": "2026-10-01",
        "weekday": 3,
        "hours": list(range(half * 12, half * 12 + 12)),
    }
    assert message["response_hour_keys"] == expected
    schema = forecast_schema(index)
    hours = schema["properties"]["hours"]
    assert hours["required"] == expected and list(hours["properties"]) == expected
    assert hours["additionalProperties"] is False
    assert all(
        triple["type"] == "array" and triple["minItems"] == triple["maxItems"] == 3
        for triple in hours["properties"].values()
    )
    assert (
        schema["required"] == ["chunk_total", "hours"]
        and schema["additionalProperties"] is False
    )


@pytest.mark.parametrize(
    "problem",
    [
        "missing",
        "extra",
        "wrong_half",
        "hr0",
        "hr25",
        "hr01",
        "HR1",
        "hour1",
        "2026-07-21_00",
        "positional",
    ],
)
def test_wrong_hour_key_sets_are_rejected(problem):
    value = reply()
    if problem == "missing":
        value["hours"].pop("hr12")
    elif problem == "extra":
        value["hours"]["hr13"] = [1, 2, 3]
    elif problem == "wrong_half":
        value["hours"] = {f"hr{number}": [1, 2, 3] for number in range(13, 25)}
    elif problem == "positional":
        value["hours"] = list(value["hours"].values())
    else:
        value["hours"][problem] = value["hours"].pop("hr1")
    with pytest.raises(ValueError):
        parse_forecast(json.dumps(value), MORNING_HOURS)


def test_duplicate_hour_key_is_rejected_before_mapping():
    text = json.dumps(reply()).replace('"hr1":', '"hr1": [1, 2, 3], "hr1":', 1)
    with pytest.raises(ValueError, match="Duplicate JSON key: hr1"):
        parse_forecast(text, MORNING_HOURS)


@pytest.mark.parametrize("problem", ["response_keys", "request_schema"])
def test_invalid_cached_hour_contract_stops_without_new_inference(
    hourly_history, tmp_path, problem
):
    client, calls = mock_client(
        lambda request: httpx.Response(
            200, json=ollama_result(reply_for_request(request))
        )
    )
    pipeline.forecast_window(
        hourly_history, "2026-07-21", 1, 1, "development", tmp_path, client, "fixed"
    )
    count = len(calls)
    rows = pipeline.records(tmp_path)
    if problem == "response_keys":
        value = json.loads(rows[0]["text"])
        value["hours"] = {f"hr{number}": [1, 2, 3] for number in range(13, 25)}
        rows[0]["text"] = json.dumps(value)
    else:
        rows[0]["request_sha256"] = "different-schema"
    path = pipeline.output(tmp_path) / "requests.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError):
        pipeline.forecast_window(
            hourly_history, "2026-07-21", 1, 1, "development", tmp_path, client, "fixed"
        )
    assert len(calls) == count


def test_hour_keys_repeat_across_dates_without_reusing_another_days_cache(
    hourly_history, tmp_path
):
    calendars = []

    def chat(request):
        user = json.loads(json.loads(request.content)["messages"][1]["content"])
        calendars.append(user["calendar"])
        point = 1.0 if user["calendar"]["date"] == "2026-07-21" else 3.0
        return httpx.Response(
            200, json=ollama_result(reply_for_request(request, point=point))
        )

    client, _ = mock_client(chat)
    hourly, _, _ = pipeline.forecast_window(
        hourly_history, "2026-07-21", 2, 1, "development", tmp_path, client, "fixed"
    )
    assert len(calendars) == 4 and hourly.loc["2026-07-21", "forecast"].eq(1).all()
    assert hourly.loc["2026-07-22", "forecast"].eq(3).all()
    assert len({row["key"] for row in pipeline.records(tmp_path)}) == 4


def test_explicit_trial_reset_removes_only_part2_artifacts(artifact_root):
    root = artifact_root
    protected = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    base = pipeline.output(root)
    (base / "failed_development/old").mkdir(parents=True)
    (base / "failed_development/old/requests.jsonl").write_text(
        "old archived responses"
    )
    (base / "requests.jsonl").write_text("old positional responses")
    (base / "protocol.json").write_text('{"version": 2}')
    (root / "llm_forecast.csv").write_text("old forecast")
    with pipeline.exclusive(root):
        inode = (base / ".pipeline.lock").stat().st_ino
    pipeline.reset_llm_trial(root)
    assert {path.name for path in base.iterdir()} == {".pipeline.lock"}
    assert (base / ".pipeline.lock").stat().st_ino == inode
    assert not (root / "llm_forecast.csv").exists()
    assert all(path.read_bytes() == content for path, content in protected.items())


def test_trial_reset_does_not_delete_a_running_stages_evidence(tmp_path):
    base = pipeline.output(tmp_path)
    with pipeline.exclusive(tmp_path):
        path = base / "requests.jsonl"
        path.write_bytes(b"in-progress responses")
        with pytest.raises(ValueError, match="Another LLM experiment"):
            pipeline.reset_llm_trial(tmp_path)
        assert path.read_bytes() == b"in-progress responses"


def test_trial_reset_refuses_an_external_artifact_directory(tmp_path):
    root, outside = tmp_path / "project", tmp_path / "outside"
    (root / "artifacts").mkdir(parents=True)
    outside.mkdir()
    (outside / "keep.txt").write_text("keep")
    (root / "artifacts/llm").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="outside the project"):
        pipeline.reset_llm_trial(root)
    assert (outside / "keep.txt").read_text() == "keep"


@pytest.mark.parametrize("days", [0, -1, 0.5, True])
def test_invalid_chunk_horizon(days):
    with pytest.raises(ValueError):
        list(forecast_chunks("2026-10-01", days))


def test_daily_coherence_can_hide_opposite_chunk_errors():
    frames, chunks = [], []
    for index, error in zip(forecast_chunks("2026-10-01", 1), (3, -3), strict=True):
        hourly, totals = frames_from_reply(
            parse_forecast(
                json.dumps(
                    reply(discrepancy=error, hour_keys=forecast_hour_keys(index))
                ),
                index,
            ),
            index,
        )
        frames.append(hourly)
        chunks.append(totals)
    hourly, chunks = pd.concat(frames), pd.concat(chunks)
    daily = daily_from_chunks(hourly, chunks)
    assert not chunks.coherent.any() and daily.coherent.all()
    assert daily.forecast.iloc[0] == daily.stated_total.iloc[0] == 48
    assert chunks.coherence_error.tolist() == [3, -3]


@pytest.mark.parametrize(
    "problem",
    ["missing_hour", "missing_half", "duplicate_half", "wrong_end", "wrong_sum"],
)
def test_daily_aggregation_rejects_incomplete_or_mismatched_halves(problem):
    pieces = [
        frames_from_reply(
            parse_forecast(
                json.dumps(reply(hour_keys=forecast_hour_keys(index))), index
            ),
            index,
        )
        for index in forecast_chunks("2026-10-01", 1)
    ]
    hourly = pd.concat([piece[0] for piece in pieces])
    chunks = pd.concat([piece[1] for piece in pieces])
    if problem == "missing_hour":
        hourly = hourly.iloc[:-1]
    elif problem == "missing_half":
        chunks = chunks.iloc[:1]
    elif problem == "duplicate_half":
        chunks = pd.concat([chunks.iloc[:1], chunks.iloc[:1]])
    elif problem == "wrong_end":
        chunks.loc[chunks.index[0], "chunk_end_exclusive"] += pd.Timedelta(hours=1)
    else:
        chunks.loc[chunks.index[0], "forecast"] += 1
    with pytest.raises(ValueError):
        daily_from_chunks(hourly, chunks)


@pytest.mark.parametrize("total", [-1, True, "24", float("nan"), float("inf")])
def test_invalid_chunk_total(total):
    value = reply()
    value["chunk_total"] = total
    with pytest.raises(ValueError):
        parse_forecast(json.dumps(value), MORNING_HOURS)


def test_twenty_four_hour_reply_is_rejected():
    value = reply()
    value["hours"] = list(value["hours"].values()) * 2
    with pytest.raises(ValueError):
        parse_forecast(json.dumps(value), MORNING_HOURS)


@pytest.mark.parametrize(
    "problem",
    [
        "json",
        "fences",
        "duplicate_key",
        "extra",
        "string",
        "bool",
        "negative",
        "nan",
        "infinity",
        "crossing",
        "days",
        "hours",
        "triple",
    ],
)
def test_strict_forecast_validation(problem):
    value = reply()
    if problem == "json":
        text = "{"
    elif problem == "fences":
        text = "```json\n" + json.dumps(value) + "\n```"
    elif problem == "duplicate_key":
        text = (
            '{"chunk_total":0,"chunk_total":24,"hours":'
            + json.dumps(value["hours"])
            + "}"
        )
    else:
        if problem == "extra":
            value["unexpected"] = 1
        elif problem == "days":
            value = {"days": [{"daily_total": 48, "hours": [[1, 2, 3]] * 24}]}
        elif problem == "hours":
            value["hours"].pop("hr12")
        elif problem == "triple":
            value["hours"]["hr1"].pop()
        else:
            value["hours"]["hr1"][0] = {
                "string": "1",
                "bool": True,
                "negative": -1,
                "nan": float("nan"),
                "infinity": float("inf"),
                "crossing": 10,
            }[problem]
        text = json.dumps(value)
    with pytest.raises(ValueError):
        parse_forecast(text, MORNING_HOURS)


def test_four_attempts_unchanged_requests_and_all_failures_recorded():
    sleeps, events = [], []
    client, calls = mock_client(
        lambda _: httpx.Response(200, json=ollama_result({"bad": True})),
        sleep=sleeps.append,
    )
    messages = [{"role": "user", "content": "fixed"}]
    with pytest.raises(GenerationError):
        client.generate(
            messages,
            forecast_schema(MORNING_HOURS),
            lambda text: parse_forecast(text, MORNING_HOURS),
            record=events.append,
        )
    requests = [
        json.loads(call.content)
        for call in calls
        if call.url.path.endswith("/api/chat")
    ]
    assert len(requests) == 4 and all(value == requests[0] for value in requests)
    assert sleeps == [1, 2, 4]
    assert [event["attempt"] for event in events] == [1, 2, 3, 4]
    assert all(
        event["received_reply"] and event["status"] == "invalid" for event in events
    )
    assert "seed" not in requests[0]["options"]
    assert requests[0]["keep_alive"] == 0
    assert requests[0]["options"]["num_ctx"] == GENERATION["context_length"]
    assert requests[0]["options"]["num_predict"] == GENERATION["max_output_tokens"]
    assert len([call for call in calls if call.url.path.endswith("/api/generate")]) == 4


def test_retry_then_success_and_coherence_does_not_trigger_retry():
    count, events = 0, []

    def chat(_):
        nonlocal count
        count += 1
        return httpx.Response(
            200, json=ollama_result({} if count == 1 else reply(discrepancy=7))
        )

    client, _ = mock_client(chat)
    result = client.generate(
        [],
        forecast_schema(MORNING_HOURS),
        lambda text: parse_forecast(text, MORNING_HOURS),
        record=events.append,
    )
    assert count == 2 and len(result.hours) == 12
    assert [row["status"] for row in events] == ["invalid", "valid"]
    stats = pipeline.reliability(events)
    assert stats["invalid_reply_share"] == 0.5 and stats["api_cost_usd"] == 0


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, SystemExit])
def test_interrupted_generation_records_evidence_without_retrying(interruption):
    events, sleeps = [], []

    def chat(_):
        raise interruption()

    client, calls = mock_client(chat, sleep=sleeps.append)
    with pytest.raises(interruption):
        client.generate(
            [],
            forecast_schema(MORNING_HOURS),
            lambda text: parse_forecast(text, MORNING_HOURS),
            record=events.append,
        )
    assert len(events) == 1 and events[0]["status"] == "interrupted"
    assert events[0]["retry_delay_seconds"] == 0 and sleeps == []
    assert len(calls) == 2  # One generation, followed by model unloading.
    assert calls[-1].url.path.endswith("/api/generate")
    assert pipeline.reliability(events)["interrupted_attempts"] == 1


def test_partial_window_resumes_without_repeating_successful_chunks(
    hourly_history, tmp_path
):
    chat_count = 0
    failing = True

    def chat(request):
        nonlocal chat_count
        chat_count += 1
        body = json.loads(request.content)
        calendar = json.loads(body["messages"][1]["content"])["calendar"]
        value = (
            {} if calendar["hours"][0] == 12 and failing else reply_for_request(request)
        )
        return httpx.Response(200, json=ollama_result(value))

    client, _ = mock_client(chat)
    with pytest.raises(pipeline.IncompleteForecastError):
        pipeline.forecast_window(
            hourly_history,
            "2026-07-21",
            1,
            1,
            "development",
            tmp_path,
            client,
            "fixed-experiment",
        )
    assert chat_count == 5
    failing = False
    hourly, daily, chunks = pipeline.forecast_window(
        hourly_history,
        "2026-07-21",
        1,
        1,
        "development",
        tmp_path,
        client,
        "fixed-experiment",
    )
    assert (
        chat_count == 6 and len(hourly) == 24 and len(daily) == 1 and len(chunks) == 2
    )
    assert len(pipeline.records(tmp_path)) == 6
    assert {row["key"].rsplit("/", 1)[-1] for row in pipeline.records(tmp_path)} == {
        "2026-07-21T00:00:00",
        "2026-07-21T12:00:00",
    }


def test_all_chunks_receive_the_same_pre_origin_history(hourly_history, tmp_path):
    client, calls = mock_client(
        lambda request: httpx.Response(
            200, json=ollama_result(reply_for_request(request))
        )
    )
    hourly, daily, chunks = pipeline.forecast_window(
        hourly_history, "2026-07-21", 2, 1, "development", tmp_path, client, "fixed"
    )
    messages = [
        json.loads(json.loads(call.content)["messages"][1]["content"])
        for call in calls
        if call.url.path.endswith("/api/chat")
    ]
    assert len(messages) == len(chunks) == 4 and len(hourly) == 48 and len(daily) == 2
    assert all(
        message["history"] == history_summary(hourly_history, "2026-07-21")
        for message in messages
    )
    assert [message["calendar"]["hours"][0] for message in messages] == [0, 12, 0, 12]


def test_exhausted_chunk_does_not_block_later_chunks_and_rerun_only_requests_missing(
    hourly_history, tmp_path
):
    failing = True
    requests = []

    def chat(request):
        body = json.loads(request.content)
        calendar = json.loads(body["messages"][1]["content"])["calendar"]
        requests.append(calendar)
        invalid = (
            failing and calendar["date"] == "2026-07-21" and calendar["hours"][0] == 0
        )
        return httpx.Response(
            200, json=ollama_result({} if invalid else reply_for_request(request))
        )

    client, _ = mock_client(chat)
    with pytest.raises(pipeline.IncompleteForecastError) as caught:
        pipeline.forecast_window(
            hourly_history, "2026-07-21", 2, 1, "development", tmp_path, client, "fixed"
        )
    assert caught.value.failed_chunks == (
        "development/2026-07-21/1/2026-07-21T00:00:00",
    )
    assert (
        len(requests) == 7
        and sum(row["status"] == "valid" for row in pipeline.records(tmp_path)) == 3
    )
    failing = False
    hourly, daily, chunks = pipeline.forecast_window(
        hourly_history, "2026-07-21", 2, 1, "development", tmp_path, client, "fixed"
    )
    assert len(requests) == 8 and requests[-1] == requests[0]
    assert len(hourly) == 48 and len(daily) == 2 and len(chunks) == 4


def test_all_failed_chunks_are_reported_without_empty_aggregation(
    hourly_history, tmp_path
):
    client, _ = mock_client(lambda _: httpx.Response(200, json=ollama_result({})))
    with pytest.raises(pipeline.IncompleteForecastError) as caught:
        pipeline.forecast_window(
            hourly_history, "2026-07-21", 1, 1, "development", tmp_path, client, "fixed"
        )
    assert len(caught.value.failed_chunks) == 2 and len(pipeline.records(tmp_path)) == 8
    assert "no incomplete forecasts or metrics" in str(caught.value)


@pytest.mark.parametrize("failure", ["crossing", "missing"])
def test_retry_requests_only_failed_hour_and_preserves_first_valid_values(
    hourly_history, tmp_path, failure
):
    payloads, sleeps = [], []

    def chat(request):
        body = json.loads(request.content)
        user = json.loads(body["messages"][1]["content"])
        payloads.append(body)
        value = reply_for_request(
            request, point=7 if user["response_hour_keys"] == ["hr9"] else 2
        )
        if len(payloads) == 1:
            if failure == "missing":
                value["hours"].pop("hr9")
            else:
                value["hours"]["hr9"] = [3, 2, 4]
        return httpx.Response(200, json=ollama_result(value))

    client, _ = mock_client(chat, sleep=sleeps.append)
    hourly, daily, chunks = pipeline.forecast_window(
        hourly_history, "2026-07-21", 1, 1, "development", tmp_path, client, "fixed"
    )
    first, retry = [json.loads(body["messages"][1]["content"]) for body in payloads[:2]]
    assert first["response_hour_keys"] == [f"hr{i}" for i in range(1, 13)]
    assert retry["response_hour_keys"] == ["hr9"]
    assert (
        first["history"] == retry["history"] and first["calendar"] == retry["calendar"]
    )
    assert payloads[1]["format"]["properties"]["hours"]["required"] == ["hr9"]
    assert payloads[0]["options"] == payloads[1]["options"] and sleeps == [1]
    assert hourly.loc["2026-07-21 08:00", "forecast"] == 7
    assert hourly.iloc[:8].forecast.eq(2).all() and hourly.iloc[9:].forecast.eq(2).all()
    assert chunks.stated_total.isna().tolist() == [True, False]
    assert daily.stated_total.isna().all() and daily.coherence_error.isna().all()
    events = pipeline.records(tmp_path)
    assert [row["status"] for row in events] == ["partial", "valid", "valid"]
    assert len(events[0]["hour_validation"]["accepted_hours"]) == 11
    assert events[0]["hour_validation"]["coherent"] is (
        True if failure == "crossing" else None
    )
    stats = pipeline.reliability(events)
    assert stats["partial_replies"] == 1 and stats[
        "invalid_reply_share"
    ] == pytest.approx(1 / 3)


def test_remaining_hours_shrink_across_four_bounded_attempts(hourly_history, tmp_path):
    requested, sleeps = [], []

    def chat(request):
        user = json.loads(json.loads(request.content)["messages"][1]["content"])
        keys = user["response_hour_keys"]
        requested.append(keys)
        number = len(requested)
        value = reply_for_request(request, point=number)
        bad = ("hr3", "hr9") if number == 1 else (("hr9",) if number in (2, 3) else ())
        for key in bad:
            value["hours"][key] = [number + 1, number, number + 2]
        return httpx.Response(200, json=ollama_result(value))

    client, _ = mock_client(chat, sleep=sleeps.append)
    hourly, _, _ = pipeline.forecast_window(
        hourly_history, "2026-07-21", 1, 1, "development", tmp_path, client, "fixed"
    )
    assert [len(keys) for keys in requested] == [12, 2, 1, 1, 12]
    assert requested[1:4] == [["hr3", "hr9"], ["hr9"], ["hr9"]]
    assert sleeps == [1, 2, 4]
    assert (
        hourly.iloc[0].forecast == 1
        and hourly.iloc[2].forecast == 2
        and hourly.iloc[8].forecast == 4
    )


def test_partial_success_survives_exhaustion_and_manual_resume(
    hourly_history, tmp_path
):
    failing, requested = True, []

    def chat(request):
        user = json.loads(json.loads(request.content)["messages"][1]["content"])
        keys = user["response_hour_keys"]
        requested.append(keys)
        value = reply_for_request(request)
        if failing and "hr9" in keys:
            if len(keys) == 12:
                value["hours"]["hr9"] = [3, 2, 4]
            else:
                value = {}
        return httpx.Response(200, json=ollama_result(value))

    client, _ = mock_client(chat)
    with pytest.raises(pipeline.IncompleteForecastError):
        pipeline.forecast_window(
            hourly_history, "2026-07-21", 1, 1, "development", tmp_path, client, "fixed"
        )
    assert [len(keys) for keys in requested] == [12, 1, 1, 1, 12]
    failing = False
    hourly, _, _ = pipeline.forecast_window(
        hourly_history, "2026-07-21", 1, 1, "development", tmp_path, client, "fixed"
    )
    assert len(requested) == 6 and requested[-1] == ["hr9"] and len(hourly) == 24
    assert (
        pipeline.reliability(pipeline.records(tmp_path))["accepted_hour_values"] == 24
    )


def test_interrupt_after_partial_response_preserves_accepted_hours(
    hourly_history, tmp_path
):
    interrupted, requested = True, []

    def chat(request):
        keys = json.loads(json.loads(request.content)["messages"][1]["content"])[
            "response_hour_keys"
        ]
        requested.append(keys)
        if keys == ["hr9"] and interrupted:
            raise KeyboardInterrupt()
        value = reply_for_request(request)
        if len(requested) == 1:
            value["hours"]["hr9"] = [3, 2, 4]
        return httpx.Response(200, json=ollama_result(value))

    client, _ = mock_client(chat)
    with pytest.raises(KeyboardInterrupt):
        pipeline.forecast_window(
            hourly_history, "2026-07-21", 1, 1, "development", tmp_path, client, "fixed"
        )
    interrupted = False
    pipeline.forecast_window(
        hourly_history, "2026-07-21", 1, 1, "development", tmp_path, client, "fixed"
    )
    assert requested == [
        [f"hr{i}" for i in range(1, 13)],
        ["hr9"],
        ["hr9"],
        [f"hr{i}" for i in range(13, 25)],
    ]


@pytest.mark.parametrize(
    "bad", [[-1, 2, 3], ["1", 2, 3], [1, True, 3], [1, 2], [3, 2, 4], [1, 4, 3]]
)
def test_independent_hour_validation_retains_valid_neighbours(bad):
    value = reply()
    value["hours"]["hr9"] = bad
    checked = inspect_forecast(json.dumps(value), MORNING_HOURS)
    assert len(checked["accepted_hours"]) == 11 and set(checked["failed_hours"]) == {
        "hr9"
    }


def test_missing_hour_is_retried_without_discarding_other_hours():
    value = reply()
    value["hours"].pop("hr9")
    checked = inspect_forecast(json.dumps(value), MORNING_HOURS)
    assert len(checked["accepted_hours"]) == 11 and checked["failed_hours"] == {
        "hr9": "Missing hour"
    }
    assert checked["coherence_error"] is None


def test_coherence_for_stitched_chunks_is_undefined_not_a_failure():
    from llm_forecasting.reporting import coherence_summary

    stats = coherence_summary([None, None])
    assert stats["scorable_periods"] == 0 and stats["undefined_periods"] == 2
    assert (
        stats["coherent_share"] is None and stats["mean_absolute_discrepancy"] is None
    )


@pytest.mark.parametrize("failure", ["http", "context", "budget"])
def test_systemic_errors_stop_without_requesting_later_chunks(
    hourly_history, tmp_path, monkeypatch, failure
):
    def chat(_):
        if failure == "http":
            return httpx.Response(401, json={"error": "Invalid configuration"})
        return httpx.Response(
            200,
            json=ollama_result(
                reply(), prompt_eval_count=GENERATION["context_length"] + 1
            ),
        )

    client, calls = mock_client(chat)
    if failure == "budget":
        monkeypatch.setattr(pipeline, "MAX_FORECAST_ATTEMPTS", 0)
    with pytest.raises((GenerationError, ValueError)) as caught:
        pipeline.forecast_window(
            hourly_history, "2026-07-21", 2, 1, "development", tmp_path, client, "fixed"
        )
    assert not isinstance(caught.value, pipeline.IncompleteForecastError)
    assert sum(call.url.path.endswith("/api/chat") for call in calls) == (
        0 if failure == "budget" else 4
    )


def test_revised_request_and_attempt_budgets():
    assert pipeline.REQUEST_COUNTS == {"development": 168, "final": 180, "october": 62}
    assert pipeline.EXPECTED_FORECAST_REQUESTS == 410
    assert pipeline.MAX_FORECAST_ATTEMPTS == 1640


@pytest.fixture
def failed_development_trial(tmp_path):
    for attempt in range(1, 5):
        pipeline.append_record(
            tmp_path,
            {
                "attempt": attempt,
                "key": "development/2026-07-21/1/2026-07-21T00:00:00",
                "stage": "development",
                "origin": "2026-07-21",
                "run": 1,
                "experiment_sha256": "old-experiment",
                "request_sha256": "old-request",
                "protocol_version": pipeline.PROTOCOL_VERSION,
                "chunk_hours": 12,
                "status": "invalid",
                "received_reply": True,
                "latency_seconds": 1.0,
                "error": "Earlier input-budget error",
                "raw_response": "preserve this evidence",
            },
        )
    return tmp_path, (pipeline.output(tmp_path) / "requests.jsonl").read_bytes()


def test_develop_automatically_archives_failed_trial_and_continues(
    failed_development_trial, hourly_history, monkeypatch, capsys
):
    root, original = failed_development_trial
    protocol = {"trial": "updated-implementation"}
    monkeypatch.setattr(pipeline, "protocol_base", lambda *_: protocol.copy())
    monkeypatch.setattr(pipeline, "load_studies", lambda _: hourly_history)
    monkeypatch.setattr(pipeline, "hourly_counts", lambda history: history)
    monkeypatch.setattr(pipeline, "DEVELOPMENT_ORIGINS", ("2026-07-21",))
    monkeypatch.setattr(pipeline, "DEVELOPMENT_DAYS", 1)
    client, calls = mock_client(
        lambda request: httpx.Response(
            200, json=ollama_result(reply_for_request(request))
        )
    )
    pipeline.develop(root, client)
    base = pipeline.output(root)
    archived = list((base / "failed_development").glob("*/requests.jsonl"))
    assert len(archived) == 1 and archived[0].read_bytes() == original
    active = pipeline.records(root)
    assert len(active) == 2 and all(row["status"] == "valid" for row in active)
    assert all(row["experiment_sha256"] == pipeline.digest(protocol) for row in active)
    assert sum(call.url.path.endswith("/api/chat") for call in calls) == 2
    assert (base / "protocol.json").is_file()
    assert "Archived 4 failed development attempts" in capsys.readouterr().out


def test_same_experiment_keeps_failed_attempts_for_resume(failed_development_trial):
    root, original = failed_development_trial
    with pipeline.exclusive(root):
        pipeline.prepare_development_journal(root, "old-experiment")
    assert (pipeline.output(root) / "requests.jsonl").read_bytes() == original
    assert not (pipeline.output(root) / "failed_development").exists()


@pytest.mark.parametrize(
    "protection",
    [
        "successful_chunk",
        "unknown_status",
        "other_stage",
        "mixed_experiments",
        "development_results",
        "frozen_protocol",
        "september_results",
        "october_csv",
        "forecast_manifest",
        "october_results",
    ],
)
def test_automatic_archive_preserves_ineligible_trials(
    failed_development_trial, protection
):
    root, _ = failed_development_trial
    base = pipeline.output(root)
    rows = pipeline.records(root)
    if protection == "successful_chunk":
        rows[0]["status"] = "valid"
    elif protection == "unknown_status":
        rows[0]["status"] = "unrecognized"
    elif protection == "other_stage":
        rows[0]["stage"] = "final"
    elif protection == "mixed_experiments":
        rows[0]["experiment_sha256"] = "another-old-experiment"
    else:
        target = {
            "development_results": base / "development/metrics.csv",
            "frozen_protocol": base / "protocol.json",
            "september_results": base / "final/manifest.json",
            "october_csv": root / "llm_forecast.csv",
            "forecast_manifest": base / "forecast_manifest.json",
            "october_results": base / "october_daily.csv",
        }[protection]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"Preserve existing results")
    journal = base / "requests.jsonl"
    journal.write_text("".join(json.dumps(row) + "\n" for row in rows))
    original = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    with pipeline.exclusive(root), pytest.raises(
        ValueError, match="not a failed-only development trial"
    ):
        pipeline.prepare_development_journal(root, "new-experiment")
    assert all(path.read_bytes() == content for path, content in original.items())
    assert not (base / "failed_development").exists()


@pytest.mark.parametrize("content", [b'{"unfinished":', b'{"status":"invalid"}\n'])
def test_malformed_journal_cannot_be_auto_archived(tmp_path, content):
    path = pipeline.output(tmp_path) / "requests.jsonl"
    path.parent.mkdir(parents=True)
    path.write_bytes(content)
    blocked, calls = mock_client(
        lambda _: pytest.fail("Malformed journal must block provider calls")
    )
    with pytest.raises(ValueError, match="damaged"):
        pipeline.develop(tmp_path, blocked)
    assert not calls and path.read_bytes() == content
    assert not (path.parent / "failed_development").exists()


def test_stable_lock_blocks_other_process_after_journal_rotation(
    failed_development_trial,
):
    root, original = failed_development_trial
    probe = """import sys
from pathlib import Path
from llm_forecasting.pipeline import exclusive
try:
    with exclusive(Path(sys.argv[1])):
        sys.exit(10)
except ValueError as error:
    print(error)
"""
    with pipeline.exclusive(root):
        pipeline.prepare_development_journal(root, "new-experiment")
        result = subprocess.run(
            [sys.executable, "-B", "-c", probe, str(root)],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert (
            result.returncode == 0
            and "Another LLM experiment is running" in result.stdout
        )
        assert (pipeline.output(root) / "requests.jsonl").read_bytes() == b""
    with pipeline.exclusive(root):
        assert pipeline.records(root) == []
    assert (
        next(
            (pipeline.output(root) / "failed_development").glob("*/requests.jsonl")
        ).read_bytes()
        == original
    )


@pytest.mark.parametrize("failure", ["rename", "new_journal"])
def test_archive_failure_stops_develop_before_inference(
    failed_development_trial, monkeypatch, failure
):
    root, original = failed_development_trial
    monkeypatch.setattr(pipeline, "protocol_base", lambda *_: {"trial": "new"})

    def denied(*args, **kwargs):
        raise PermissionError("Simulated archive failure")

    if failure == "rename":
        monkeypatch.setattr(pipeline.os, "replace", denied)
    else:
        monkeypatch.setattr(Path, "touch", denied)
    client, calls = mock_client(
        lambda _: pytest.fail("No inference after archive failure")
    )
    with pytest.raises(OSError, match="Could not finish archiving"):
        pipeline.develop(root, client)
    assert not calls
    base = pipeline.output(root)
    if failure == "rename":
        assert (base / "requests.jsonl").read_bytes() == original
    else:
        assert (
            next((base / "failed_development").glob("*/requests.jsonl")).read_bytes()
            == original
        )
        assert not (base / "requests.jsonl").exists()
    assert not (base / "protocol.json").exists()


def test_legacy_protocol_is_preserved_and_rejected(tmp_path):
    path = pipeline.output(tmp_path) / "protocol.json"
    pipeline.write_json(path, {"version": 1, "schemas": {"7": {}}})
    original = path.read_bytes()
    with pytest.raises(ValueError, match="Incompatible LLM protocol"):
        pipeline.frozen_protocol(tmp_path)
    blocked, calls = mock_client(
        lambda _: pytest.fail("Legacy protocol must block all provider calls")
    )
    with pytest.raises(ValueError, match="Incompatible LLM protocol"):
        pipeline.develop(tmp_path, blocked)
    assert not calls
    assert path.read_bytes() == original


@pytest.mark.parametrize("problem", ["legacy", "wrong_version", "wrong_size"])
def test_incompatible_journal_blocks_development_without_provider_calls(
    hourly_history, tmp_path, problem
):
    client, _ = mock_client(
        lambda request: httpx.Response(
            200, json=ollama_result(reply_for_request(request))
        )
    )
    pipeline.forecast_window(
        hourly_history, "2026-07-21", 1, 1, "development", tmp_path, client, "fixed"
    )
    path = pipeline.output(tmp_path) / "requests.jsonl"
    rows = pipeline.records(tmp_path)
    if problem == "legacy":
        for row in rows:
            row.pop("protocol_version")
            row.pop("chunk_hours")
    elif problem == "wrong_version":
        rows[-1]["protocol_version"] = 1
    else:
        rows[-1]["chunk_hours"] = 168
    original = "".join(json.dumps(row) + "\n" for row in rows).encode()
    path.write_bytes(original)
    blocked, calls = mock_client(
        lambda _: pytest.fail("Legacy journal must block even identity checks")
    )
    with pytest.raises(ValueError, match="Incompatible LLM request journal"):
        pipeline.develop(tmp_path, blocked)
    assert not calls and path.read_bytes() == original


def test_attempt_budget_prevents_requests(hourly_history, tmp_path, monkeypatch):
    client, calls = mock_client(
        lambda _: pytest.fail("Budget should prevent inference")
    )
    monkeypatch.setattr(pipeline, "MAX_FORECAST_ATTEMPTS", 0)
    with pytest.raises(ValueError, match="budget"):
        pipeline.forecast_window(
            hourly_history,
            "2026-07-21",
            7,
            1,
            "development",
            tmp_path,
            client,
            "fixed-experiment",
        )
    assert not calls


def test_zero_history_and_coherence_tolerance(hourly_history):
    summary = history_summary(hourly_history * 0, "2026-07-21")
    assert summary["sparse_days"] == []
    assert all(row[2:6] == [0, 0, 0, 0] for row in summary["weekday_hour_profiles"])
    dates = next(forecast_chunks("2026-10-01", 1))
    _, coherent = frames_from_reply(
        parse_forecast(json.dumps(reply(discrepancy=1e-7)), dates), dates
    )
    _, incoherent = frames_from_reply(
        parse_forecast(json.dumps(reply(discrepancy=1e-4)), dates), dates
    )
    assert coherent.coherent.all() and not incoherent.coherent.any()


@pytest.mark.parametrize(
    "context,output,inputs,valid",
    [
        (16384, 8192, 5518, True),
        (8192, 4096, 4096, True),
        (8192, 4096, 4097, False),
        (16384, 8192, 8192, True),
        (16384, 8192, 8193, False),
        (16384, 4096, 10000, True),
    ],
)
def test_prompt_budget_reserves_actual_output_allowance(
    monkeypatch, context, output, inputs, valid
):
    monkeypatch.setitem(GENERATION, "context_length", context)
    monkeypatch.setitem(GENERATION, "max_output_tokens", output)
    client, calls = mock_client(
        lambda _: httpx.Response(
            200, json=ollama_result(reply(), prompt_eval_count=inputs)
        )
    )
    events = []
    if valid:
        value = client.generate(
            [], forecast_schema(MORNING_HOURS), parse_morning, record=events.append
        )
        assert len(value.hours) == 12 and len(events) == 1
    else:
        with pytest.raises(
            GenerationError, match=f"{context-output}-token prompt budget"
        ):
            client.generate(
                [], forecast_schema(MORNING_HOURS), parse_morning, record=events.append
            )
        assert len(events) == 4
    assert all(row["input_token_budget"] == context - output for row in events)
    body = json.loads(calls[0].content)
    assert (
        body["options"]["num_ctx"] == context
        and body["options"]["num_predict"] == output
    )


def test_explicit_output_allowance_controls_input_budget(monkeypatch):
    monkeypatch.setitem(GENERATION, "context_length", 16384)
    monkeypatch.setitem(GENERATION, "max_output_tokens", 8192)
    client, _ = mock_client(
        lambda _: httpx.Response(
            200, json=ollama_result(reply(), prompt_eval_count=10000)
        )
    )
    events = []
    client.generate(
        [],
        forecast_schema(MORNING_HOURS),
        parse_morning,
        max_output_tokens=4096,
        record=events.append,
    )
    assert events[0]["input_token_budget"] == 12288
    assert events[0]["request"]["options"]["num_predict"] == 4096


def test_invalid_context_allocation_blocks_provider_call():
    client, calls = mock_client(
        lambda _: pytest.fail("Invalid allocation must block inference")
    )
    with pytest.raises(ValueError, match="leave room"):
        client.generate(
            [],
            forecast_schema(MORNING_HOURS),
            parse_morning,
            max_output_tokens=GENERATION["context_length"],
        )
    assert not calls


def test_crossed_quantiles_are_still_rejected_when_prompt_fits_budget():
    value = reply()
    value["hours"]["hr4"] = [137.2, 132.0, 150.6]
    client, _ = mock_client(
        lambda _: httpx.Response(200, json=ollama_result(value, prompt_eval_count=5518))
    )
    with pytest.raises(GenerationError, match="p10 <= p50 <= p90") as caught:
        client.generate([], forecast_schema(MORNING_HOURS), parse_morning)
    assert len(caught.value.records) == 4


def test_unrequested_retry_hours_cannot_overwrite_cached_successes(
    hourly_history, tmp_path
):
    count, requested = 0, []

    def chat(request):
        nonlocal count
        count += 1
        keys = json.loads(json.loads(request.content)["messages"][1]["content"])[
            "response_hour_keys"
        ]
        requested.append(keys)
        value = reply_for_request(request)
        if count == 1:
            value["hours"]["hr9"] = [3, 2, 4]
        elif count == 2:
            value["hours"]["hr1"] = [800, 900, 1000]
        return httpx.Response(200, json=ollama_result(value))

    client, _ = mock_client(chat)
    hourly, _, _ = pipeline.forecast_window(
        hourly_history, "2026-07-21", 1, 1, "development", tmp_path, client, "fixed"
    )
    assert requested[1:3] == [["hr9"], ["hr9"]]
    assert hourly.forecast.eq(2).all()
    assert pipeline.records(tmp_path)[1]["hour_validation"] is None


def test_global_budget_stop_preserves_partial_hour_cache(
    hourly_history, tmp_path, monkeypatch
):
    def chat(request):
        value = reply_for_request(request)
        value["hours"]["hr9"] = [3, 2, 4]
        return httpx.Response(200, json=ollama_result(value))

    client, _ = mock_client(chat)
    monkeypatch.setattr(pipeline, "MAX_FORECAST_ATTEMPTS", 1)
    with pytest.raises(ValueError, match="budget"):
        pipeline.forecast_window(
            hourly_history, "2026-07-21", 1, 1, "development", tmp_path, client, "fixed"
        )
    events = pipeline.records(tmp_path)
    assert (
        len(events) == 1 and len(events[0]["hour_validation"]["accepted_hours"]) == 11
    )


def test_provider_error_cannot_leak_key_into_saved_evidence():
    client, _ = mock_client(
        lambda _: httpx.Response(401, json={"error": "Invalid key private-test-key"}),
        provider="openai",
        api_key="private-test-key",
    )
    with pytest.raises(GenerationError) as caught:
        client.generate(
            [],
            forecast_schema(MORNING_HOURS),
            lambda text: parse_forecast(text, MORNING_HOURS),
        )
    assert "private-test-key" not in json.dumps(caught.value.records)
    assert "private-test-key" not in str(caught.value)
    assert "[redacted]" in json.dumps(caught.value.records)


@pytest.mark.parametrize(
    "failure", ["transport", "http", "truncated", "context", "unfinished"]
)
def test_runtime_and_response_failures(failure):
    def chat(request):
        if failure == "transport":
            raise httpx.ReadTimeout("test timeout", request=request)
        if failure == "http":
            return httpx.Response(503, json={"error": "unavailable"})
        options = {
            "truncated": {"done_reason": "length"},
            "context": {
                "prompt_eval_count": GENERATION["context_length"]
                - GENERATION["max_output_tokens"]
                + 1
            },
            "unfinished": {"done": False},
        }[failure]
        return httpx.Response(200, json=ollama_result(reply(), **options))

    client, _ = mock_client(chat)
    with pytest.raises(GenerationError) as error:
        client.generate(
            [],
            forecast_schema(MORNING_HOURS),
            lambda text: parse_forecast(text, MORNING_HOURS),
        )
    assert len(error.value.records) == 4
    assert "Last failure" in str(error.value)
    stats = pipeline.reliability(error.value.records)
    assert stats["transport_failures"] == (4 if failure == "transport" else 0)
    assert stats["http_failures"] == (4 if failure == "http" else 0)


@pytest.mark.parametrize("provider", ["openai", "gemini"])
def test_hosted_adapter_auth_schema_usage_and_key_redaction(provider):
    def chat(request):
        body = json.loads(request.content)
        if provider == "openai":
            assert request.headers["authorization"] == "Bearer private-test-key"
            assert body["text"]["format"]["type"] == "json_schema"
            assert body["store"] is False
            return httpx.Response(
                200,
                json={
                    "status": "completed",
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {"type": "output_text", "text": json.dumps(reply())}
                            ],
                        }
                    ],
                    "usage": {"input_tokens": 12, "output_tokens": 20},
                },
            )
        assert request.headers["x-goog-api-key"] == "private-test-key"
        assert body["generationConfig"]["responseMimeType"] == "application/json"
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "finishReason": "STOP",
                        "content": {"parts": [{"text": json.dumps(reply())}]},
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 12,
                    "candidatesTokenCount": 18,
                    "thoughtsTokenCount": 2,
                },
            },
        )

    client, calls = mock_client(chat, provider=provider, api_key="private-test-key")
    events = []
    result = client.generate(
        [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "predict"},
        ],
        forecast_schema(MORNING_HOURS),
        lambda text: parse_forecast(text, MORNING_HOURS),
        record=events.append,
    )
    assert len(result.hours) == 12 and len(calls) == 1
    assert events[0]["input_tokens"] == 12 and events[0]["output_tokens"] == 20
    assert events[0]["api_cost_usd"] is None
    assert "private-test-key" not in json.dumps(events)


def test_generic_configuration_and_cloud_requirements(monkeypatch):
    for name in ("LLM_PROVIDER", "LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    assert Settings.from_env() == Settings()
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    with pytest.raises(ValueError, match="require"):
        Settings.from_env()
    monkeypatch.setenv("LLM_MODEL", "configured-model")
    monkeypatch.setenv("LLM_API_KEY", "secret")
    assert Settings.from_env().public()["model"] == "configured-model"
    assert "secret" not in json.dumps(Settings.from_env().public())
    monkeypatch.setenv("LLM_BASE_URL", "http://insecure.invalid")
    with pytest.raises(ValueError, match="HTTPS"):
        Settings.from_env()


@pytest.mark.parametrize("problem", ["version", "digest", "missing", "hosted"])
def test_model_identity_guard(problem):
    def handler(request):
        if request.url.path.endswith("version"):
            return httpx.Response(
                200,
                json={"version": "other" if problem == "version" else EXPECTED_OLLAMA},
            )
        models = (
            []
            if problem == "missing"
            else [
                {
                    "name": DEFAULT_MODEL,
                    "digest": "other" if problem == "digest" else EXPECTED_DIGEST,
                }
            ]
        )
        return httpx.Response(200, json={"models": models})

    settings = (
        Settings()
        if problem != "hosted"
        else Settings("openai", "https://provider.invalid/v1", "test", "key")
    )
    client = LLMClient(
        settings, http=httpx.Client(transport=httpx.MockTransport(handler))
    )
    with pytest.raises(ValueError):
        client.identity()


def test_shared_metrics_and_repeatability():
    index = pd.date_range("2026-09-01", periods=720, freq="h", name="hour_start")
    actual = pd.Series(1.0, index=index)
    frames = [
        pd.DataFrame({"forecast": value, "p10": 0.5, "p90": 4.0}, index=index)
        for value in (1.0, 2.0, 3.0)
    ]
    daily = frames[1].forecast.resample("D").sum().to_frame()
    scores = pipeline.score_window(actual, frames[1], daily)
    assert scores["all"]["hourly_wape"] == 1
    assert scores["all"]["daily_wape"] == 1
    assert scores["all"]["bias"] == 1
    assert scores["all"]["total_error"] == 720
    assert scores["last_7_days"]["hours"] == 168
    assert scores["all"]["daily_coverage"] is None
    repeat = pipeline.repeatability(frames)
    assert repeat["monthly_total_range"] == 1440
    assert [row["mae"] for row in repeat["pairwise_hourly_mae"]] == [1, 2, 1]
    assert repeat["mean_hourly_std"] == pytest.approx(np.std([1, 2, 3]))
    zero = pipeline.score_window(actual * 0, frames[1], daily)
    assert zero["all"]["hourly_wape"] is None


def test_atomic_export_and_existing_submission_protection(tmp_path):
    index = pd.date_range("2026-10-01", periods=744, freq="h", name="hour_start")
    frame = pd.DataFrame({"forecast": 1.0, "p10": 0.0, "p90": 2.0}, index=index)
    path = tmp_path / "llm_forecast.csv"
    with pytest.raises(ValueError):
        pipeline.export_october(frame.iloc[:-1], path)
    assert not path.exists()
    pipeline.export_october(frame, path)
    original = path.read_bytes()
    assert len(pd.read_csv(path)) == 744
    assert pd.read_csv(path).hour_start.iloc[0] == "2026-10-01 00:00:00"
    with pytest.raises(ValueError, match="overwrite"):
        pipeline.export_october(frame * 2, path)
    assert path.read_bytes() == original
    assert not list(tmp_path.glob(".pending-*"))


def test_full_manual_stage_sequence_with_mocked_models(artifact_root, monkeypatch):
    run_manual_stage_sequence(artifact_root, monkeypatch)


def test_incomplete_stages_continue_all_windows_and_resume_without_publication(
    artifact_root, monkeypatch
):
    run_manual_stage_sequence(artifact_root, monkeypatch, recover_missing=True)


def test_per_hour_recovery_completes_full_experiment_with_truthful_coherence(
    artifact_root, monkeypatch
):
    run_manual_stage_sequence(artifact_root, monkeypatch, partial_recovery=True)


def run_manual_stage_sequence(
    artifact_root, monkeypatch, recover_missing=False, partial_recovery=False
):
    root = artifact_root
    project = Path(__file__).resolve().parents[1]
    for name in {*pipeline.SOURCE_FILES, *pipeline.FROZEN_FILES}:
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(project / name, destination)
    (root / "data").mkdir()
    timestamps = pd.date_range("2026-06-01", "2026-10-01", freq="h", inclusive="left")
    studies = pd.DataFrame(
        {
            "study_id": [f"S{i}" for i in range(len(timestamps))],
            "arrival_time": timestamps.strftime("%Y-%m-%d %H:%M:%S"),
            "hospital_id": "H",
            "modality": "CT",
            "specialty_required": "GENERAL",
            "criticality": "ROUTINE",
            "patient_location": "ED",
            "day_total_studies": 24,
        }
    )
    studies.to_csv(root / "data/studies.csv", index=False)
    pipeline.write_json(
        root / "artifacts/selection.json", {"selected_candidate": "seasonal_naive"}
    )
    idx = pd.date_range("2026-09-01", periods=720, freq="h", name="hour_start")
    actual = pd.Series(1.0, index=idx)
    pred = pd.DataFrame({"forecast": 1.25, "p10": 1.0, "p90": 2.0}, index=idx)
    pred.loc[:"2026-09-15", "forecast"] = 1.05
    pipeline.write_frame(
        root / "artifacts/final/2026-09-01_seasonal_naive_hourly.csv",
        pred.assign(actual=actual),
    )
    metrics = [
        {"candidate": "seasonal_naive", "scope": scope, **values}
        for scope, values in pipeline.score_window(
            actual, pred, pred.forecast.resample("D").sum().to_frame()
        ).items()
    ]
    pipeline.write_frame(
        root / "artifacts/final/metrics.csv",
        pd.DataFrame(metrics).set_index(["candidate", "scope"]),
    )
    monkeypatch.setattr(
        pipeline, "frozen_selection", lambda _: ({}, "part1-test-commit")
    )
    chats = []
    failing = {
        "development": recover_missing,
        "final": recover_missing,
        "october": recover_missing,
    }
    september_failures = 0

    def handler(request):
        nonlocal september_failures
        path = request.url.path
        if path.endswith("/api/version"):
            return httpx.Response(200, json={"version": EXPECTED_OLLAMA})
        if path.endswith("/api/tags"):
            return httpx.Response(
                200,
                json={
                    "models": [
                        {
                            "name": DEFAULT_MODEL,
                            "digest": EXPECTED_DIGEST,
                            "details": {"format": "gguf"},
                        }
                    ]
                },
            )
        if path.endswith("/api/generate"):
            return httpx.Response(200, json={"done": True})
        body = json.loads(request.content)
        chats.append(body)
        user = json.loads(body["messages"][1]["content"])
        if "calendar" in user:
            calendar = user["calendar"]
            date, hour = calendar["date"], calendar["hours"][0]
            invalid = hour == 0 and (
                (date == "2026-07-07" and failing["development"])
                or (date == "2026-10-01" and failing["october"])
                or (
                    date == "2026-09-01" and failing["final"] and september_failures < 4
                )
            )
            if invalid and date == "2026-09-01":
                september_failures += 1
            value = (
                {}
                if invalid
                else reply(
                    point=1.10 + len(chats) / 10000,
                    discrepancy=1,
                    hour_keys=user["response_hour_keys"],
                )
            )
            if partial_recovery and len(user["response_hour_keys"]) == 12:
                first_key = user["response_hour_keys"][0]
                triple = value["hours"][first_key]
                triple[0] = triple[1] + 1
        else:
            from llm_forecasting.explain import TEMPLATE

            sentences = TEMPLATE.split(". ")
            value = {
                "sentences": [
                    text if text.endswith(".") else text + "." for text in sentences
                ]
            }
        return httpx.Response(200, json=ollama_result(value))

    client = LLMClient(
        http=httpx.Client(transport=httpx.MockTransport(handler)), sleep=lambda _: None
    )
    preserved = {
        name: (root / name).read_bytes()
        for name in (
            "forecast.csv",
            "artifacts/daily_forecast.csv",
            "artifacts/selection.json",
        )
    }
    if recover_missing:
        with pytest.raises(pipeline.IncompleteForecastError):
            pipeline.develop(root, client)
        assert len(chats) == 171
        assert not (root / "artifacts/llm/protocol.json").exists()
        assert not list((root / "artifacts/llm/development").glob("*.csv"))
        assert any(
            row["origin"] == "2026-08-04" and row["status"] == "valid"
            for row in pipeline.records(root)
        )
        failing["development"] = False
    pipeline.develop(root, client)
    multiplier = 2 if partial_recovery else 1
    assert len(chats) == 168 * multiplier + (4 if recover_missing else 0)
    protocol = pipeline.read_json(root / "artifacts/llm/protocol.json")
    assert protocol["version"] == 4 and protocol["chunk_hours"] == 12
    assert protocol["hour_label_mapping"] == {
        f"hr{hour + 1}": hour for hour in range(24)
    }
    assert protocol["schema_policy"]["exact_requested_hour_keys"] is True
    assert protocol["schema_policy"]["key_order_significant"] is False
    assert "schema_template" in protocol and "execution_resume" not in protocol
    assert (
        protocol["expected_forecast_requests"] == 410
        and protocol["max_forecast_attempts"] == 1640
    )
    assert len(pd.read_csv(root / "artifacts/llm/development/metrics.csv")) == 6
    assert all(
        len(pd.read_csv(path)) == 56
        for path in (root / "artifacts/llm/development").glob("*_chunks.csv")
    )
    with pytest.raises(ValueError, match="Commit"):
        pipeline.frozen_protocol(root)

    def git(*args):
        subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.name", "Temporary test")
    git("config", "user.email", "test@example.invalid")
    git("add", ".")
    git("commit", "-qm", "Freeze mocked development")
    pipeline.frozen_protocol(root)
    source = root / "llm_forecasting/prompts.py"
    saved_source = source.read_bytes()
    source.write_bytes(saved_source + b"\n# changed\n")
    with pytest.raises(ValueError, match="Frozen file"):
        pipeline.evaluate_final(root, client)
    source.write_bytes(saved_source)
    if recover_missing:
        with pytest.raises(pipeline.IncompleteForecastError):
            pipeline.evaluate_final(root, client)
        assert len(chats) == 355
        assert (
            pipeline.read_json(root / "artifacts/llm/final/manifest.json")["complete"]
            is False
        )
        assert not list((root / "artifacts/llm/final").glob("*.csv"))
        assert not (root / "artifacts/llm/final/repeatability.json").exists()
        failing["final"] = False
    pipeline.evaluate_final(root, client)
    assert len(chats) == 348 * multiplier + (8 if recover_missing else 0)
    assert all(
        len(pd.read_csv(path)) == 60
        for path in (root / "artifacts/llm/final").glob("*_chunks.csv")
    )
    final = [row for row in pipeline.records(root) if row["stage"] == "final"]
    per_run = []
    for run in (1, 2, 3):
        first_requests = {}
        for row in final:
            if row["run"] == run:
                first_requests.setdefault(row["key"].rsplit("/", 1)[-1], row["request"])
        per_run.append(first_requests)
    assert len(per_run[0]) == 60 and per_run[0] == per_run[1] == per_run[2]
    pipeline.evaluate_final(root, client)
    assert len(chats) == 348 * multiplier + (8 if recover_missing else 0)
    with pytest.raises(ValueError, match="frozen"):
        pipeline.develop(root, client)
    if recover_missing:
        with pytest.raises(pipeline.IncompleteForecastError):
            pipeline.forecast(root, client)
        assert len(chats) == 421
        assert not (root / "llm_forecast.csv").exists()
        assert not list((root / "artifacts/llm").glob("october_*"))
        assert not (root / "artifacts/llm/forecast_manifest.json").exists()
        failing["october"] = False
    pipeline.forecast(root, client)
    assert len(chats) == 410 * multiplier + (12 if recover_missing else 0)
    pipeline.forecast(root, client)
    assert len(chats) == 410 * multiplier + (12 if recover_missing else 0)
    assert len(pd.read_csv(root / "llm_forecast.csv")) == 744
    assert pd.read_csv(root / "llm_forecast.csv").columns.tolist() == [
        "hour_start",
        "forecast",
        "p10",
        "p90",
    ]
    assert len(pipeline.records(root)) == 410 * multiplier + (
        12 if recover_missing else 0
    )
    assert len(pd.read_csv(root / "artifacts/llm/october_chunks.csv")) == 62
    assert (
        pipeline.read_json(root / "artifacts/llm/final/repeatability.json")[
            "monthly_total_range"
        ]
        > 0
    )
    from llm_forecasting.explain import explain_demo

    explain_demo(root, client)
    from llm_forecasting.reporting import notebook_cells, saved_report_data

    bundle = saved_report_data(root)
    assert len(bundle["comparison"]) == 8
    assert "LLM wins on 15 days" in bundle["daily_comparison_summary"]
    assert "Part 1 wins on 15 days" in bundle["daily_comparison_summary"]
    assert len(bundle["where_models_win"]) == 4
    assert bundle["coherence"].groupby("level").size().to_dict() == {
        "chunk": 7,
        "daily": 7,
    }
    if partial_recovery:
        assert bundle["coherence"].scorable_periods.eq(0).all()
        assert bundle["coherence"].coherent_share.isna().all()
        assert len(bundle["request_coherence"]) == 7
        assert bundle["request_coherence"].scorable_periods.sum() == 820
        assert any(
            "Invalid hourly reply" in example for example in bundle["failure_examples"]
        )
    assert len(bundle["failure_examples"]) >= 2
    assert len(notebook_cells(root)) >= 5
    for cell in notebook_cells(root):
        if cell.cell_type == "code":
            compile(cell.source, "LLM notebook cell", "exec")
    assert all((root / name).read_bytes() == value for name, value in preserved.items())
    # Altering raw development evidence is detected even after later appends.
    journal = root / "artifacts/llm/requests.jsonl"
    journal.write_bytes(
        journal.read_bytes().replace(b'"attempt": 1', b'"attempt": 2', 1)
    )
    with pytest.raises(ValueError, match="evidence"):
        pipeline.frozen_protocol(root)
