"""Origin-safe summaries and strict numeric forecast contracts."""

import json
from dataclasses import dataclass
import math
from typing import Annotated, Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from forecasting.artifacts import validate_bounds
from forecasting.data import day_audit

CHUNK_HOURS = 12
COHERENCE_TOLERANCE = 1e-6
HOUR_LABEL_MAPPING = {f"hr{hour + 1}": hour for hour in range(24)}
SCHEMA_POLICY = {
    "exact_requested_hour_keys": True,
    "all_keys_required": True,
    "additional_keys_allowed": False,
    "key_order_significant": False,
    "retain_valid_hours": True,
    "retry_only_remaining_hours": True,
    "missing_keys_are_partial": True,
}

SYSTEM_PROMPT = """You forecast recorded teleradiology study arrivals from synthetic history.
Use ONLY the supplied pre-origin summaries and requested future calendar.
Recorded zero counts do not prove zero underlying demand; sparse days are not proven outages.
Do not invent holidays, timezones, staffing, hospitals or future observations.
Predict every requested hour directly. Do not extrapolate from actual counts inside the horizon.
Return only JSON matching the schema: chunk_total and an hours object with the requested keys.
An initial request contains 12 keys; a retry may contain fewer. Return predictions ONLY for
response_hour_keys, even when the calendar gives other hours as context.
Copy ONLY the supplied response_hour_keys into hours. hr1 means 00:00, hr12 means 11:00,
hr13 means 12:00, and hr24 means 23:00. Do not restart numbering in the afternoon.
Use the requested date and weekday to choose historical patterns. JSON key order does not matter.
Each hourly array is [p10,p50,p90]: finite nonnegative numbers, with p10 <= p50 <= p90.
chunk_total is your stated sum of p50 values for ONLY the hours requested in this call.
Do not output timestamps or additional fields.
Hourly quantiles are marginal; their sums are not daily quantiles."""

Number = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Triple = Annotated[list[Number], Field(min_length=3, max_length=3)]


class ForecastReply(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    chunk_total: Number
    hours: Annotated[dict[str, Triple], Field(min_length=1, max_length=CHUNK_HOURS)]

    @model_validator(mode="after")
    def ordered_quantiles(self):
        if any(not low <= middle <= high for low, middle, high in self.hours.values()):
            raise ValueError("Require p10 <= p50 <= p90 for every hour")
        return self


class ForecastEnvelope(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    chunk_total: Number
    hours: Annotated[dict[str, Any], Field(max_length=CHUNK_HOURS)]


@dataclass
class RecoveredForecast:
    hours: dict[str, list[float]]
    chunk_total: float | None


TRIPLE_VALIDATOR = TypeAdapter(Triple)


def strict_json(text):
    def constant(value):
        raise ValueError(f"Non-JSON numeric constant: {value}")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(text, parse_constant=constant, object_pairs_hook=unique)


def response_schema_template():
    return ForecastReply.model_json_schema()


def forecast_hour_keys(hours):
    if (
        not isinstance(hours, pd.DatetimeIndex)
        or hours.tz is not None
        or not 1 <= len(hours) <= CHUNK_HOURS
        or hours.hasnans
        or not hours.is_unique
        or not hours.is_monotonic_increasing
        or not hours.equals(hours.floor("h"))
        or hours[0].normalize() != hours[-1].normalize()
        or hours[0].hour // CHUNK_HOURS != hours[-1].hour // CHUNK_HOURS
    ):
        raise ValueError(
            "Require one to twelve distinct ordered hours within one half-day"
        )
    return [f"hr{hour + 1}" for hour in hours.hour]


def forecast_schema(hours):
    keys = forecast_hour_keys(hours)
    schema = response_schema_template()
    triple = schema["properties"]["hours"]["additionalProperties"]
    schema["properties"]["hours"] = {
        "type": "object",
        "properties": {key: triple for key in keys},
        "required": keys,
        "additionalProperties": False,
    }
    return schema


def prediction_values(reply, hours):
    keys = forecast_hour_keys(hours)
    if set(reply.hours) != set(keys):
        raise ValueError(f"Require exactly the requested hour keys: {', '.join(keys)}")
    return [reply.hours[key] for key in keys]


def parse_forecast(text, hours):
    reply = ForecastReply.model_validate(strict_json(text))
    prediction_values(reply, hours)
    return reply


def inspect_forecast(text, hours):
    """Validate independent hours without repairing or losing valid neighbours."""
    keys = forecast_hour_keys(hours)
    envelope = ForecastEnvelope.model_validate(strict_json(text))
    if set(envelope.hours) - set(keys):
        raise ValueError("Response contains unrequested hour keys")
    accepted, failures, medians = {}, {}, {}
    for key in keys:
        if key not in envelope.hours:
            failures[key] = "Missing hour"
            continue
        try:
            triple = TRIPLE_VALIDATOR.validate_python(envelope.hours[key], strict=True)
            medians[key] = triple[1]
            if not triple[0] <= triple[1] <= triple[2]:
                raise ValueError("Require p10 <= p50 <= p90")
        except ValueError as error:
            failures[key] = str(error)
        else:
            accepted[key] = triple
    total = sum(medians.values()) if len(medians) == len(keys) else None
    if total is not None and not math.isfinite(total):
        total = None
    discrepancy = envelope.chunk_total - total if total is not None else None
    return {
        "requested_hour_keys": keys,
        "accepted_hours": accepted,
        "failed_hours": failures,
        "stated_total": envelope.chunk_total,
        "hourly_p50_total": total,
        "coherence_error": discrepancy,
        "coherent": (
            abs(discrepancy) <= COHERENCE_TOLERANCE if discrepancy is not None else None
        ),
    }


def history_summary(series, origin):
    origin = pd.Timestamp(origin)
    if origin.tz is not None or origin != origin.normalize():
        raise ValueError("Origin must be a timezone-naive midnight")
    if not isinstance(series.index, pd.DatetimeIndex) or series.index.tz is not None:
        raise ValueError("History must have timezone-naive hourly timestamps")
    history = series.loc[series.index < origin].copy()
    if len(history) < 168:
        raise ValueError("Require at least seven complete pre-origin days")
    expected = pd.date_range(history.index[0], origin, freq="h", inclusive="left")
    if history.index[0] != history.index[0].normalize() or not history.index.equals(
        expected
    ):
        raise ValueError(
            "History must be complete, ordered hours ending before the origin"
        )
    values = history.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Historical counts must be finite and nonnegative")
    table = history.to_frame("count")
    table["weekday"] = table.index.day_of_week
    table["hour"] = table.index.hour
    profiles = []
    for (weekday, hour), group in table.groupby(["weekday", "hour"]):
        counts = group["count"]
        profiles.append(
            [
                int(weekday),
                int(hour),
                round(float(counts.mean()), 2),
                round(float(counts.median()), 2),
                round(float(counts.quantile(0.1)), 2),
                round(float(counts.quantile(0.9)), 2),
                len(counts),
            ]
        )
    daily = history.resample("D").sum()
    weekday_totals = [
        [int(weekday), round(float(counts.mean()), 2), round(float(counts.median()), 2)]
        for weekday, counts in daily.groupby(daily.index.day_of_week)
    ]
    audit = day_audit(history)
    return {
        "history_start": str(history.index[0]),
        "history_end": str(history.index[-1]),
        "timezone": "unspecified",
        "weekday_definition": "Monday=0, Sunday=6",
        "profile_columns": ["weekday", "hour", "mean", "median", "p10", "p90", "n"],
        "weekday_hour_profiles": profiles,
        "recent_28_daily_totals": [
            [str(day.date()), float(total)] for day, total in daily.tail(28).items()
        ],
        "daily_weekday_columns": ["weekday", "mean", "median"],
        "daily_totals_by_weekday": weekday_totals,
        "empty_hours": int(history.eq(0).sum()),
        "empty_days": int(daily.eq(0).sum()),
        "sparse_day_rule": "below 10% of the historical weekday median; diagnostic only",
        "sparse_days": [
            [str(day.date()), float(row.recorded_total)]
            for day, row in audit.loc[audit.near_empty].iterrows()
        ],
        "unusual_days_removed": False,
    }


def forecast_chunks(origin, days):
    origin = pd.Timestamp(origin)
    if (
        origin.tz is not None
        or origin != origin.normalize()
        or isinstance(days, bool)
        or not isinstance(days, int)
        or days <= 0
    ):
        raise ValueError("Require a timezone-naive midnight and positive day count")
    for offset in range(0, days * 24, CHUNK_HOURS):
        start = origin + pd.Timedelta(hours=offset)
        yield pd.date_range(start, periods=CHUNK_HOURS, freq="h", name="hour_start")


def validate_chunk_index(hours):
    if (
        not isinstance(hours, pd.DatetimeIndex)
        or hours.tz is not None
        or len(hours) != CHUNK_HOURS
        or hours[0].hour not in (0, 12)
        or hours[0] != hours[0].floor("h")
        or not hours.equals(pd.date_range(hours[0], periods=CHUNK_HOURS, freq="h"))
    ):
        raise ValueError(
            "Require exactly 12 consecutive naive hours starting at midnight or noon"
        )


def forecast_messages(summary, hours, requested_hours=None):
    validate_chunk_index(hours)
    requested_hours = hours if requested_hours is None else requested_hours
    keys = forecast_hour_keys(requested_hours)
    if not set(requested_hours) <= set(hours):
        raise ValueError("Requested retry hours must belong to the original chunk")
    calendar = {
        "date": str(hours[0].date()),
        "weekday": int(hours[0].day_of_week),
        "hours": hours.hour.tolist(),
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "history": summary,
                    "calendar": calendar,
                    "response_hour_keys": keys,
                },
                separators=(",", ":"),
                allow_nan=False,
            ),
        },
    ]


def frames_from_reply(reply, hours):
    validate_chunk_index(hours)
    hourly = pd.DataFrame(
        prediction_values(reply, hours), columns=["p10", "forecast", "p90"], index=hours
    )
    hourly = hourly[["forecast", "p10", "p90"]]
    validate_bounds(hourly)
    chunks = pd.DataFrame(
        {
            "chunk_end_exclusive": [hours[-1] + pd.Timedelta(hours=1)],
            "forecast": [float(hourly.forecast.sum())],
            "stated_total": [
                np.nan if reply.chunk_total is None else reply.chunk_total
            ],
        },
        index=pd.DatetimeIndex([hours[0]], name="chunk_start"),
    )
    chunks["coherence_error"] = chunks.stated_total - chunks.forecast
    chunks["coherent"] = (chunks.coherence_error.abs() <= COHERENCE_TOLERANCE).where(
        chunks.coherence_error.notna()
    )
    return hourly, chunks


def daily_from_chunks(hourly, chunks):
    index = hourly.index
    if (
        index.empty
        or index.tz is not None
        or len(index) % 24
        or index[0] != index[0].normalize()
        or not index.equals(pd.date_range(index[0], periods=len(index), freq="h"))
        or not chunks.index.equals(index[::CHUNK_HOURS])
        or not pd.DatetimeIndex(chunks.chunk_end_exclusive).equals(
            chunks.index + pd.Timedelta(hours=CHUNK_HOURS)
        )
    ):
        raise ValueError(
            "Require complete ordered days with exactly two matching 12-hour chunks each"
        )
    expected_totals = hourly.forecast.resample(f"{CHUNK_HOURS}h").sum()
    if not np.allclose(
        chunks.forecast, expected_totals, atol=COHERENCE_TOLERANCE, rtol=0
    ):
        raise ValueError("Chunk point totals differ from hourly sums")
    daily = hourly.forecast.resample("D").sum().to_frame().rename_axis("date")
    daily["stated_total"] = chunks.stated_total.resample("D").sum(min_count=2)
    daily["coherence_error"] = daily.stated_total - daily.forecast
    daily["coherent"] = (daily.coherence_error.abs() <= COHERENCE_TOLERANCE).where(
        daily.coherence_error.notna()
    )
    return daily
