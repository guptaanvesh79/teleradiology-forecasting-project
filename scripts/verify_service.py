"""Check the running Docker API against every saved forecast row."""

import csv
import json
import math
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "http://127.0.0.1:8000"


def get(path):
    with urlopen(BASE_URL + path, timeout=10) as response:
        return json.load(response)


def read_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def compare_rows(actual, expected, key):
    assert len(actual) == len(expected), (len(actual), len(expected))
    for received, saved in zip(actual, expected, strict=True):
        assert received[key].replace("T", " ") == saved[key]
        for field in ("forecast", "p10", "p90"):
            assert math.isclose(received[field], float(saved[field]), rel_tol=0, abs_tol=1e-8), (key, field)


def main():
    assert get("/health") == {"status": "ready", "hours": 744, "days": 31}
    hourly = get("/forecast/hourly")
    daily = get("/forecast/daily")
    compare_rows(hourly["data"], read_rows(ROOT / "forecast.csv"), "hour_start")
    compare_rows(daily["data"], read_rows(ROOT / "artifacts/daily_forecast.csv"), "date")
    metadata = json.loads((ROOT / "artifacts/forecast_metadata.json").read_text())
    assert hourly["metadata"]["model"] == metadata["model"]
    assert hourly["metadata"]["timezone"] == "unspecified"
    assert math.isclose(
        math.fsum(row["forecast"] for row in hourly["data"]),
        math.fsum(row["forecast"] for row in daily["data"]),
        rel_tol=0, abs_tol=1e-6,
    )
    assert len(get("/forecast/hourly?start_date=2026-10-25&end_date=2026-10-31")["data"]) == 168
    for path, expected_status in (
        ("/forecast/hourly?start_date=invalid", 422),
        ("/forecast/daily?start_date=2026-10-20&end_date=2026-10-10", 422),
        ("/forecast/hourly?end_date=2026-11-01", 404),
    ):
        try:
            get(path)
        except HTTPError as error:
            assert error.code == expected_status, (path, error.code)
        else:
            raise AssertionError(f"Expected HTTP {expected_status}: {path}")
    print("PASS: live API matches all 744 hourly and 31 daily rows, totals, metadata, filters, and errors")


if __name__ == "__main__":
    main()
