from datetime import timezone
from pathlib import Path

from fiber.io import load_csv, parse_timestamp


def test_load_mini_csv():
    rec = load_csv(Path(__file__).parent / "data" / "mini.csv")
    assert rec.n_profiles == 2
    assert rec.n_points == 3
    assert rec.n_sentinel == 2
    assert rec.valid[0].tolist() == [True, True, False]
    assert rec.raw_c[0, 0] == 20.0


def test_parse_plus_three():
    dt, missing = parse_timestamp("2026-09-14 00:03:19+03")
    assert missing is False
    utc = dt.astimezone(timezone.utc)
    assert utc.hour == 21
    assert utc.minute == 3


def test_parse_naive_marked():
    dt, missing = parse_timestamp("2025-10-16 12:00:06")
    assert missing is True
    assert dt.tzinfo is None
