import numpy as np
import pandas as pd
import pytest

from src.data import build_dataset as bd
from src.data import validate as v

LOCS = [
    {"name": "A City", "weight": 0.75},
    {"name": "B Town", "weight": 0.25},
]


def raw_demand(hours, values=None):
    t = pd.date_range("2024-01-01", periods=hours, freq="h", tz="UTC")
    vals = values if values is not None else np.full(hours, 25000.0)
    return pd.DataFrame({
        "period": t.strftime("%Y-%m-%dT%H"),
        "value": [str(x) for x in vals],
        "retrieved_at_utc": "2026-01-01",
    })


def test_duplicate_timestamps_are_reported_and_removed():
    raw = raw_demand(48)
    raw = pd.concat([raw, raw.iloc[[5]]], ignore_index=True)
    s, rep = bd.clean_demand(raw)
    assert rep["duplicate_timestamps"] == 1
    assert s.index.is_unique and len(s) == 48


def test_missing_hours_become_nan_and_are_reported():
    raw = raw_demand(48).drop(index=[10, 11, 12])
    s, rep = bd.clean_demand(raw)
    assert len(s) == 48
    assert rep["missing_hours_in_range"] == 3
    assert rep["gap_runs"][0]["hours"] == 3
    assert s.isna().sum() == 3


def test_out_of_range_values_are_nulled():
    vals = np.full(24, 25000.0)
    vals[3] = 0.0
    vals[4] = 999999.0
    s, rep = bd.clean_demand(raw_demand(24, vals))
    assert rep["out_of_range_set_to_nan"] == 2
    assert s.isna().sum() == 2


def test_index_is_utc_hourly():
    s, _ = bd.clean_demand(raw_demand(72))
    assert v.is_utc_hourly(s.index)
    assert ((s.index[1:] - s.index[:-1]) == pd.Timedelta(hours=1)).all()


def test_gap_runs_groups_consecutive_hours():
    idx = pd.DatetimeIndex(
        ["2024-01-01 01:00", "2024-01-01 02:00", "2024-01-01 06:00"], tz="UTC"
    )
    runs = v.gap_runs(idx)
    assert [r[2] for r in runs] == [2, 1]


def test_weather_weighted_index_is_correct():
    t = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    rows = []
    for loc, temp in [("A City", 10.0), ("B Town", 30.0)]:
        for ts in t:
            rows.append({"time": ts, "location": loc, "temperature_2m": temp})
    w = bd.weather_wide(pd.DataFrame(rows), ["temperature_2m"], LOCS, "era5")
    assert w["era5_temperature_2m__ca_pw"].iloc[0] == pytest.approx(0.75 * 10 + 0.25 * 30)
    assert "era5_temperature_2m__a_city" in w.columns


def test_weather_weights_must_sum_to_one():
    bad = [{"name": "A City", "weight": 0.5}, {"name": "B Town", "weight": 0.2}]
    df = pd.DataFrame({"time": pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC"),
                       "location": "A City", "temperature_2m": 1.0})
    with pytest.raises(ValueError):
        bd.weather_wide(df, ["temperature_2m"], bad, "era5")


def test_missing_location_makes_weighted_index_nan():
    t = pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC")
    df = pd.DataFrame({"time": t, "location": "A City", "temperature_2m": 10.0})
    w = bd.weather_wide(df, ["temperature_2m"], LOCS, "era5")
    assert w["era5_temperature_2m__ca_pw"].isna().all()


def test_duplicate_weather_rows_rejected():
    t = pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC")
    df = pd.DataFrame({"time": t, "location": "A City", "temperature_2m": 10.0})
    df = pd.concat([df, df])
    with pytest.raises(ValueError):
        bd.weather_wide(df, ["temperature_2m"], LOCS, "era5")


def test_circular_peak_window_wraps_midnight():
    assert v.in_circular_window(1, 23, 4)
    assert v.in_circular_window(23, 23, 4)
    assert not v.in_circular_window(12, 23, 4)


def _synthetic_dataset(shift_hours=0):
    idx = pd.date_range("2024-06-01", periods=24 * 20, freq="h", tz="UTC")
    h = idx.hour.values
    demand = 25000 + 8000 * np.cos(2 * np.pi * (h - (1 + shift_hours)) / 24)
    temp = 25 + 8 * np.cos(2 * np.pi * (h - (22 + shift_hours)) / 24)
    return pd.DataFrame(
        {"demand_mwh": demand, "era5_temperature_2m__ca_pw": temp}, index=idx
    )


def test_alignment_check_passes_on_sane_timing():
    problems = bd.run_checks(_synthetic_dataset(), cfg={})
    assert problems == []


def test_alignment_check_catches_a_shifted_series():
    problems = bd.run_checks(_synthetic_dataset(shift_hours=10), cfg={})
    assert any("peaks" in p for p in problems)


def test_eia_hour_ending_stamps_become_hour_beginning():
    raw = pd.DataFrame({"period": ["2024-01-01T01", "2024-01-01T02"],
                        "value": ["20000", "21000"], "retrieved_at_utc": "2026-01-01"})
    s, _ = bd.clean_demand(raw)
    assert s.index[0] == pd.Timestamp("2024-01-01 00:00", tz="UTC")  # hour ending 01:00 covers 00:00 to 01:00
    assert s.iloc[0] == 20000
