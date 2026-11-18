"""Regression tests for the dataset build, including the exact failure we hit on real data."""
import numpy as np
import pandas as pd

from src.data import build_dataset as bd

VARS = ["temperature_2m", "apparent_temperature", "relative_humidity_2m", "dew_point_2m",
        "wind_speed_10m", "cloud_cover", "precipitation"]
LOCS = [{"name": "A City", "lat": 34.0, "lon": -118.0, "weight": 0.6},
        {"name": "B Town", "lat": 37.0, "lon": -122.0, "weight": 0.4}]


def write_raw(tmp_path, hours=24 * 20):
    """Raw files shaped like the real ones: EIA hour-ENDING stamps starting 2021-01-01T00,
    weather hour-BEGINNING starting 2021-01-01 00:00 UTC."""
    t_eia = pd.date_range("2021-01-01 00:00", periods=hours, freq="h", tz="UTC")
    begin = t_eia - pd.Timedelta(hours=1)  # true hour-beginning of each EIA reading
    demand = 25000 + 8000 * np.cos(2 * np.pi * (begin.hour.values - 1) / 24)
    pd.DataFrame({"period": t_eia.strftime("%Y-%m-%dT%H"), "value": demand.astype(str),
                  "retrieved_at_utc": "x"}).to_parquet(tmp_path / "eia_caiso_demand.parquet")

    t_w = pd.date_range("2021-01-01 00:00", periods=hours, freq="h", tz="UTC")
    rows = []
    for loc in LOCS:
        df = pd.DataFrame({"time": t_w, "location": loc["name"]})
        for v in VARS:
            df[v] = 20 + 8 * np.cos(2 * np.pi * (t_w.hour.values - 22) / 24)
        rows.append(df)
    w = pd.concat(rows, ignore_index=True)
    w.to_parquet(tmp_path / "weather_era5.parquet")
    w.to_parquet(tmp_path / "weather_forecast_archive.parquet")


def cfg_for(tmp_path):
    return {"paths": {"raw": str(tmp_path)},
            "weather": {"variables": VARS, "locations": LOCS},
            "evaluation": {"local_timezone": "America/Los_Angeles"}}


def test_first_demand_hour_without_weather_is_trimmed_not_a_failure(tmp_path):
    write_raw(tmp_path)
    cfg = cfg_for(tmp_path)
    ds, report = bd.build(cfg)
    assert ds.index.min() == pd.Timestamp("2021-01-01 00:00", tz="UTC")   # weather start, not 2020-12-31 23:00
    assert report["hours_trimmed_no_weather"] == ["2020-12-31 23:00:00+00:00"]
    assert bd.run_checks(ds, cfg) == []


def test_missing_weather_in_the_middle_is_still_a_failure(tmp_path):
    write_raw(tmp_path)
    cfg = cfg_for(tmp_path)
    ds, _ = bd.build(cfg)
    ds.loc[ds.index[100], "era5_temperature_2m__ca_pw"] = np.nan
    problems = bd.run_checks(ds, cfg)
    assert len(problems) == 1 and "1 hours have missing weather" in problems[0]


def test_missing_weather_message_is_compact(tmp_path):
    write_raw(tmp_path)
    cfg = cfg_for(tmp_path)
    ds, _ = bd.build(cfg)
    ds.loc[ds.index[100], [c for c in ds.columns if c.startswith("fc_")]] = np.nan
    msg = bd.run_checks(ds, cfg)[0]
    assert len(msg) < 300  # not an 84-column dump


def test_trim_only_touches_the_edges():
    idx = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")
    ds = pd.DataFrame({"demand_mwh": 1.0, "era5_x": 1.0, "fc_x": 1.0}, index=idx)
    ds.iloc[0, 1:] = np.nan      # leading edge: no weather
    ds.iloc[-1, 1:] = np.nan     # trailing edge: no weather
    ds.iloc[5, 1] = np.nan       # interior hole: must survive
    out, trimmed = bd.trim_to_weather_coverage(ds)
    assert len(out) == 8 and len(trimmed) == 2
    assert out["era5_x"].isna().sum() == 1
