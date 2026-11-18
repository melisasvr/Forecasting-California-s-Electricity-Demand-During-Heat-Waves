"""Download hourly weather for the configured California locations.

Two raw sources are kept separate on purpose:
  era5             : gridded reanalysis (not station observations)
  forecast_archive : archived NWP forecasts, the realistic stand-in for
                     "forecast weather" in backtests

All timestamps are requested in UTC to match EIA.
"""
import time
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import yaml


def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def fetch_location(url, loc, variables, start, end, tz="UTC"):
    params = {
        "latitude": loc["lat"],
        "longitude": loc["lon"],
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(variables),
        "timezone": tz,
    }
    r = requests.get(url, params=params, timeout=120)
    r.raise_for_status()
    h = r.json()["hourly"]
    df = pd.DataFrame(h)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["location"] = loc["name"]
    return df


def download(source_name, url, cfg, end):
    frames = []
    years = range(int(cfg["weather"]["start"][:4]), int(end[:4]) + 1)
    for loc in cfg["weather"]["locations"]:
        for y in years:
            s = max(f"{y}-01-01", cfg["weather"]["start"])
            e = min(f"{y}-12-31", end)
            try:
                df = fetch_location(url, loc, cfg["weather"]["variables"], s, e)
                frames.append(df)
                print(f"  {source_name} {loc['name']} {y}: {len(df)} rows")
            except requests.HTTPError as exc:
                # e.g. the archive may not reach back this far. Log it, keep going.
                print(f"  {source_name} {loc['name']} {y}: FAILED ({exc})")
            time.sleep(1.0)
    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True)
    out["source"] = source_name
    out["retrieved_at_utc"] = datetime.now(timezone.utc).isoformat()
    return out


def main():
    cfg = load_config()
    raw = Path(cfg["paths"]["raw"])
    end = (date.today()).isoformat()
    for name, key in [("era5", "era5_url"), ("forecast_archive", "forecast_archive_url")]:
        df = download(name, cfg["weather"][key], cfg, end)
        if df is None:
            print(f"No data for {name}")
            continue
        var = cfg["weather"]["variables"][0]
        first_valid = df.loc[df[var].notna(), "time"].min()
        print(f"{name}: earliest non-null {var} = {first_valid}")
        df.to_parquet(raw / f"weather_{name}.parquet", index=False)


if __name__ == "__main__":
    main()
