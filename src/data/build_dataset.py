"""Clean raw EIA + weather files and build one UTC-indexed hourly dataset.

Inputs  (data/raw):       eia_caiso_demand.parquet, weather_era5.parquet,
                          weather_forecast_archive.parquet
Outputs (data/processed): dataset.parquet
        (reports):        data_quality.json

Everything stays in UTC. A Pacific-time column is added only for convenience.
Demand gaps are reported, NOT filled. Later steps drop forecast origins whose
inputs or targets are missing.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data import validate as v

PLAUSIBLE_DEMAND_MWH = (8_000, 60_000)  # CAISO hourly demand sits well inside this


def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def slug(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def clean_demand(raw, lo=PLAUSIBLE_DEMAND_MWH[0], hi=PLAUSIBLE_DEMAND_MWH[1]):
    df = raw.copy()
    # EIA-930 stamps each reading with the END of its hour (hourly integrated MW).
    # Shift back one hour so the index is the hour BEGINNING. Then a local calendar day
    # is exactly hours 00:00 to 23:00, and demand lines up with the weather timestamps.
    df["time"] = pd.to_datetime(df["period"], format="%Y-%m-%dT%H", utc=True) - pd.Timedelta(hours=1)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    report = {"raw_rows": int(len(df)), "timestamp_convention": "EIA hour-ending shifted to hour-beginning (UTC)"}

    dup = v.find_duplicates(df, ["time"])
    report["duplicate_timestamps"] = int(dup["time"].nunique())
    # keep the most recently retrieved row for any duplicated timestamp
    df = df.sort_values("time", kind="stable").drop_duplicates("time", keep="last")

    s = df.set_index("time")["value"].astype(float)
    report["null_or_non_numeric"] = int(s.isna().sum())
    bad = v.out_of_range(s, lo, hi)
    report["out_of_range_set_to_nan"] = int(bad.sum())
    s[bad] = np.nan

    full = pd.date_range(s.index.min(), s.index.max(), freq="h")
    missing = full.difference(s.index)
    report["missing_hours_in_range"] = int(len(missing))
    report["gap_runs"] = [
        {"start": str(a), "end": str(b), "hours": int(n)} for a, b, n in v.gap_runs(missing)
    ]
    s = s.reindex(full)
    s.name = "demand_mwh"
    report["demand_nan_hours_total"] = int(s.isna().sum())
    report["demand_first"] = str(s.index.min())
    report["demand_last"] = str(s.index.max())
    return s, report


def weather_wide(raw, variables, locations, prefix):
    """One row per UTC hour; per-location columns plus a population-weighted index."""
    df = raw.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    if df.duplicated(["time", "location"]).any():
        raise ValueError(f"{prefix}: duplicate (time, location) rows")

    weights = pd.Series({loc["name"]: loc["weight"] for loc in locations})
    if abs(weights.sum() - 1.0) > 1e-6:
        raise ValueError(f"population weights sum to {weights.sum()}, expected 1.0")
    unknown = set(df["location"]) - set(weights.index)
    if unknown:
        raise ValueError(f"{prefix}: unknown locations {unknown}")

    cols = {}
    for var in variables:
        p = df.pivot(index="time", columns="location", values=var).reindex(columns=list(weights.index))
        for loc in p.columns:
            cols[f"{prefix}_{var}__{slug(loc)}"] = p[loc]
        # NaN if any location is missing that hour, so a weighted index never hides a hole
        cols[f"{prefix}_{var}__ca_pw"] = p.mul(weights, axis=1).sum(axis=1, min_count=p.shape[1])
    return pd.DataFrame(cols)


def trim_to_weather_coverage(ds):
    """Drop leading/trailing hours where ALL weather is missing.

    Demand can start or end an hour outside the weather range (for example, shifting EIA's
    hour-ending stamps back one hour puts the first demand hour just before the weather
    starts). Gaps in the MIDDLE are deliberately left alone so run_checks can flag them.
    """
    wcols = [c for c in ds.columns if c.startswith(("era5_", "fc_"))]
    has_weather = ds[wcols].notna().any(axis=1)
    if not has_weather.any():
        return ds, ds.index
    first, last = has_weather[has_weather].index[[0, -1]]
    keep = (ds.index >= first) & (ds.index <= last)
    return ds[keep], ds.index[~keep]


def run_checks(ds, cfg):
    """Return a list of failure messages. Empty list means all checks passed."""
    problems = []
    if not v.is_utc_hourly(ds.index):
        problems.append("index is not a unique, increasing, tz-aware UTC index")

    weather_cols = [c for c in ds.columns if c.startswith(("era5_", "fc_"))]
    missing = ds[weather_cols].isna()
    bad_hours = missing.any(axis=1)
    if bad_hours.any():
        n_cols = int(missing.any(axis=0).sum())
        sample = [str(t) for t in ds.index[bad_hours][:3]]
        problems.append(
            f"{int(bad_hours.sum())} hours have missing weather ({n_cols} columns affected), "
            f"first few: {sample}"
        )

    inf_cols = [c for c in ds.columns if ds[c].dtype.kind == "f" and np.isinf(ds[c]).any()]
    if inf_cols:
        problems.append(f"infinite values in {inf_cols}")

    # Timezone alignment: both series must peak at physically sensible UTC hours.
    temp = ds["era5_temperature_2m__ca_pw"].dropna()
    t_peak = v.hour_of_peak(temp)
    if not v.in_circular_window(t_peak, 21, 2):
        problems.append(f"temperature peaks at {t_peak}:00 UTC, expected 21:00 to 02:00")
    d_peak = v.hour_of_peak(ds["demand_mwh"].dropna())
    if not v.in_circular_window(d_peak, 23, 4):
        problems.append(f"demand peaks at {d_peak}:00 UTC, expected 23:00 to 04:00")
    return problems


def build(cfg):
    raw = Path(cfg["paths"]["raw"])
    wcfg = cfg["weather"]
    demand, report = clean_demand(pd.read_parquet(raw / "eia_caiso_demand.parquet"))
    era5 = weather_wide(
        pd.read_parquet(raw / "weather_era5.parquet"), wcfg["variables"], wcfg["locations"], "era5"
    )
    fc = weather_wide(
        pd.read_parquet(raw / "weather_forecast_archive.parquet"),
        wcfg["variables"], wcfg["locations"], "fc",
    )
    ds = pd.concat([demand, era5.reindex(demand.index), fc.reindex(demand.index)], axis=1)
    ds, trimmed = trim_to_weather_coverage(ds)
    report["hours_trimmed_no_weather"] = [str(t) for t in trimmed]
    report["dataset_first"] = str(ds.index.min())
    report["dataset_last"] = str(ds.index.max())
    ds.index.name = "time_utc"
    ds["time_local"] = ds.index.tz_convert(cfg["evaluation"]["local_timezone"])
    return ds, report


def main():
    cfg = load_config()
    ds, report = build(cfg)
    problems = run_checks(ds, cfg)
    report["checks_failed"] = problems
    Path("reports").mkdir(exist_ok=True)
    with open("reports/data_quality.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v_ for k, v_ in report.items() if k != "gap_runs"}, indent=2))
    print(f"gap runs: {len(report['gap_runs'])} (details in reports/data_quality.json)")
    out = Path(cfg["paths"]["processed"]) / "dataset.parquet"
    if problems:
        print("\nCHECKS FAILED:")
        for p in problems:
            print(" -", p)
        if out.exists():
            out.unlink()
            print("\nRemoved the old dataset.parquet so nothing downstream silently uses stale data.")
        sys.exit(1)
    ds.to_parquet(out)
    print(f"\nAll checks passed. Saved {ds.shape} to {out}")


if __name__ == "__main__":
    main()
