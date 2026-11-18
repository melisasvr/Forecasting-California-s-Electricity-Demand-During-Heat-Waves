"""Pre-specified heat-wave labeling.

The rule lives in configs/config.yaml and must NOT be tuned after seeing model errors.

  daily index  = population-weighted mean of each location's daily MAX temperature
                 (local calendar day, ERA5 reanalysis)
  hot day      = index above the chosen percentile of the index for that calendar month
                 (percentile taken over all years; used for labeling and reporting only,
                 never as a model feature)
  heat wave    = at least `min_consecutive_days` calendar-consecutive hot days

Reporting split (decided before any model errors were computed): a month-relative rule
also flags unusually warm winter and early-spring spells. Headline "cooling-season
heat-wave days" are heat-wave days that fall in `headline_months` (May to October).
Everything else is labeled an off-season warm spell: still reported, never in the
headline numbers.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.build_dataset import slug

DEFAULT_HEADLINE_MONTHS = (5, 6, 7, 8, 9, 10)
DAY_TYPE_LABELS = (
    "Cooling-season heat-wave day",
    "Off-season warm spell day",
    "Hot day (not part of a 3+ day run)",
    "Normal day",
)


def day_types(labels):
    """One category per labeled day, shared by the dashboard, tables and plots."""
    return pd.Series(
        np.select(
            [labels["cooling_season_heat_wave_day"], labels["heat_wave_day"], labels["hot_day"]],
            list(DAY_TYPE_LABELS[:3]), default=DAY_TYPE_LABELS[3],
        ),
        index=labels.index, name="day_type",
    )


def daily_temperature_index(ds, locations, tz, source="era5", min_hours=23):
    """Population-weighted daily max temperature, indexed by naive local date."""
    cols = {loc["name"]: f"{source}_temperature_2m__{slug(loc['name'])}" for loc in locations}
    temps = ds[list(cols.values())]
    local_date = ds.index.tz_convert(tz).normalize().tz_localize(None)
    g = temps.groupby(local_date)
    daily_max = g.max()
    hours = g.count().min(axis=1)
    weights = pd.Series({cols[loc["name"]]: loc["weight"] for loc in locations})
    idx = (daily_max * weights).sum(axis=1, min_count=len(weights))
    idx = idx[hours >= min_hours]  # drop partial days at the edges of the data
    idx.name = "temp_index_c"
    idx.index.name = "local_date"
    return idx


def label_heat_waves(index_c, percentile, min_days, headline_months=DEFAULT_HEADLINE_MONTHS):
    s = index_c.dropna().sort_index()
    month = s.index.month
    thr = s.groupby(month).quantile(percentile)
    threshold = pd.Series(thr.reindex(month).to_numpy(), index=s.index)
    hot = s > threshold

    # a new run starts when hot/not-hot flips or when a calendar day is missing
    gap = s.index.to_series().diff() != pd.Timedelta(days=1)
    run_id = (hot.ne(hot.shift(fill_value=False)) | gap).cumsum()
    run_len = hot.groupby(run_id).transform("size")
    wave = hot & (run_len >= min_days)

    wave_id = pd.Series(pd.NA, index=s.index, dtype="Int64")
    codes, _ = pd.factorize(run_id[wave])
    wave_id[wave] = codes + 1
    out = pd.DataFrame({
        "temp_index_c": s,
        "threshold_c": threshold,
        "hot_day": hot,
        "heat_wave_day": wave,
        "wave_id": wave_id,
    })
    out["cooling_season_heat_wave_day"] = (
        out["heat_wave_day"] & out.index.month.isin(list(headline_months))
    )
    return out


def summarize_waves(labels):
    rows = []
    w = labels[labels["wave_id"].notna()]
    for wid, g in w.groupby("wave_id"):
        n_in = int(g["cooling_season_heat_wave_day"].sum())
        if n_in == len(g):
            kind = "cooling-season"
        elif n_in == 0:
            kind = "off-season"
        else:
            kind = "straddles season edge"
        rows.append({
            "wave_id": int(wid),
            "start": g.index.min(),
            "end": g.index.max(),
            "days": len(g),
            "peak_temp_index_c": g["temp_index_c"].max(),
            "kind": kind,
        })
    return pd.DataFrame(rows, columns=["wave_id", "start", "end", "days", "peak_temp_index_c", "kind"])


def main():
    cfg = yaml.safe_load(open("configs/config.yaml"))
    proc = Path(cfg["paths"]["processed"])
    ds = pd.read_parquet(proc / "dataset.parquet")
    h = cfg["heatwave"]
    idx = daily_temperature_index(ds, cfg["weather"]["locations"], cfg["evaluation"]["local_timezone"])
    labels = label_heat_waves(idx, h["percentile"], h["min_consecutive_days"], h["headline_months"])
    labels.to_parquet(proc / "heatwave_labels.parquet")
    waves = summarize_waves(labels)
    print(f"days labeled: {len(labels)}")
    print(waves["kind"].value_counts().to_string())
    print(f"cooling-season heat-wave days: {int(labels['cooling_season_heat_wave_day'].sum())}")
    print(f"Saved {proc / 'heatwave_labels.parquet'}")


if __name__ == "__main__":
    main()
