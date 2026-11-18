"""Build the forecast feature matrix: one row per (forecast origin, horizon step).

Protocol (setup 1): a forecast is issued at local midnight (America/Los_Angeles).
Demand is known up to the hour just before the origin. The model forecasts the
24 hours that follow. Weather for the target hours is the ARCHIVED FORECAST
("fc"), never ERA5, unless --weather era5 is passed for the clearly labeled
perfect-weather experiment.

Every demand feature is built from timestamps <= origin - 1h.
"""
import argparse
import json
from pathlib import Path

import holidays
import numpy as np
import pandas as pd
import yaml

META_COLS = ["origin_utc", "target_utc", "target_local_date", "y"]


def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def _h(n):
    return pd.Timedelta(hours=n)


def make_origins(index, tz, max_history_hours, horizon):
    """Local-midnight origins, as UTC timestamps, with enough history and a full target window."""
    first = index.min() + _h(max_history_hours + 1)
    last = index.max() - _h(horizon - 1)
    first_local = first.tz_convert(tz)
    start = first_local.normalize()
    if start < first_local:
        start = start + pd.DateOffset(days=1)
    end = last.tz_convert(tz).normalize()
    origins = pd.date_range(start, end, freq="D").tz_convert("UTC")
    return origins[(origins >= first) & (origins <= last)]


def holiday_features(local_dates, cap):
    """Is-holiday and signed distance to nearest holidays (US federal + California)."""
    years = range(min(d.year for d in local_dates) - 1, max(d.year for d in local_dates) + 2)
    hol = holidays.country_holidays("US", subdiv="CA", years=years)
    hd = np.array(sorted(d.toordinal() for d in hol.keys()))
    d = np.array([x.toordinal() for x in local_dates])
    nxt = hd[np.searchsorted(hd, d, side="left")] - d
    prv = d - hd[np.searchsorted(hd, d, side="right") - 1]
    return (nxt == 0).astype(int), np.minimum(nxt, cap), np.minimum(prv, cap)


def build_features(ds, cfg, weather="fc", require_target=True):
    fcfg, tz = cfg["features"], cfg["evaluation"]["local_timezone"]
    H = cfg["evaluation"]["horizon_hours"]
    lags, wins = fcfg["demand_lags_hours"], fcfg["rolling_windows_hours"]
    if min(lags) < H:
        raise ValueError("a demand lag shorter than the horizon would use unavailable data")

    demand = ds["demand_mwh"]
    prefix = f"{weather}_"
    wcols = [c for c in ds.columns if c.startswith(prefix)]
    T = ds[f"{prefix}temperature_2m__ca_pw"]
    origins = make_origins(ds.index, tz, max(max(lags), max(wins)), H)
    last_obs = origins - _h(1)

    # features that depend only on the origin
    org = pd.DataFrame(index=range(len(origins)))
    for w in wins:
        r = demand.rolling(w, min_periods=w)
        org[f"demand_roll_mean_{w}h"] = r.mean().reindex(last_obs).to_numpy()
        org[f"demand_roll_std_{w}h"] = r.std().reindex(last_obs).to_numpy()
    r24max = T.rolling(24, min_periods=24).max()
    org["temp_pw_prev_day_max"] = r24max.reindex(last_obs).to_numpy()
    org["temp_pw_prev_3day_mean_max"] = np.mean(
        [r24max.reindex(last_obs - _h(24 * i)).to_numpy() for i in range(3)], axis=0
    )
    # forecast values for the target day: known at the origin, so legal
    org["temp_pw_target_window_max"] = (
        T.rolling(H, min_periods=H).max().shift(-(H - 1)).reindex(origins).to_numpy()
    )
    org["temp_pw_target_window_mean"] = (
        T.rolling(H, min_periods=H).mean().shift(-(H - 1)).reindex(origins).to_numpy()
    )

    frames = []
    for k in range(H):
        t = origins + _h(k)
        local = t.tz_convert(tz)
        df = pd.DataFrame({
            "origin_utc": pd.Series(origins),
            "target_utc": pd.Series(t),
            "target_local_date": pd.Series(local.normalize().tz_localize(None)),
            "y": demand.reindex(t).to_numpy(),
            "horizon_step": k,
        })
        for L in lags:
            df[f"demand_lag_{L}h"] = demand.reindex(t - _h(L)).to_numpy()

        hour = local.hour.to_numpy()
        dow = local.day_of_week.to_numpy()
        df["hour"], df["dow"], df["month"] = hour, dow, local.month.to_numpy()
        df["is_weekend"] = (dow >= 5).astype(int)
        hol, nxt, prv = holiday_features(list(local.date), fcfg["holiday_cap_days"])
        df["is_holiday"], df["days_to_next_holiday"], df["days_since_prev_holiday"] = hol, nxt, prv

        W = ds[wcols].reindex(t).reset_index(drop=True)
        df = pd.concat([df, W], axis=1)

        Tt = T.reindex(t).to_numpy()
        df["cdh"] = np.maximum(Tt - fcfg["degree_base_c"], 0)
        df["hdh"] = np.maximum(fcfg["degree_base_c"] - Tt, 0)
        df["temp_bin"] = np.digitize(Tt, fcfg["temp_bin_edges_c"])
        df["temp_x_hour_sin"] = Tt * np.sin(2 * np.pi * hour / 24)
        df["temp_x_hour_cos"] = Tt * np.cos(2 * np.pi * hour / 24)
        df["temp_pw_lag24h"] = T.reindex(t - _h(24)).to_numpy()

        frames.append(pd.concat([df, org], axis=1))

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["origin_utc", "horizon_step"], kind="stable").reset_index(drop=True)
    feature_cols = [c for c in out.columns if c not in META_COLS]
    before = len(out)
    out = out.dropna(subset=feature_cols + (["y"] if require_target else [])).reset_index(drop=True)
    info = {
        "weather_source": weather,
        "origins": int(len(origins)),
        "rows_before_dropping_nan": int(before),
        "rows_after": int(len(out)),
        "rows_dropped_for_missing_values": int(before - len(out)),
        "first_origin": str(out["origin_utc"].min()),
        "last_origin": str(out["origin_utc"].max()),
        "n_features": len(feature_cols),
    }
    return out, feature_cols, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weather", choices=["fc", "era5"], default="fc")
    args = ap.parse_args()
    cfg = load_config()
    ds = pd.read_parquet(Path(cfg["paths"]["processed"]) / "dataset.parquet")
    out, feature_cols, info = build_features(ds, cfg, weather=args.weather)
    proc = Path(cfg["paths"]["processed"])
    if args.weather == "fc":
        out.to_parquet(proc / "features.parquet", index=False)
        with open(proc / "feature_list.json", "w") as f:
            json.dump({"target": "y", "weather_source": "fc (archived forecasts)",
                       "features": feature_cols}, f, indent=2)
    else:
        out.to_parquet(proc / "features_PERFECT_WEATHER_experiment.parquet", index=False)
        print("NOTE: perfect-weather build. Not an operational forecast.")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
