"""Synthetic hourly dataset shaped like data/processed/dataset.parquet. Used only by tests."""
import numpy as np
import pandas as pd

CITIES = ["los_angeles", "san_diego", "fresno", "sacramento", "bay_area"]
WEIGHTS = [0.464367, 0.152963, 0.046773, 0.073502, 0.262395]


def make_dataset(start="2021-01-01", end="2023-06-30 23:00", seed=1):
    idx = pd.date_range(start, end, freq="h", tz="UTC")
    rng = np.random.default_rng(seed)
    doy, h = idx.dayofyear.values, idx.hour.values
    base = 17 - 8 * np.cos(2 * np.pi * (doy - 15) / 365) + 6 * np.sin(2 * np.pi * (h - 16) / 24)
    shock = np.zeros(len(idx))
    for yr in (2021, 2022, 2023):                       # a few multi-day heat spells each summer
        for s0 in ("07-10", "08-05", "09-01"):
            a = pd.Timestamp(f"{yr}-{s0}", tz="UTC")
            shock += ((idx >= a) & (idx < a + pd.Timedelta(days=5))).astype(float) * 8
    ds, tot = pd.DataFrame(index=idx), 0
    for name, wt in zip(CITIES, WEIGHTS):
        t = base + rng.normal(0, 0.5, len(idx)) + shock
        ds[f"era5_temperature_2m__{name}"] = t
        ds[f"fc_temperature_2m__{name}"] = t + rng.normal(0, 1.0, len(idx))
        tot = tot + wt * t
    ds["era5_temperature_2m__ca_pw"] = tot
    fc = tot + rng.normal(0, 1.0, len(idx))
    ds["fc_temperature_2m__ca_pw"] = fc
    ds["fc_relative_humidity_2m__ca_pw"] = 50 + rng.normal(0, 3, len(idx))
    local_dow = idx.tz_convert("America/Los_Angeles").day_of_week.values
    ds["demand_mwh"] = (24000 + 750 * np.maximum(tot - 20, 0) + 3500 * np.sin(2 * np.pi * (h - 1) / 24)
                        - 1800 * (local_dow >= 5) + rng.normal(0, 250, len(idx)))
    return ds


CFG = {
    "seed": 42,
    "paths": {},
    "weather": {"locations": [{"name": n.replace("_", " ").title(), "weight": w, "lat": 35.0, "lon": -119.0}
                              for n, w in zip(CITIES, WEIGHTS)]},
    "heatwave": {"percentile": 0.90, "min_consecutive_days": 3, "headline_months": [5, 6, 7, 8, 9, 10]},
    "evaluation": {"holdout_weeks": 8, "horizon_hours": 24, "quantiles": [0.1, 0.5, 0.9],
                   "local_timezone": "America/Los_Angeles", "peak_hours_local": [16, 17, 18, 19, 20, 21],
                   "min_train_days": 365, "refit_every_days": 120},
    "features": {"demand_lags_hours": [24, 48, 72, 168, 336], "rolling_windows_hours": [24, 72, 168],
                 "degree_base_c": 18.3, "temp_bin_edges_c": [10, 15, 20, 25, 30, 35, 40],
                 "holiday_cap_days": 7},
    "models": {"lightgbm": {"n_estimators": 40, "learning_rate": 0.1, "num_leaves": 15, "min_child_samples": 20,
                            "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8},
               "ridge": {"alphas": [1.0, 10.0, 100.0]},
               "sarimax": {"window_days": 28, "order": [1, 0, 1], "fourier_harmonics": 3, "maxiter": 60,
                           "evaluate_every_n_origins": 3}},
}
