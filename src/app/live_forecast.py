"""Live demo: forecast the current local day with LIVE forecast weather.

Weather comes from the Open-Meteo Forecast API (not ERA5 reanalysis). Demand comes from a fresh EIA download
of the last ~25 days. The model is the LightGBM trained on all available history (models/lightgbm_final.joblib).
The forecast origin is the most recent local midnight for which demand through the hour before it is available.
"""
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import pandas as pd
import requests

from src.data import build_dataset as bd
from src.data.download_eia import fetch_demand
from src.features.build_features import build_features


def fetch_live_weather(cfg, past_days=14, forecast_days=3, get=requests.get):
    frames = []
    for loc in cfg["weather"]["locations"]:
        r = get(cfg["weather"]["live_forecast_url"], timeout=60, params={
            "latitude": loc["lat"], "longitude": loc["lon"], "hourly": ",".join(cfg["weather"]["variables"]),
            "past_days": past_days, "forecast_days": forecast_days, "timezone": "UTC"})
        r.raise_for_status()
        df = pd.DataFrame(r.json()["hourly"])
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df["location"] = loc["name"]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def fetch_recent_demand(cfg, api_key, days=25):
    start = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H")
    raw = fetch_demand(api_key, cfg["eia"]["respondent"], cfg["eia"]["type"], start)
    raw["retrieved_at_utc"] = datetime.now(timezone.utc).isoformat()
    return raw


def assemble_live_dataset(cfg, demand_raw, weather_raw):
    demand, _ = bd.clean_demand(demand_raw)
    wide = bd.weather_wide(weather_raw, cfg["weather"]["variables"], cfg["weather"]["locations"], "fc")
    return pd.concat([demand.reindex(wide.index).rename("demand_mwh"), wide], axis=1)


def predict_latest(cfg, ds, model):
    feats, _, _ = build_features(ds, cfg, weather="fc", require_target=False)
    if feats.empty:
        raise RuntimeError("Not enough recent demand data to build features. Try again after EIA publishes more hours.")
    origin = feats["origin_utc"].max()
    rows = feats[feats["origin_utc"] == origin]
    out = rows[["origin_utc", "target_utc", "horizon_step"]].join(model.predict(rows))
    out["actual"] = ds["demand_mwh"].reindex(pd.DatetimeIndex(out["target_utc"])).to_numpy()
    out["fc_temp_index_c"] = ds["fc_temperature_2m__ca_pw"].reindex(pd.DatetimeIndex(out["target_utc"])).to_numpy()
    return out, origin


def run_live(cfg, root, api_key=None):
    api_key = api_key or os.environ.get(cfg["eia"]["api_key_env"])
    if not api_key:
        raise RuntimeError(f"Set {cfg['eia']['api_key_env']} (in your .env file) to fetch fresh EIA demand.")
    path = Path(root) / cfg["paths"]["models"] / "lightgbm_final.joblib"
    if not path.exists():
        raise RuntimeError("No trained model found. Run `python -m src.evaluation.walk_forward` first.")
    model = joblib.load(path)
    ds = assemble_live_dataset(cfg, fetch_recent_demand(cfg, api_key), fetch_live_weather(cfg))
    out, origin = predict_latest(cfg, ds, model)
    return out, {"origin_utc": origin, "fetched_utc": datetime.now(timezone.utc)}
