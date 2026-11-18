"""Walk-forward (rolling-origin, expanding-window) evaluation plus a final untouched holdout.

Protocol
  * Forecast origin = local midnight; horizon = the next 24 hours (see src/features/build_features.py).
  * VALIDATION: origins from (first origin + min_train_days) up to the holdout start, in blocks of
    `refit_every_days`. Each block is predicted by models trained ONLY on rows whose targets end before
    the block starts, then the window moves forward and models are refit (expanding window).
  * HOLDOUT: the last `holdout_weeks` weeks of origins. Models are trained once on everything before the
    holdout starts and never see it. Nothing is tuned or selected on it. The model list and all settings
    are fixed in configs/config.yaml before this runs.

Outputs
  data/processed/predictions.parquet   one row per (model, period, origin, horizon step)
  models/lightgbm_holdout.joblib       trained before the holdout (used for evaluation and importance)
  models/lightgbm_final.joblib         trained on all data (used by the live forecast demo)
  reports/run_log.json                 settings, seed, versions, timings, fold boundaries
"""
import argparse
import copy
import json
import platform
import subprocess
import time
from pathlib import Path

import joblib
import lightgbm
import numpy as np
import pandas as pd
import sklearn
import statsmodels
import yaml

from src.models.baselines import make_baselines
from src.models.lightgbm_model import LightGBMModel
from src.models.ridge_model import RidgeModel
from src.models.sarimax_model import SarimaxModel

KEEP = ["origin_utc", "target_utc", "target_local_date", "horizon_step", "y"]
ALL_MODELS = ["baseline_weekly", "baseline_yearly", "baseline_mean7", "ridge", "lightgbm", "sarimax"]


def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def build_models(cfg, ds, which):
    tz = cfg["evaluation"]["local_timezone"]
    seed = cfg.get("seed", 42)
    models = {b.name: b for b in make_baselines(ds["demand_mwh"], tz)}
    models["ridge"] = RidgeModel(cfg["models"]["ridge"]["alphas"])
    models["lightgbm"] = LightGBMModel(cfg["models"]["lightgbm"], cfg["evaluation"]["quantiles"], seed)
    models["sarimax"] = SarimaxModel(ds, tz, cfg["models"]["sarimax"], cfg["evaluation"]["horizon_hours"])
    return [models[n] for n in which]


def assert_no_overlap(train, test):
    """Hard leakage guard: every training target must be earlier than the first test origin."""
    if train["target_utc"].max() >= test["origin_utc"].min():
        raise AssertionError("training targets overlap the test period")


def split_origins(origins, cfg, holdout_weeks):
    origins = pd.Series(sorted(origins)).reset_index(drop=True)
    holdout_start = origins.iloc[-7 * holdout_weeks]
    first_val = origins.iloc[0] + pd.Timedelta(days=cfg["evaluation"]["min_train_days"])
    val = origins[(origins >= first_val) & (origins < holdout_start)].reset_index(drop=True)
    return val, origins[origins >= holdout_start].reset_index(drop=True), holdout_start


def _collect(model, test, period, fold):
    pred = model.predict(test)
    df = test[KEEP].copy()
    for col in ("yhat", "q10", "q50", "q90"):
        df[col] = pred[col].to_numpy() if col in pred else np.nan
    df["model"], df["period"], df["fold"] = model.name, period, str(fold)   # always text: parquet needs one type
    return df[df["yhat"].notna()]


def run(cfg, feats, features, ds, models, refit_days=None, holdout_weeks=None, log=print):
    refit_days = refit_days or cfg["evaluation"]["refit_every_days"]
    holdout_weeks = holdout_weeks or cfg["evaluation"]["holdout_weeks"]
    val, hold, holdout_start = split_origins(feats["origin_utc"].unique(), cfg, holdout_weeks)
    log(f"validation origins: {len(val)} ({val.iloc[0].date()} to {val.iloc[-1].date()}); "
        f"holdout origins: {len(hold)} (from {holdout_start.date()}); refit every {refit_days} days")

    out, folds, timings = [], [], {m.name: 0.0 for m in models}
    blocks = [val.iloc[i:i + refit_days] for i in range(0, len(val), refit_days)]
    for i, blk in enumerate(blocks):
        test = feats[feats["origin_utc"].isin(blk)]
        train = feats[feats["origin_utc"] < blk.iloc[0]]
        assert_no_overlap(train, test)
        folds.append({"fold": i, "train_rows": int(len(train)), "train_last_target": str(train["target_utc"].max()),
                      "test_first_origin": str(blk.iloc[0]), "test_last_origin": str(blk.iloc[-1])})
        for m in models:
            t0 = time.time()
            m.fit(train, features)
            out.append(_collect(m, test, "validation", i))
            timings[m.name] += time.time() - t0
        log(f"  fold {i + 1}/{len(blocks)} done (test {blk.iloc[0].date()} to {blk.iloc[-1].date()})")

    # ---- final holdout: fit once on everything before it
    test = feats[feats["origin_utc"] >= holdout_start]
    train = feats[feats["origin_utc"] < holdout_start]
    assert_no_overlap(train, test)
    folds.append({"fold": "holdout", "train_rows": int(len(train)), "train_last_target": str(train["target_utc"].max()),
                  "test_first_origin": str(hold.iloc[0]), "test_last_origin": str(hold.iloc[-1])})
    holdout_models = {}
    for m in models:
        t0 = time.time()
        m.fit(train, features)
        out.append(_collect(m, test, "holdout", "holdout"))
        holdout_models[m.name] = copy.deepcopy(m)
        timings[m.name] += time.time() - t0
    log("  holdout done")
    preds = pd.concat(out, ignore_index=True)
    info = {"holdout_start": str(holdout_start), "n_validation_origins": int(len(val)), "n_holdout_origins": int(len(hold)),
            "refit_every_days": refit_days, "folds": folds, "seconds_per_model": {k: round(v, 1) for k, v in timings.items()}}
    return preds, info, holdout_models


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--refit-every-days", type=int, default=None)
    ap.add_argument("--holdout-weeks", type=int, default=None)
    ap.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    ap.add_argument("--skip-sarimax", action="store_true")
    ap.add_argument("--fast", action="store_true", help="refit every 84 days and use fewer trees (quick preview only)")
    args = ap.parse_args(argv)

    cfg = load_config()
    if args.fast:
        cfg["evaluation"]["refit_every_days"] = 84
        cfg["models"]["lightgbm"]["n_estimators"] = 150
    proc, rep, mdir = (Path(cfg["paths"][k]) for k in ("processed", "reports", "models"))
    feats = pd.read_parquet(proc / "features.parquet")
    features = json.load(open(proc / "feature_list.json"))["features"]
    ds = pd.read_parquet(proc / "dataset.parquet")
    which = [m for m in args.models if not (args.skip_sarimax and m == "sarimax")]
    models = build_models(cfg, ds, which)

    t0 = time.time()
    preds, info, holdout_models = run(cfg, feats, features, ds, models, args.refit_every_days, args.holdout_weeks)
    preds.to_parquet(proc / "predictions.parquet", index=False)

    mdir.mkdir(exist_ok=True)
    if "lightgbm" in holdout_models:
        joblib.dump(holdout_models["lightgbm"], mdir / "lightgbm_holdout.joblib")
        final = LightGBMModel(cfg["models"]["lightgbm"], cfg["evaluation"]["quantiles"], cfg.get("seed", 42))
        final.fit(feats, features)                                 # all data, for the live demo only
        joblib.dump(final, mdir / "lightgbm_final.joblib")
    json.dump({"features": features}, open(mdir / "feature_list.json", "w"), indent=2)

    rep.mkdir(exist_ok=True)
    log = {"finished_utc": pd.Timestamp.now("UTC").isoformat(), "total_seconds": round(time.time() - t0, 1),
           "git_commit": _git_commit(), "seed": cfg.get("seed", 42), "models": which,
           "evaluation": cfg["evaluation"], "model_settings": cfg["models"], "n_features": len(features),
           "versions": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
                        "sklearn": sklearn.__version__, "lightgbm": lightgbm.__version__,
                        "statsmodels": statsmodels.__version__},
           **info}
    json.dump(log, open(rep / "run_log.json", "w"), indent=2)
    print(f"Saved {len(preds):,} predictions to {proc / 'predictions.parquet'} in {log['total_seconds']}s")


if __name__ == "__main__":
    main()
