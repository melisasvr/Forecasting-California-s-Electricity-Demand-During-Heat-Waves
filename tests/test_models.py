import numpy as np
import pandas as pd
import pytest

from src.evaluation import metrics as m
from src.features import build_features as bf
from src.features import heatwave as hw
from src.models.baselines import make_baselines
from src.models.lightgbm_model import LightGBMModel
from src.models.ridge_model import RidgeModel
from src.models.sarimax_model import SarimaxModel
from tests.synth import CFG, make_dataset


@pytest.fixture(scope="module")
def data():
    ds = make_dataset(end="2022-06-30 23:00")
    feats, cols, _ = bf.build_features(ds, CFG)
    split = feats["origin_utc"].sort_values().iloc[int(len(feats) * 0.8)]
    return ds, feats, cols, feats[feats["origin_utc"] < split], feats[feats["origin_utc"] >= split]


def test_weekly_baseline_equals_the_168h_lag_feature(data):
    ds, feats, cols, train, test = data
    base = {b.name: b for b in make_baselines(ds["demand_mwh"], CFG["evaluation"]["local_timezone"])}
    pred = base["baseline_weekly"].predict(test)["yhat"].to_numpy()
    assert np.allclose(pred, test["demand_lag_168h"].to_numpy(), equal_nan=True)


def test_yearly_baseline_uses_364_days_back(data):
    ds, feats, cols, train, test = data
    base = {b.name: b for b in make_baselines(ds["demand_mwh"], CFG["evaluation"]["local_timezone"])}
    row = test.iloc[100]
    expected = ds["demand_mwh"].loc[row["target_utc"] - pd.Timedelta(days=364)]
    assert base["baseline_yearly"].predict(test.iloc[[100]])["yhat"].iloc[0] == pytest.approx(expected)


def test_mean7_baseline_only_averages_same_day_type(data):
    ds, feats, cols, train, test = data
    tz = CFG["evaluation"]["local_timezone"]
    b = {x.name: x for x in make_baselines(ds["demand_mwh"], tz)}["baseline_mean7"]
    row = test.iloc[200:201]
    t = row["target_utc"].iloc[0]
    want = []
    for k in range(1, 8):
        src = t - pd.Timedelta(hours=24 * k)
        if (src.tz_convert(tz).day_of_week >= 5) == (t.tz_convert(tz).day_of_week >= 5):
            want.append(ds["demand_mwh"].loc[src])
    assert b.predict(row)["yhat"].iloc[0] == pytest.approx(np.mean(want))


def test_baselines_never_use_demand_newer_than_24h_before_target(data):
    ds, feats, cols, train, test = data
    tz = CFG["evaluation"]["local_timezone"]
    d2 = ds["demand_mwh"].copy()
    cut = test["origin_utc"].iloc[0]
    d2.loc[d2.index >= cut] = 1e9                     # destroy everything from the first test origin on
    first_day = test[test["origin_utc"] == cut]
    for b in make_baselines(d2, tz):
        pred = b.predict(first_day)["yhat"].to_numpy()
        assert (pred < 1e8).all(), b.name              # all inputs were older than the origin


def test_ridge_and_lightgbm_learn_something(data):
    ds, feats, cols, train, test = data
    naive = np.abs(test["y"] - test["y"].mean()).mean()
    for mdl in (RidgeModel(CFG["models"]["ridge"]["alphas"]),
                LightGBMModel(CFG["models"]["lightgbm"], CFG["evaluation"]["quantiles"], seed=42)):
        mdl.fit(train, cols)
        p = mdl.predict(test)["yhat"]
        assert np.isfinite(p).all()
        assert np.abs(test["y"] - p).mean() < 0.6 * naive, mdl.name


def test_lightgbm_quantiles_are_ordered_and_sensible(data):
    ds, feats, cols, train, test = data
    mdl = LightGBMModel(CFG["models"]["lightgbm"], CFG["evaluation"]["quantiles"], seed=42).fit(train, cols)
    p = mdl.predict(test)
    assert (p["q10"] <= p["q50"]).all() and (p["q50"] <= p["q90"]).all()
    cov = ((test["y"] >= p["q10"]) & (test["y"] <= p["q90"])).mean()
    assert 0.4 < cov < 0.99


def test_lightgbm_is_reproducible(data):
    ds, feats, cols, train, test = data
    a = LightGBMModel(CFG["models"]["lightgbm"], seed=42).fit(train, cols).predict(test)["yhat"].to_numpy()
    b = LightGBMModel(CFG["models"]["lightgbm"], seed=42).fit(train, cols).predict(test)["yhat"].to_numpy()
    assert np.allclose(a, b)


def test_lightgbm_contributions_add_up_to_the_prediction(data):
    ds, feats, cols, train, test = data
    mdl = LightGBMModel(CFG["models"]["lightgbm"], seed=42).fit(train, cols)
    sample = test.iloc[:50]
    c = mdl.contributions(sample)
    assert c.shape == (50, len(cols) + 1)
    assert np.allclose(c.sum(axis=1), mdl.predict(sample)["yhat"].to_numpy(), atol=1e-3)


def test_sarimax_forecasts_are_finite_and_respect_origin_cutoff(data):
    ds, feats, cols, train, test = data
    cfg_s = {**CFG["models"]["sarimax"], "evaluate_every_n_origins": 5}
    mdl = SarimaxModel(ds, CFG["evaluation"]["local_timezone"], cfg_s).fit(train, cols)
    p = mdl.predict(test)
    ok = p["yhat"].notna()
    assert ok.sum() > 100
    assert np.isfinite(p.loc[ok]).all().all() and (p.loc[ok, "q10"] <= p.loc[ok, "q90"]).all()
    # destroy demand from the first test origin on: forecasts for that origin must not change
    o = test["origin_utc"].iloc[0]
    d2 = ds.copy()
    d2.loc[d2.index >= o, "demand_mwh"] = 1e9
    mdl2 = SarimaxModel(d2, CFG["evaluation"]["local_timezone"], cfg_s)
    mdl2.full, mdl2.base, mdl2.full_cols, mdl2.base_cols = mdl.full, mdl.base, mdl.full_cols, mdl.base_cols
    first = test[test["origin_utc"] == o]
    assert np.allclose(mdl2.predict(first)["yhat"].to_numpy(), p.loc[first.index, "yhat"].to_numpy())


def test_annotate_and_metric_tables(data):
    ds, feats, cols, train, test = data
    labels = hw.label_heat_waves(hw.daily_temperature_index(ds, CFG["weather"]["locations"],
                                                            CFG["evaluation"]["local_timezone"]), 0.9, 3)
    preds = test[["origin_utc", "target_utc", "target_local_date", "horizon_step", "y"]].copy()
    preds["model"], preds["period"], preds["yhat"] = "x", "validation", test["y"] * 1.1
    for q in ("q10", "q50", "q90"):
        preds[q] = np.nan
    ann = m.annotate(preds, ds, hw.day_types(labels), CFG)
    assert {"local_hour", "season", "temp_bin", "day_type", "day_kind"} <= set(ann.columns)
    tab = m.metric_table(ann, ["model", "day_type"], CFG["evaluation"]["peak_hours_local"])
    assert (tab["MAPE_pct"].round(6) == 10.0).all()            # forecasting every hour 10% too high = 10% MAPE
    overall = m.metric_table(ann, ["model"], CFG["evaluation"]["peak_hours_local"])
    assert overall["n"].iloc[0] == ann["day_type"].notna().sum() or overall["n"].iloc[0] == len(ann)