import numpy as np
import pandas as pd
import pytest

from src.evaluation import walk_forward as wf
from src.features import build_features as bf
from tests.synth import CFG, make_dataset


@pytest.fixture(scope="module")
def result():
    ds = make_dataset()
    feats, cols, _ = bf.build_features(ds, CFG)
    models = wf.build_models(CFG, ds, wf.ALL_MODELS)
    preds, info, hold_models = wf.run(CFG, feats, cols, ds, models, log=lambda *_: None)
    return ds, feats, cols, preds, info, hold_models


def test_all_models_produce_predictions_in_both_periods(result):
    _, _, _, preds, _, _ = result
    got = preds.groupby(["model", "period"]).size().unstack()
    assert set(got.index) == set(wf.ALL_MODELS)
    assert (got.fillna(0) > 0).all().all()


def test_holdout_is_the_last_eight_weeks_and_untouched_by_training(result):
    _, feats, _, preds, info, _ = result
    last = feats["origin_utc"].drop_duplicates().sort_values().iloc[-56:]
    hold = preds[preds["period"] == "holdout"]
    assert hold["origin_utc"].min() >= last.iloc[0] and hold["origin_utc"].nunique() <= 56
    val = preds[preds["period"] == "validation"]
    assert val["origin_utc"].max() < pd.Timestamp(info["holdout_start"])
    for f in info["folds"]:                                     # every fold trains strictly before it tests
        assert pd.Timestamp(f["train_last_target"]) < pd.Timestamp(f["test_first_origin"])
    assert pd.Timestamp(info["folds"][-1]["train_last_target"]) < pd.Timestamp(info["holdout_start"])


def test_validation_starts_after_the_minimum_training_window(result):
    _, feats, _, preds, _, _ = result
    first = feats["origin_utc"].min() + pd.Timedelta(days=CFG["evaluation"]["min_train_days"])
    assert preds[preds["period"] == "validation"]["origin_utc"].min() >= first


def test_predictions_are_finite_and_quantiles_ordered(result):
    _, _, _, preds, _, _ = result
    assert np.isfinite(preds["yhat"]).all() and np.isfinite(preds["y"]).all()
    q = preds[preds["model"].isin(["lightgbm", "sarimax"])]
    assert (q["q10"] <= q["q90"]).all()


def test_learned_models_beat_the_weekly_baseline_on_synthetic_data(result):
    _, _, _, preds, _, _ = result
    from src.evaluation.metrics import common_rows, mae
    c = common_rows(preds[preds["period"] == "validation"])
    err = {m: mae(g["y"], g["yhat"]) for m, g in c.groupby("model")}
    assert err["lightgbm"] < err["baseline_weekly"]


def test_training_window_never_contains_the_holdout_even_for_the_final_models(result):
    _, feats, _, _, info, hold_models = result
    # a model retrained on holdout rows would reproduce them almost perfectly; the saved one must not
    hold = feats[feats["origin_utc"] >= pd.Timestamp(info["holdout_start"])]
    pred = hold_models["lightgbm"].predict(hold)["yhat"]
    assert np.abs(hold["y"] - pred).mean() > 50          # synthetic noise sd is 250, so not memorised


def test_leakage_guard_raises_on_overlap(result):
    _, feats, _, _, _, _ = result
    with pytest.raises(AssertionError):
        wf.assert_no_overlap(feats, feats.tail(100))
