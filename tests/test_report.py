from pathlib import Path

import pandas as pd
import pytest

from src.evaluation import report as R
from src.evaluation import walk_forward as wf
from src.features import build_features as bf
from src.features import heatwave as hw
from src.models.lightgbm_model import LightGBMModel
from tests.synth import CFG, make_dataset


@pytest.fixture(scope="module")
def outputs(tmp_path_factory):
    d = tmp_path_factory.mktemp("run")
    proc, rep, mdir = d / "processed", d / "reports", d / "models"
    for p in (proc, rep, mdir):
        p.mkdir()
    ds = make_dataset(end="2023-09-30 23:00")                      # includes summer 2023 heat spells in validation
    feats, cols, _ = bf.build_features(ds, CFG)
    preds, info, _ = wf.run(CFG, feats, cols, ds, wf.build_models(CFG, ds, wf.ALL_MODELS), log=lambda *_: None)
    ds.to_parquet(proc / "dataset.parquet")
    feats.to_parquet(proc / "features.parquet")
    preds.to_parquet(proc / "predictions.parquet")
    import joblib
    joblib.dump(LightGBMModel(CFG["models"]["lightgbm"], seed=42).fit(feats, cols), mdir / "lightgbm_final.joblib")
    readme = d / "README.md"
    readme.write_text("# Title\n\n<!-- DATA_FACTS:START -->\nSTALE_PLACEHOLDER\n<!-- DATA_FACTS:END -->\n\ntext\n\n<!-- RESULTS:START -->\nSTALE_PLACEHOLDER\n<!-- RESULTS:END -->\n")
    T, F, facts = R.generate(CFG, proc, rep, mdir, readme, log=lambda *_: None)
    return d, rep, readme, T, F, facts


def test_tables_exist_and_cover_every_required_breakdown(outputs):
    d, rep, readme, T, F, facts = outputs
    for name in ["overall", "by_hour", "by_weekday_weekend", "by_season", "by_temperature_bin", "by_day_type", "largest_errors"]:
        assert (rep / "tables" / f"{name}.csv").exists() and len(T[name]) > 0, name
    assert set(T["overall"]["model"]) == set(wf.ALL_MODELS)
    assert "PeakHourMAE" in T["overall"] and "coverage_80pct" in T["overall"]


def test_every_table_includes_the_baseline(outputs):
    _, _, _, T, _, _ = outputs
    for name in ["overall", "by_hour", "by_weekday_weekend", "by_season", "by_temperature_bin", "by_day_type"]:
        assert "baseline_weekly" in set(T[name]["model"]), name


def test_models_are_compared_on_identical_rows(outputs):
    _, _, _, T, _, _ = outputs
    n = T["overall"].groupby("period")["n"].nunique()
    assert (n == 1).all()


def test_figures_are_written(outputs):
    _, rep, _, _, F, _ = outputs
    assert "Model comparison" in F and "Error heatmap" in F and "Residual analysis" in F
    for rel in F.values():
        assert (rep / "figures" / Path(rel).name).stat().st_size > 5000


def test_readme_blocks_are_filled_from_computed_numbers_only(outputs):
    _, _, readme, T, _, facts = outputs
    text = readme.read_text()
    assert "STALE_PLACEHOLDER" not in text
    mae = T["overall"].query("model == 'lightgbm' and period == 'validation'")["MAE"].iloc[0]
    assert f"{mae:,.0f}" in text
    assert "text" in text and "# Title" in text           # surrounding text untouched


def test_inject_is_idempotent_and_appends_when_markers_missing():
    once = R.inject("abc", "X", "1")
    twice = R.inject(once, "X", "2")
    assert twice.count("<!-- X:START -->") == 1 and "2" in twice and "\\1" not in twice


def test_heatwave_plot_window_chosen_without_looking_at_errors(outputs):
    from src.visualization import extreme_events as ev
    ds = make_dataset(end="2023-09-30 23:00")
    labels = hw.label_heat_waves(hw.daily_temperature_index(ds, CFG["weather"]["locations"], "America/Los_Angeles"), 0.9, 3)
    waves = hw.summarize_waves(labels)
    evaluated = set(labels.index)
    pick = ev.pick_heat_window(waves, evaluated)
    assert pick is not None and pick[2]["peak_temp_index_c"] == waves[waves["kind"] != "off-season"]["peak_temp_index_c"].max()
