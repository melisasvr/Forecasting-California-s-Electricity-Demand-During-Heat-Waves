"""Turn predictions into tables, figures, reports/RESULTS.md and the results blocks of README.md.

Every number written here is computed from data/processed/predictions.parquet and the dataset.
Nothing is typed in by hand.
"""
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from src.evaluation import metrics as M
from src.features import heatwave as hw
from src.visualization import extreme_events as ev
from src.visualization import plots as P

PRIMARY, BASELINE = P.PRIMARY, P.BASELINE
MODEL_ORDER = ["baseline_weekly", "baseline_yearly", "baseline_mean7", "ridge", "lightgbm", "sarimax"]


# ------------------------------------------------------------------ formatting helpers
def md_table(df, fmt=None):
    fmt = fmt or {}
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if pd.isna(v):
                cells.append("")
            elif c in fmt:
                cells.append(fmt[c].format(v))
            elif isinstance(v, (float, np.floating)):
                cells.append(f"{v:,.2f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def inject(text, marker, block):
    start, end = f"<!-- {marker}:START -->", f"<!-- {marker}:END -->"
    new = f"{start}\n{block}\n{end}"
    pat = re.compile(re.escape(start) + r".*?" + re.escape(end), re.S)
    return pat.sub(lambda _m: new, text) if pat.search(text) else text.rstrip() + "\n\n" + new + "\n"


def order(df, col="model"):
    key = {m: i for i, m in enumerate(MODEL_ORDER)}
    return df.sort_values(col, key=lambda s: s.map(key).fillna(99))


# ------------------------------------------------------------------ data facts (descriptive, no models)
def data_facts(ds, labels, waves, cfg):
    tz = cfg["evaluation"]["local_timezone"]
    d = ds["demand_mwh"]
    ok = d.dropna()
    lines = [
        f"- **Coverage:** {len(ok):,} hourly demand readings from {ok.index.min().date()} to {ok.index.max().date()}, "
        f"with {int(d.isna().sum())} hours missing (not imputed).",
    ]
    counts = waves["kind"].value_counts()
    n_head = int((waves["kind"] != "off-season").sum())
    days = int(labels["cooling_season_heat_wave_day"].sum())
    lines.append(f"- **Heat waves:** the fixed rule finds {len(waves)} waves: {n_head} in the May to October cooling season "
                 f"({days} heat-wave days) and {int(counts.get('off-season', 0))} off-season warm spells.")
    win = (pd.Timestamp("2022-08-28"), pd.Timestamp("2022-09-12"))
    hit = waves[(waves["end"] >= win[0]) & (waves["start"] <= win[1]) & (waves["kind"] != "off-season")]
    if len(hit):
        w = hit.sort_values("peak_temp_index_c").iloc[-1]
        sub = ds[(ds.index.tz_convert(tz).tz_localize(None) >= win[0]) & (ds.index.tz_convert(tz).tz_localize(None) < win[1] + pd.Timedelta(days=1))]
        pk_t = sub["demand_mwh"].idxmax()
        lines.append(f"- **September 2022:** the rule detects a {int(w['days'])}-day wave ({w['start'].date()} to {w['end'].date()}) "
                     f"with a peak statewide temperature index of {w['peak_temp_index_c']:.1f} C. Peak hourly demand in "
                     f"{win[0].date()} to {win[1].date()} was {sub['demand_mwh'].max():,.0f} MWh, in the hour starting "
                     f"{pk_t.tz_convert(tz).strftime('%H:%M')} Pacific on {pk_t.tz_convert(tz).date()}.")
    loc = ds.index.tz_convert(tz)
    daily_peak = ds["demand_mwh"].groupby(loc.normalize().tz_localize(None)).agg(["max", "count"])
    daily_peak = daily_peak[daily_peak["count"] >= 23]["max"].to_frame("peak").join(labels, how="inner")
    sel = daily_peak[daily_peak.index.month.isin(cfg["heatwave"]["headline_months"])]
    if sel["cooling_season_heat_wave_day"].any():
        a = sel.loc[sel["cooling_season_heat_wave_day"], "peak"].mean()
        b = sel.loc[~sel["heat_wave_day"], "peak"].mean()
        lines.append(f"- **Heat and load:** in May to October the average daily peak was {a:,.0f} MWh on heat-wave days versus "
                     f"{b:,.0f} MWh on days that were not part of any wave ({(a / b - 1):.0%} higher). A raw comparison, not a causal estimate.")
    both = ds[["era5_temperature_2m__ca_pw", "fc_temperature_2m__ca_pw"]].dropna()
    lines.append(f"- **Weather sources:** ERA5 and the archived forecasts differ by {(both.iloc[:, 0] - both.iloc[:, 1]).abs().mean():.2f} C "
                 "on average (statewide temperature index).")
    return "\n".join(lines)


# ------------------------------------------------------------------ results
def pct_change(new, old):
    return (new - old) / old * 100 if old else np.nan


def narrative(overall, by_type):
    out = []
    for period in ("validation", "holdout"):
        t = overall[overall["period"] == period].set_index("model")
        if PRIMARY in t.index and BASELINE in t.index:
            a, b = t.loc[PRIMARY, "MAE"], t.loc[BASELINE, "MAE"]
            best = t["MAE"].idxmin()
            out.append(f"- **{period.capitalize()}:** {PRIMARY} MAE {a:,.0f} MWh vs {b:,.0f} MWh for {BASELINE} "
                       f"({pct_change(a, b):+.1f}%). Lowest MAE among all models: {best} ({t.loc[best, 'MAE']:,.0f} MWh).")
        w = by_type[(by_type["period"] == period) & (by_type["day_type"] == hw.DAY_TYPE_LABELS[0])].set_index("model")
        if PRIMARY in w.index and BASELINE in w.index and w.loc[PRIMARY, "n"] > 0:
            a, b = w.loc[PRIMARY, "MAE"], w.loc[BASELINE, "MAE"]
            out.append(f"  - On cooling-season heat-wave days ({int(w.loc[PRIMARY, 'n']):,} forecast hours): {PRIMARY} MAE {a:,.0f} "
                       f"vs {b:,.0f} for {BASELINE} ({pct_change(a, b):+.1f}%).")
    return "\n".join(out)


def results_markdown(tables, figs, narr, run_log):
    ov = order(tables["overall"])
    cols = ["model", "n", "MAE", "RMSE", "MAPE_pct", "sMAPE_pct", "PeakHourMAE"]
    fmt = {"n": "{:,.0f}", "MAE": "{:,.0f}", "RMSE": "{:,.0f}", "MAPE_pct": "{:.2f}", "sMAPE_pct": "{:.2f}", "PeakHourMAE": "{:,.0f}"}
    parts = ["### Summary", narr, ""]
    for period, nm in (("validation", "Walk-forward validation"), ("holdout", "Final holdout (last 8 weeks)")):
        t = ov[ov["period"] == period][cols].rename(columns={"MAPE_pct": "MAPE %", "sMAPE_pct": "sMAPE %", "PeakHourMAE": "Peak-hour MAE"})
        fmt2 = {k.replace("MAPE_pct", "MAPE %").replace("sMAPE_pct", "sMAPE %").replace("PeakHourMAE", "Peak-hour MAE"): v for k, v in fmt.items()}
        parts += [f"### {nm}: all models on identical rows (MWh per hour)", md_table(t, fmt2), ""]
    bt = tables["by_day_type"]
    piv = bt.pivot_table(index=["period", "day_type"], columns="model", values="MAE").reset_index()
    piv = piv[["period", "day_type"] + [m for m in MODEL_ORDER if m in piv.columns]]
    parts += ["### MAE by day type", md_table(piv, {m: "{:,.0f}" for m in MODEL_ORDER}), ""]
    pi = tables["overall"].dropna(subset=["coverage_80pct"]) if "coverage_80pct" in tables["overall"] else pd.DataFrame()
    if len(pi):
        pi = order(pi)[["period", "model", "pinball_10", "pinball_50", "pinball_90", "coverage_80pct", "mean_width"]]
        parts += ["### Prediction intervals (10% to 90%, nominal coverage 80%)",
                  md_table(pi, {"coverage_80pct": "{:.1%}", "pinball_10": "{:,.0f}", "pinball_50": "{:,.0f}",
                                "pinball_90": "{:,.0f}", "mean_width": "{:,.0f}"}), ""]
    le = tables["largest_errors"]
    if len(le):
        parts += [f"### Ten largest forecast-error days for {PRIMARY} (mean absolute error per day)",
                  md_table(le, {"MAE": "{:,.0f}", "peak_error": "{:,.0f}", "temp_index_max_c": "{:.1f}"}), ""]
    parts += ["### Figures"] + [f"![{k}]({v})" for k, v in figs.items()]
    parts += ["", f"_Run: {run_log.get('finished_utc', 'n/a')}, seed {run_log.get('seed', 'n/a')}, "
                  f"{run_log.get('n_features', '?')} features, models refit every {run_log.get('refit_every_days', '?')} days._"]
    return "\n".join(parts)


def generate(cfg, proc=None, rep=None, mdir=None, readme=None, log=print):
    proc = Path(proc or cfg["paths"]["processed"])
    rep = Path(rep or cfg["paths"]["reports"])
    mdir = Path(mdir or cfg["paths"]["models"])
    tz = cfg["evaluation"]["local_timezone"]
    peak = cfg["evaluation"]["peak_hours_local"]
    (rep / "tables").mkdir(parents=True, exist_ok=True)
    (rep / "figures").mkdir(parents=True, exist_ok=True)

    preds = pd.read_parquet(proc / "predictions.parquet")
    ds = pd.read_parquet(proc / "dataset.parquet")
    if (proc / "heatwave_labels.parquet").exists():
        labels = pd.read_parquet(proc / "heatwave_labels.parquet")
    else:
        h = cfg["heatwave"]
        labels = hw.label_heat_waves(hw.daily_temperature_index(ds, cfg["weather"]["locations"], tz),
                                     h["percentile"], h["min_consecutive_days"], h["headline_months"])
    waves = hw.summarize_waves(labels)
    dts = hw.day_types(labels)
    run_log = json.load(open(rep / "run_log.json")) if (rep / "run_log.json").exists() else {}

    # ---- tables on identical rows
    common = M.common_rows(preds)
    ann = M.annotate(common, ds, dts, cfg)
    ann_all = M.annotate(preds, ds, dts, cfg)
    T = {}
    T["overall"] = M.metric_table(ann, ["model", "period"], peak)
    T["overall_each_models_own_rows"] = M.metric_table(ann_all, ["model", "period"], peak)
    T["by_hour"] = M.metric_table(ann, ["model", "period", "local_hour"], peak)
    T["by_weekday_weekend"] = M.metric_table(ann, ["model", "period", "day_kind"], peak)
    T["by_season"] = M.metric_table(ann, ["model", "period", "season"], peak)
    T["by_temperature_bin"] = M.metric_table(ann, ["model", "period", "temp_bin"], peak)
    T["by_day_type"] = M.metric_table(ann, ["model", "period", "day_type"], peak)
    daily = (ann[ann["model"] == PRIMARY].assign(ae=lambda d: (d["y"] - d["yhat"]).abs())
             .groupby(["period", "target_local_date"]).agg(MAE=("ae", "mean"), peak_error=("ae", "max")).reset_index())
    tmax = ds["era5_temperature_2m__ca_pw"].groupby(ds.index.tz_convert(tz).normalize().tz_localize(None)).max()
    daily["day_type"] = daily["target_local_date"].map(dts)
    daily["temp_index_max_c"] = daily["target_local_date"].map(tmax)
    T["largest_errors"] = daily.sort_values("MAE", ascending=False).head(10).rename(columns={"target_local_date": "date"})
    T["largest_errors"]["date"] = T["largest_errors"]["date"].dt.date.astype(str)
    for name, t in T.items():
        t.to_csv(rep / "tables" / f"{name}.csv", index=False)

    # ---- figures (each isolated so one failure does not stop the report)
    F, figdir = {}, rep / "figures"
    rel = lambda p: str(Path("reports/figures") / Path(p).name)

    def attempt(key, fn, *a, **k):
        try:
            F[key] = rel(fn(*a, **k))
        except Exception as exc:  # noqa: BLE001
            log(f"  figure '{key}' skipped: {exc}")

    attempt("Model comparison", P.fig_model_comparison, T["overall"], figdir / "model_comparison.png")
    evaluated = set(ann_all["target_local_date"].unique())
    heat = ev.pick_heat_window(waves, evaluated)
    if heat:
        attempt("Actual vs predicted, heat wave", P.fig_actual_vs_predicted, ann_all, heat[:2],
                f"Heat-wave case: {heat[2]['start'].date()} to {heat[2]['end'].date()} (peak index {heat[2]['peak_temp_index_c']:.1f} C)",
                figdir / "actual_vs_predicted_heat_wave.png", tz)
    nw = ev.pick_normal_week(dts, evaluated)
    if nw:
        attempt("Actual vs predicted, normal week", P.fig_actual_vs_predicted, ann_all, nw,
                f"Normal week: {nw[0].date()} to {nw[1].date()}", figdir / "actual_vs_predicted_normal_week.png", tz)
    sept = (pd.Timestamp("2022-08-28"), pd.Timestamp("2022-09-12"))
    if {sept[0], sept[1]} <= evaluated:
        attempt("September 2022 case study", P.fig_case_study, ann_all, ds, sept, figdir / "case_study_september_2022.png", tz,
                "September 2022 California heat wave")
    attempt("Error heatmap", P.fig_error_heatmap, ann_all, figdir / "error_heatmap_hour_weekday.png")
    attempt("Residual analysis", P.fig_residuals, ann_all, M._bin_labels(cfg["features"]["temp_bin_edges_c"]),
            figdir / "residual_analysis.png")
    attempt("Weather locations and weights", P.fig_locations, cfg["weather"]["locations"], figdir / "weather_locations_weights.png")
    mpath = mdir / "lightgbm_final.joblib"
    if mpath.exists() and (proc / "features.parquet").exists():
        feats = pd.read_parquet(proc / "features.parquet")
        sample = feats.sample(min(4000, len(feats)), random_state=cfg.get("seed", 42))
        attempt("Feature importance (SHAP)", P.fig_importance, joblib.load(mpath), sample, figdir / "feature_importance_shap.png")

    narr = narrative(T["overall"], T["by_day_type"])
    md = results_markdown(T, F, narr, run_log)
    facts = data_facts(ds, labels, waves, cfg)
    (rep / "RESULTS.md").write_text("# Results\n\n_Generated by `python -m src.evaluation.report`. Do not edit by hand._\n\n" + md + "\n")
    if readme and Path(readme).exists():
        text = Path(readme).read_text(encoding="utf-8")
        text = inject(text, "DATA_FACTS", facts)
        text = inject(text, "RESULTS", md)
        Path(readme).write_text(text, encoding="utf-8")
    log(f"Report written to {rep} ({len(F)} figures, {len(T)} tables)")
    return T, F, facts


def main():
    cfg = yaml.safe_load(open("configs/config.yaml"))
    generate(cfg, readme="README.md")


if __name__ == "__main__":
    main()
