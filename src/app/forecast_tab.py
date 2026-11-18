"""Streamlit helpers for the 'Forecast' (backtest) and 'Live demo' tabs."""
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

PRIMARY, BASELINE = "lightgbm", "baseline_weekly"
COLORS = {"baseline_weekly": "#7f8c8d", "baseline_yearly": "#bbbbbb", "baseline_mean7": "#999999",
          "ridge": "#1f77b4", "lightgbm": "#d62728", "sarimax": "#2ca02c"}


def _forecast_figure(day, tz, title):
    fig = go.Figure()
    d = day.copy()
    d["t"] = pd.DatetimeIndex(d["target_utc"]).tz_convert(tz).tz_localize(None)
    d = d.sort_values("t")
    a = d[d["model"] == PRIMARY] if (d["model"] == PRIMARY).any() else d.drop_duplicates("t")
    fig.add_trace(go.Scatter(x=a["t"], y=a["y"], name="Actual", line=dict(color="#222", width=3)))
    for m, g in d.groupby("model"):
        if m == PRIMARY and g["q10"].notna().any():
            fig.add_trace(go.Scatter(x=g["t"], y=g["q90"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=g["t"], y=g["q10"], fill="tonexty", fillcolor="rgba(214,39,40,0.18)",
                                     line=dict(width=0), name="10-90% interval"))
        fig.add_trace(go.Scatter(x=g["t"], y=g["yhat"], name=m, line=dict(color=COLORS.get(m, "#555"),
                                 dash="dash" if m.startswith("baseline") else "solid")))
    fig.update_layout(title=title, height=430, yaxis_title="Demand (MWh per hour)", legend=dict(orientation="h", y=1.12))
    return fig


def render_forecast_tab(st, cfg, ds, proc_dir, rep_dir, plot_kw):
    path = Path(proc_dir) / "predictions.parquet"
    st.info("**BACKTEST forecasts.** Weather inputs are the archived forecasts that would have been known at midnight. "
            "ERA5 reanalysis appears only as the reference for what actually happened. This is not a live forecast.")
    if not path.exists():
        st.warning("No predictions yet. Run `python -m src.evaluation.walk_forward`.")
        return
    tz = cfg["evaluation"]["local_timezone"]
    preds = pd.read_parquet(path)
    period = st.radio("Evaluation period", ["validation", "holdout"], horizontal=True,
                      help="Validation = walk-forward backtest. Holdout = final 8 weeks, never used for any choice.")
    sub = preds[preds["period"] == period]
    days = sorted(sub["target_local_date"].unique())
    default = len(days) - 1
    if period == "validation" and pd.Timestamp("2022-09-06") in days:
        default = days.index(pd.Timestamp("2022-09-06"))
    sel = st.selectbox("Forecast date (Pacific day)", days, index=default,
                       format_func=lambda d: pd.Timestamp(d).strftime("%Y-%m-%d (%a)"))
    models = st.multiselect("Models", sorted(sub["model"].unique()), default=[m for m in (BASELINE, PRIMARY) if m in set(sub["model"])])
    day = sub[(sub["target_local_date"] == sel) & (sub["model"].isin(models))]
    if day.empty:
        st.info("No forecast for that selection.")
        return
    st.plotly_chart(_forecast_figure(day, tz, f"24-hour forecast for {pd.Timestamp(sel).date()} (issued at local midnight)"), **plot_kw)

    t_idx = pd.DatetimeIndex(day[day["model"] == day["model"].iloc[0]]["target_utc"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Forecast temp index, daily max (assumed)", f"{ds['fc_temperature_2m__ca_pw'].reindex(t_idx).max():.1f} C")
    c2.metric("Actual temp index, daily max (ERA5)", f"{ds['era5_temperature_2m__ca_pw'].reindex(t_idx).max():.1f} C")
    err = day.assign(ae=(day["y"] - day["yhat"]).abs()).groupby("model")["ae"].mean()
    if PRIMARY in err:
        c3.metric(f"{PRIMARY} MAE this day", f"{err[PRIMARY]:,.0f} MWh")
    st.caption("Weather assumptions: the model sees the archived forecast (first number). The second number is what ERA5 says happened.")
    st.markdown("**Mean absolute error on this day**")
    st.dataframe(err.round(0).rename("MAE (MWh)").to_frame())

    tbl = Path(rep_dir) / "tables" / "overall.csv"
    if tbl.exists():
        o = pd.read_csv(tbl)
        o = o[o["period"] == period][["model", "n", "MAE", "RMSE", "MAPE_pct", "PeakHourMAE"]].round(2)
        st.markdown(f"**Overall metrics, {period} (all models on identical rows)**")
        st.dataframe(o.set_index("model"))


def render_live_tab(st, cfg, root, plot_kw):
    from src.app.live_forecast import run_live
    tz = cfg["evaluation"]["local_timezone"]
    st.warning("**LIVE FORECAST WEATHER.** This tab uses the Open-Meteo forecast API for the weather inputs, "
               "not ERA5 reanalysis, and fresh demand from EIA. It is a demo, not an operational product.")
    st.caption("Needs your EIA key in `.env`, an internet connection and a trained model. "
               "The forecast is for the current Pacific day, issued at the most recent midnight with complete demand data.")
    if st.button("Run live forecast"):
        try:
            with st.spinner("Fetching live weather and recent demand..."):
                out, meta = run_live(cfg, root)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Live forecast failed: {exc}")
            return
        st.success(f"Forecast origin: {meta['origin_utc'].tz_convert(tz):%Y-%m-%d %H:%M} Pacific. "
                   f"Weather fetched {meta['fetched_utc']:%Y-%m-%d %H:%M} UTC.")
        out = out.assign(model=PRIMARY, y=out["actual"])
        st.plotly_chart(_forecast_figure(out, tz, "Live 24-hour forecast (actual demand shown where already published)"), **plot_kw)
        st.dataframe(out[["target_utc", "yhat", "q10", "q90", "actual", "fc_temp_index_c"]].round(1))
