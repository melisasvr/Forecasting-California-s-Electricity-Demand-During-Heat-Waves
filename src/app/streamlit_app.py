"""CAISO demand and heat: data explorer.

Run from the project root:
    streamlit run src/app/streamlit_app.py

Everything here is computed from the real processed dataset. The Forecast tab shows backtest
forecasts from data/processed/predictions.parquet; the Live demo tab uses live forecast weather.
"""
import inspect
import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml
from dotenv import load_dotenv
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")   # so the Live demo can find EIA_API_KEY
from src.app.forecast_tab import render_forecast_tab, render_live_tab  # noqa: E402
from src.features import heatwave as hw  # noqa: E402

st.set_page_config(page_title="CAISO demand and heat", layout="wide")

# streamlit renamed this argument across versions
if "width" in inspect.signature(st.plotly_chart).parameters:
    PLOT_KW = {"width": "stretch"}
else:
    PLOT_KW = {"use_container_width": True}

SUMMER, OFFSEASON, HOT, NORMAL = (
    "Cooling-season heat-wave day",
    "Off-season warm spell day",
    "Hot day (not part of a 3+ day run)",
    "Normal day",
)
DAY_COLORS = {SUMMER: "#d62728", OFFSEASON: "#9b59b6", HOT: "#ff9f43", NORMAL: "#7f8c8d"}


@st.cache_data(show_spinner="Loading data...")
def load():
    cfg = yaml.safe_load(open(ROOT / "configs" / "config.yaml"))
    proc = Path(os.environ.get("CAISO_PROCESSED_DIR") or ROOT / cfg["paths"]["processed"])
    path = proc / "dataset.parquet"
    if not path.exists():
        return None
    ds = pd.read_parquet(path)
    tz = cfg["evaluation"]["local_timezone"]
    local = ds.index.tz_convert(tz)
    ds = ds.assign(
        local_time=local.tz_localize(None),
        local_date=local.normalize().tz_localize(None),
        local_hour=local.hour,
    )
    g = ds.groupby("local_date")["demand_mwh"]
    daily = pd.DataFrame({"mean_mwh": g.mean(), "peak_mwh": g.max(), "n": g.count()})
    daily = daily[daily["n"] >= 23]

    h = cfg["heatwave"]
    idx = hw.daily_temperature_index(ds, cfg["weather"]["locations"], tz, "era5")
    labels = hw.label_heat_waves(
        idx, h["percentile"], h["min_consecutive_days"],
        h.get("headline_months", hw.DEFAULT_HEADLINE_MONTHS),
    )
    waves = hw.summarize_waves(labels)
    daily = daily.join(labels, how="inner")
    daily["day_type"] = np.select(
        [daily["cooling_season_heat_wave_day"], daily["heat_wave_day"], daily["hot_day"]],
        [SUMMER, OFFSEASON, HOT], default=NORMAL,
    )
    qpath = ROOT / "reports" / "data_quality.json"
    quality = json.load(open(qpath)) if qpath.exists() else None
    return cfg, ds, daily, labels, waves, quality


loaded = load()
if loaded is None:
    st.error("data/processed/dataset.parquet not found. Run `python -m src.data.build_dataset` first.")
    st.stop()
cfg, ds, daily, labels, waves, quality = loaded
hcfg = cfg["heatwave"]
headline_months = hcfg.get("headline_months", list(hw.DEFAULT_HEADLINE_MONTHS))
month_name = lambda m: pd.Timestamp(2000, m, 1).strftime("%B")

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("About this view")
    st.info(
        "**Weather source by tab.** Overview, Heat vs demand and Case study use HISTORICAL ERA5 reanalysis "
        "(gridded model output, not station observations). The Forecast tab uses archived forecasts as model "
        "input. Only the Live demo tab uses LIVE forecast weather."
    )
    st.markdown("**Heat-wave rule (fixed in advance)**")
    st.markdown(
        f"- Index: population-weighted daily max temperature across "
        f"{len(cfg['weather']['locations'])} locations\n"
        f"- Hot day: above the {int(hcfg['percentile'] * 100)}th percentile for its calendar month\n"
        f"- Heat wave: {hcfg['min_consecutive_days']}+ hot days in a row\n"
        f"- Headline results: waves in {month_name(min(headline_months))} to "
        f"{month_name(max(headline_months))}. Waves in other months are labeled "
        f"off-season warm spells."
    )
    st.caption("Only the Forecast and Live demo tabs contain model predictions.")

st.title("California electricity demand and heat")
st.caption("CAISO hourly demand (EIA-930), 2021 to present. Times shown in Pacific time.")

tab_over, tab_heat, tab_case, tab_fc, tab_live, tab_map, tab_dq = st.tabs(
    ["Overview", "Heat vs demand", "Case study", "Forecast", "Live demo", "Weather map", "Data quality"]
)
proc_dir = Path(os.environ.get("CAISO_PROCESSED_DIR") or ROOT / cfg["paths"]["processed"])

# ---------------------------------------------------------------- overview
with tab_over:
    n_headline_waves = int((waves["kind"] != "off-season").sum())
    n_off_waves = int((waves["kind"] == "off-season").sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Days of data", f"{len(daily):,}")
    c2.metric("Hours of demand", f"{int(ds['demand_mwh'].notna().sum()):,}")
    c3.metric("Cooling-season heat waves", f"{n_headline_waves}")
    c4.metric("Cooling-season heat-wave days", f"{int(daily['cooling_season_heat_wave_day'].sum())}")
    st.caption(f"Plus {n_off_waves} off-season warm spells (purple): unusually warm for their month. "
               "Most are mild winter spells, though a few spring events (March and April) are genuinely hot. "
               "They are labeled but kept out of the headline numbers.")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily.index, y=daily["peak_mwh"], mode="lines", name="Daily peak"))
    fig.add_trace(go.Scatter(x=daily.index, y=daily["mean_mwh"], mode="lines", name="Daily average"))
    for _, w in waves.iterrows():
        off = w["kind"] == "off-season"
        fig.add_vrect(x0=w["start"], x1=w["end"] + pd.Timedelta(days=1),
                      fillcolor="#b57bee" if off else "#ff4d4d",
                      opacity=0.30 if off else 0.45, line_width=0)
    fig.update_layout(
        title="Daily demand. Red = cooling-season heat waves, purple = off-season warm spells.", height=460,
        yaxis_title="Demand (MWh per hour)", xaxis_rangeslider_visible=True,
        legend=dict(orientation="h", y=1.12),
    )
    st.plotly_chart(fig, **PLOT_KW)

    if len(waves):
        st.subheader("Detected waves")
        show = waves.copy()
        show["start"], show["end"] = show["start"].dt.date, show["end"].dt.date
        show["peak_temp_index_c"] = show["peak_temp_index_c"].round(1)
        st.dataframe(show.set_index("wave_id"))

# ---------------------------------------------------------------- heat vs demand
with tab_heat:
    st.subheader("Hotter days, higher peaks")
    plot_df = daily.rename_axis("date").reset_index()
    fig = px.scatter(
        plot_df, x="temp_index_c", y="peak_mwh", color="day_type",
        color_discrete_map=DAY_COLORS, opacity=0.6, hover_data={"date": True},
        labels={"temp_index_c": "Temperature index (°C, ERA5)", "peak_mwh": "Daily peak demand (MWh)",
                "day_type": ""},
        category_orders={"day_type": list(DAY_COLORS)},
    )
    fig.update_layout(height=470, legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, **PLOT_KW)

    st.subheader("What an average day looks like")
    months = st.multiselect(
        "Compare within these months", list(range(1, 13)), default=headline_months,
        format_func=month_name,
    )
    hourly = ds.join(labels[["heat_wave_day"]], on="local_date")
    hourly = hourly.dropna(subset=["heat_wave_day", "demand_mwh"])
    hourly = hourly[hourly["local_date"].dt.month.isin(months)]
    hourly["Day type"] = np.where(hourly["heat_wave_day"].astype(bool), "Heat-wave days", "Other days")
    n_wave = hourly.loc[hourly["Day type"] == "Heat-wave days", "local_date"].nunique()
    if n_wave == 0:
        st.info("No heat-wave days in the selected months.")
    else:
        prof = hourly.groupby(["Day type", "local_hour"])["demand_mwh"].mean().reset_index()
        fig = px.line(prof, x="local_hour", y="demand_mwh", color="Day type",
                      color_discrete_map={"Heat-wave days": "#d62728", "Other days": "#7f8c8d"},
                      labels={"local_hour": "Hour of day (Pacific, hour beginning)", "demand_mwh": "Average demand (MWh)"})
        fig.update_layout(height=400, legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, **PLOT_KW)

        sel = daily[daily.index.month.isin(months)]
        a = sel.loc[sel["heat_wave_day"], "peak_mwh"].mean()
        b = sel.loc[~sel["heat_wave_day"], "peak_mwh"].mean()
        m1, m2, m3 = st.columns(3)
        m1.metric("Avg daily peak, heat-wave days", f"{a:,.0f} MWh")
        m2.metric("Avg daily peak, other days", f"{b:,.0f} MWh")
        m3.metric("Heat-wave days in selection", f"{n_wave}")
        st.caption("Raw averages from the data, compared within the same months. "
                   "Not a causal estimate: heat-wave days also differ in other ways.")

# ---------------------------------------------------------------- case study
with tab_case:
    st.subheader("Zoom in on any window")
    min_d, max_d = daily.index.min().date(), daily.index.max().date()
    default = (date(2022, 8, 28), date(2022, 9, 12))
    if not (min_d <= default[0] and default[1] <= max_d):
        default = (max_d - timedelta(days=14), max_d)
    rng = st.date_input("Window (default: the September 2022 heat wave)", value=default,
                        min_value=min_d, max_value=max_d)
    if not isinstance(rng, (tuple, list)) or len(rng) != 2:
        st.info("Pick both a start and an end date.")
    else:
        s, e = pd.Timestamp(rng[0]), pd.Timestamp(rng[1])
        w = ds[(ds["local_date"] >= s) & (ds["local_date"] <= e)]
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                            row_heights=[0.6, 0.4],
                            subplot_titles=("Hourly demand", "Statewide temperature index (population-weighted)"))
        fig.add_trace(go.Scatter(x=w["local_time"], y=w["demand_mwh"], name="Demand", line=dict(color="#1f77b4")), 1, 1)
        fig.add_trace(go.Scatter(x=w["local_time"], y=w["era5_temperature_2m__ca_pw"],
                                 name="ERA5 reanalysis", line=dict(color="#d62728")), 2, 1)
        fig.add_trace(go.Scatter(x=w["local_time"], y=w["fc_temperature_2m__ca_pw"],
                                 name="Archived forecast", line=dict(color="#ff9f43", dash="dot")), 2, 1)
        fig.update_yaxes(title_text="MWh per hour", row=1, col=1)
        fig.update_yaxes(title_text="°C", row=2, col=1)
        fig.update_layout(height=620, legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, **PLOT_KW)

        if w["demand_mwh"].notna().any():
            pk = w["demand_mwh"].idxmax()
            m1, m2 = st.columns(2)
            m1.metric("Peak demand in window", f"{w['demand_mwh'].max():,.0f} MWh")
            m2.metric("When (Pacific, hour beginning)", str(w.loc[pk, "local_time"])[:16])
        hit = waves[(waves["end"] >= s) & (waves["start"] <= e)]
        if len(hit):
            st.markdown("**Waves overlapping this window (by the fixed rule)**")
            hit = hit.copy()
            hit["start"], hit["end"] = hit["start"].dt.date, hit["end"].dt.date
            st.dataframe(hit.set_index("wave_id").round(1))
        else:
            st.markdown("No heat wave by the fixed rule overlaps this window.")

# ---------------------------------------------------------------- forecast (backtest) and live demo
with tab_fc:
    render_forecast_tab(st, cfg, ds, proc_dir, ROOT / cfg["paths"]["reports"], PLOT_KW)
with tab_live:
    render_live_tab(st, cfg, ROOT, PLOT_KW)

# ---------------------------------------------------------------- map
with tab_map:
    st.subheader("Where the weather comes from")
    locs = pd.DataFrame(cfg["weather"]["locations"])
    scatter = getattr(px, "scatter_map", None)
    if scatter is not None:
        fig = scatter(locs, lat="lat", lon="lon", size="weight", hover_name="name",
                      hover_data={"weight": ":.2f", "lat": False, "lon": False},
                      zoom=4.6, size_max=45, map_style="open-street-map")
    else:
        fig = px.scatter_mapbox(locs, lat="lat", lon="lon", size="weight", hover_name="name",
                                zoom=4.6, size_max=45, mapbox_style="open-street-map")
    fig.update_layout(height=520, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, **PLOT_KW)
    st.info("Weights are shares of 2020 Census populations: Los Angeles, San Diego, Fresno and Sacramento counties, "
            "plus four core Bay Area counties (Alameda, Contra Costa, San Francisco, Santa Clara). "
            "Other counties, such as Orange, Riverside and San Bernardino, are not represented.")
    st.dataframe(locs.rename(columns={"name": "Location", "lat": "Lat", "lon": "Lon",
                                      "population": "Population (2020)", "weight": "Weight"}))

# ---------------------------------------------------------------- data quality
with tab_dq:
    st.subheader("What the cleaning step found")
    if quality is None:
        st.info("reports/data_quality.json not found. Run `python -m src.data.build_dataset`.")
    else:
        keys = ["raw_rows", "duplicate_timestamps", "null_or_non_numeric", "out_of_range_set_to_nan",
                "missing_hours_in_range", "demand_nan_hours_total", "demand_first", "demand_last"]
        st.table(pd.DataFrame({"value": {k: str(quality.get(k)) for k in keys}}))
        if quality.get("gap_runs"):
            st.markdown("**Gaps in the demand series**")
            st.dataframe(pd.DataFrame(quality["gap_runs"]))
        st.markdown("**Checks failed:** " + (", ".join(quality["checks_failed"]) or "none"))
    nan_hours = ds.index[ds["demand_mwh"].isna()]
    st.markdown(f"**Hours with no demand value: {len(nan_hours)}**")
    if len(nan_hours):
        st.dataframe(pd.DataFrame({"hour (UTC)": nan_hours.astype(str)}))
    st.caption("EIA revises past values. This dataset holds today's version of each hour, "
               "not what was first published.")