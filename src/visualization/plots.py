"""Static, publication-style charts (matplotlib). Every function saves a PNG and returns its path."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PRIMARY, BASELINE = "lightgbm", "baseline_weekly"
COLORS = {"actual": "#222222", BASELINE: "#7f8c8d", PRIMARY: "#d62728", "ridge": "#1f77b4",
          "sarimax": "#2ca02c", "baseline_yearly": "#bbbbbb", "baseline_mean7": "#999999"}
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})


def _dates(ax):
    loc = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _local(preds, tz):
    out = preds.copy()
    out["t_local"] = pd.DatetimeIndex(out["target_utc"]).tz_convert(tz).tz_localize(None)
    return out.sort_values("t_local")


def fig_model_comparison(table, path):
    """table: rows model x period with MAE and PeakHourMAE."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, period in zip(axes, ["validation", "holdout"]):
        t = table[table["period"] == period].sort_values("MAE")
        ax.barh(t["model"], t["MAE"], color=[COLORS.get(m, "#555") for m in t["model"]])
        for y, v in enumerate(t["MAE"]):
            ax.text(v, y, f" {v:,.0f}", va="center", fontsize=9)
        ax.set_title(f"{period.capitalize()} (MAE, MWh per hour)")
        ax.invert_yaxis()
    return _save(fig, path)


def fig_actual_vs_predicted(preds, window, title, path, tz, models=(BASELINE, PRIMARY)):
    s, e = window
    d = _local(preds[(preds["target_local_date"] >= s) & (preds["target_local_date"] <= e)], tz)
    fig, ax = plt.subplots(figsize=(11, 4))
    a = d[d["model"] == PRIMARY]
    ax.plot(a["t_local"], a["y"], color=COLORS["actual"], lw=1.8, label="Actual")
    for m in models:
        g = d[d["model"] == m]
        ax.plot(g["t_local"], g["yhat"], color=COLORS[m], lw=1.3, ls="--" if m == BASELINE else "-", label=m)
        if m == PRIMARY and g["q10"].notna().any():
            ax.fill_between(g["t_local"], g["q10"], g["q90"], color=COLORS[m], alpha=0.15, label="10-90% interval")
    ax.set_ylabel("Demand (MWh per hour)")
    ax.set_title(title)
    ax.legend(ncol=4, frameon=False, loc="upper left")
    _dates(ax)
    return _save(fig, path)


def fig_case_study(preds, ds, window, path, tz, title):
    s, e = window
    d = _local(preds[(preds["target_local_date"] >= s) & (preds["target_local_date"] <= e)], tz)
    a, b = d[d["model"] == PRIMARY], d[d["model"] == BASELINE]
    fig, ax = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True, gridspec_kw={"height_ratios": [3, 1.6, 1.6]})
    ax[0].plot(a["t_local"], a["y"], color=COLORS["actual"], lw=1.8, label="Actual")
    ax[0].plot(b["t_local"], b["yhat"], color=COLORS[BASELINE], ls="--", lw=1.2, label=BASELINE)
    ax[0].plot(a["t_local"], a["yhat"], color=COLORS[PRIMARY], lw=1.3, label=PRIMARY)
    if a["q10"].notna().any():
        ax[0].fill_between(a["t_local"], a["q10"], a["q90"], color=COLORS[PRIMARY], alpha=0.15, label="10-90% interval")
    ax[0].set_ylabel("Demand (MWh)")
    ax[0].set_title(title)
    ax[0].legend(ncol=4, frameon=False, loc="upper left")
    t_idx = pd.DatetimeIndex(a["target_utc"])
    ax[1].plot(a["t_local"], ds["era5_temperature_2m__ca_pw"].reindex(t_idx).to_numpy(), color="#d62728", label="ERA5 (what happened)")
    ax[1].plot(a["t_local"], ds["fc_temperature_2m__ca_pw"].reindex(t_idx).to_numpy(), color="#ff9f43", ls=":", label="Archived forecast (model input)")
    ax[1].set_ylabel("Temp index (C)")
    ax[1].legend(frameon=False, ncol=2, loc="upper left")
    ax[2].axhline(0, color="#999", lw=0.8)
    ax[2].plot(a["t_local"], a["y"] - a["yhat"], color=COLORS[PRIMARY], lw=1.1, label=f"{PRIMARY} error")
    ax[2].plot(b["t_local"], b["y"] - b["yhat"], color=COLORS[BASELINE], lw=1.0, ls="--", label=f"{BASELINE} error")
    ax[2].set_ylabel("Actual - forecast (MWh)")
    ax[2].legend(frameon=False, ncol=2, loc="upper left")
    _dates(ax[2])
    return _save(fig, path)


def fig_error_heatmap(ann, path, model=PRIMARY):
    g = ann[ann["model"] == model].assign(abs_err=lambda d: (d["y"] - d["yhat"]).abs())
    piv = g.pivot_table(index="local_hour", columns="dow", values="abs_err", aggfunc="mean").reindex(columns=range(7))
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(piv.to_numpy(), aspect="auto", cmap="magma_r", origin="lower")
    ax.set_xticks(range(7), DOW)
    ax.set_yticks(range(0, 24, 2), range(0, 24, 2))
    ax.set_xlabel("Day of week (target)")
    ax.set_ylabel("Hour of day, Pacific (hour beginning)")
    ax.set_title(f"{model}: mean absolute error by hour and weekday")
    fig.colorbar(im, ax=ax, label="MAE (MWh)")
    return _save(fig, path)


def fig_residuals(ann, order_bins, path, model=PRIMARY):
    g = ann[ann["model"] == model].assign(res=lambda d: d["y"] - d["yhat"])
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    tb = g.groupby("temp_bin")["res"].agg(["mean", "std", "count"]).reindex(order_bins).dropna()
    ax[0].bar(tb.index, tb["mean"], yerr=tb["std"], color="#d62728", alpha=0.75, capsize=3)
    ax[0].axhline(0, color="#444", lw=0.8)
    ax[0].set_title("Residual by temperature bin (C)")
    ax[0].tick_params(axis="x", rotation=45)
    hr = g.groupby("local_hour")["res"].agg(["mean", "std"])
    ax[1].bar(hr.index, hr["mean"], yerr=hr["std"], color="#1f77b4", alpha=0.75)
    ax[1].axhline(0, color="#444", lw=0.8)
    ax[1].set_title("Residual by hour of day")
    dw = g.groupby("dow")["res"].agg(["mean", "std"]).reindex(range(7))
    ax[2].bar(DOW, dw["mean"], yerr=dw["std"], color="#2ca02c", alpha=0.75)
    ax[2].axhline(0, color="#444", lw=0.8)
    ax[2].set_title("Residual by day of week")
    ax[0].set_ylabel("Actual - forecast (MWh), mean and sd")
    fig.suptitle(f"{model}: residual analysis (validation and holdout)", y=1.02)
    return _save(fig, path)


def fig_importance(model, X, path, temp_col="fc_temperature_2m__ca_pw", top=20):
    contrib = model.contributions(X)[:, :-1]
    imp = pd.Series(np.abs(contrib).mean(axis=0), index=model.features).sort_values().tail(top)
    fig, ax = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [1.2, 1]})
    ax[0].barh(imp.index, imp.to_numpy(), color="#d62728")
    ax[0].set_xlabel("Mean |SHAP value| (MWh)")
    ax[0].set_title(f"Top {top} features (LightGBM TreeSHAP)")
    if temp_col in model.features:
        j = model.features.index(temp_col)
        sc = ax[1].scatter(X[temp_col], contrib[:, j], c=X["hour"], cmap="viridis", s=6, alpha=0.6)
        fig.colorbar(sc, ax=ax[1], label="Hour of day")
        ax[1].set_xlabel("Forecast temperature index (C)")
        ax[1].set_ylabel("SHAP value of temperature (MWh)")
        ax[1].set_title("How temperature moves the forecast")
    return _save(fig, path)


def fig_locations(locations, path):
    fig, ax = plt.subplots(figsize=(6.5, 7))
    for loc in locations:
        ax.scatter(loc["lon"], loc["lat"], s=loc["weight"] * 6000, alpha=0.45, color="#1f77b4")
        ax.annotate(f"{loc['name']}\n{loc['weight']:.1%}", (loc["lon"], loc["lat"]), ha="center", va="center", fontsize=9)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Weather locations and population weights\n(2020 Census county populations)")
    ax.set_aspect(1.25)
    return _save(fig, path)
