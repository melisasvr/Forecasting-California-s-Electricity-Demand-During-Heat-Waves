"""Forecast metrics. All functions take array-likes of equal length."""
import numpy as np


def _arr(x):
    return np.asarray(x, dtype=float)


def mae(y, yhat):
    y, yhat = _arr(y), _arr(yhat)
    return float(np.mean(np.abs(y - yhat)))


def rmse(y, yhat):
    y, yhat = _arr(y), _arr(yhat)
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def mape(y, yhat):
    """Percent. Demand is never near zero for CAISO, but guard anyway."""
    y, yhat = _arr(y), _arr(yhat)
    if np.any(y == 0):
        raise ValueError("MAPE undefined when y contains zeros")
    return float(np.mean(np.abs((y - yhat) / y)) * 100)


def smape(y, yhat):
    y, yhat = _arr(y), _arr(yhat)
    denom = (np.abs(y) + np.abs(yhat)) / 2
    return float(np.mean(np.abs(y - yhat) / denom) * 100)


def peak_hour_mae(y, yhat, local_hours, peak_hours=(16, 17, 18, 19, 20, 21)):
    """MAE restricted to pre-specified peak hours (local Pacific hour-of-day)."""
    y, yhat = _arr(y), _arr(yhat)
    mask = np.isin(np.asarray(local_hours), list(peak_hours))
    if not mask.any():
        raise ValueError("No peak hours present")
    return mae(y[mask], yhat[mask])


def pinball_loss(y, q_pred, q):
    y, q_pred = _arr(y), _arr(q_pred)
    diff = y - q_pred
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def interval_coverage(y, lower, upper):
    y, lower, upper = _arr(y), _arr(lower), _arr(upper)
    return float(np.mean((y >= lower) & (y <= upper)))


# ---------------------------------------------------------------------------------------
# Breakdown tables used by the report
# ---------------------------------------------------------------------------------------
import pandas as pd  # noqa: E402

SEASONS = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
           6: "Summer", 7: "Summer", 8: "Summer", 9: "Fall", 10: "Fall", 11: "Fall"}


def _bin_labels(edges):
    labs = [f"<{edges[0]}"] + [f"{a}-{b}" for a, b in zip(edges[:-1], edges[1:])] + [f">={edges[-1]}"]
    return labs


def annotate(preds, ds, day_type_series, cfg):
    """Add grouping columns (local hour, weekday, season, temperature bin, day type) to predictions."""
    out = preds.copy()
    tz = cfg["evaluation"]["local_timezone"]
    t = pd.DatetimeIndex(out["target_utc"])
    local = t.tz_convert(tz)
    out["local_hour"] = local.hour
    out["dow"] = local.day_of_week
    out["day_kind"] = np.where(out["dow"] >= 5, "Weekend", "Weekday")
    out["season"] = local.month.map(SEASONS)
    edges = cfg["features"]["temp_bin_edges_c"]
    temp = ds["era5_temperature_2m__ca_pw"].reindex(t).to_numpy()
    cut = pd.cut(temp, [-np.inf] + list(edges) + [np.inf], labels=_bin_labels(edges), right=False)
    out["temp_bin"] = pd.Series(cut, index=out.index).astype(object)
    out["day_type"] = out["target_local_date"].map(day_type_series)
    return out


def _metric_row(g, peak_hours):
    y, p = g["y"].to_numpy(float), g["yhat"].to_numpy(float)
    row = {"n": int(len(g)), "MAE": mae(y, p), "RMSE": rmse(y, p), "MAPE_pct": mape(y, p), "sMAPE_pct": smape(y, p)}
    mask = g["local_hour"].isin(list(peak_hours)).to_numpy()
    row["PeakHourMAE"] = mae(y[mask], p[mask]) if mask.any() else np.nan
    if g[["q10", "q50", "q90"]].notna().all().all():
        for col, q in (("q10", 0.1), ("q50", 0.5), ("q90", 0.9)):
            row[f"pinball_{int(q * 100)}"] = pinball_loss(y, g[col].to_numpy(float), q)
        row["coverage_80pct"] = interval_coverage(y, g["q10"].to_numpy(float), g["q90"].to_numpy(float))
        row["mean_width"] = float((g["q90"] - g["q10"]).mean())
    return row


def metric_table(df, by, peak_hours):
    """Metrics for every combination of the `by` columns (always include 'model')."""
    by = [by] if isinstance(by, str) else list(by)
    rows = []
    for key, g in df.dropna(subset=["yhat", "y"]).groupby(by, dropna=True, observed=True):
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(by, key)), **_metric_row(g, peak_hours)})
    return pd.DataFrame(rows)


def common_rows(preds):
    """Keep only (period, origin, horizon) rows for which every model produced a forecast."""
    models = preds["model"].nunique()
    key = preds["period"].astype(str) + "|" + preds["origin_utc"].astype(str) + "|" + preds["horizon_step"].astype(str)
    n = preds.assign(_k=key).groupby("_k")["model"].transform("nunique")
    return preds[n == models].copy()
