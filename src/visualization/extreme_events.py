"""Rule-based selection of windows to plot. Selection never looks at forecast errors."""
import pandas as pd

from src.features import heatwave as hw


def pick_heat_window(waves, evaluated_dates, pad_days=2):
    """The cooling-season wave with the highest peak temperature index that lies fully in the evaluated period."""
    if waves is None or waves.empty:
        return None
    ok = waves[(waves["kind"] != "off-season")
               & waves["start"].isin(evaluated_dates) & waves["end"].isin(evaluated_dates)]
    if ok.empty:
        return None
    w = ok.sort_values("peak_temp_index_c", ascending=False).iloc[0]
    return w["start"] - pd.Timedelta(days=pad_days), w["end"] + pd.Timedelta(days=pad_days), w


def pick_normal_week(day_type, evaluated_dates):
    """Latest July/August week (Mon-Sun style 7-day block) in which every day is a 'Normal day'."""
    normal = set(day_type[day_type == hw.DAY_TYPE_LABELS[3]].index) & set(evaluated_dates)
    for year in sorted({d.year for d in evaluated_dates}, reverse=True):
        for start in pd.date_range(f"{year}-07-10", f"{year}-09-10", freq="7D"):
            days = set(pd.date_range(start, periods=7, freq="D"))
            if days <= normal:
                return start, start + pd.Timedelta(days=6)
    return None
