"""Small, testable data checks used by build_dataset.py."""
import numpy as np
import pandas as pd


def find_duplicates(df, keys):
    return df[df.duplicated(keys, keep=False)]


def find_missing_hours(index):
    """Hourly timestamps absent between the first and last entry of a sorted index."""
    full = pd.date_range(index.min(), index.max(), freq="h")
    return full.difference(index)


def gap_runs(missing):
    """Group missing timestamps into consecutive runs: (start, end, hours)."""
    if len(missing) == 0:
        return []
    missing = missing.sort_values()
    breaks = np.where((missing[1:] - missing[:-1]) > pd.Timedelta(hours=1))[0] + 1
    runs = []
    for chunk in np.split(np.arange(len(missing)), breaks):
        runs.append((missing[chunk[0]], missing[chunk[-1]], len(chunk)))
    return runs


def out_of_range(s, lo, hi):
    return (s < lo) | (s > hi)


def is_utc_hourly(index):
    return (
        index.tz is not None
        and str(index.tz) == "UTC"
        and index.is_monotonic_increasing
        and index.is_unique
    )


def hour_of_peak(s):
    """UTC hour-of-day with the highest mean value."""
    return int(s.groupby(s.index.hour).mean().idxmax())


def in_circular_window(hour, start, end):
    """True if hour is within [start, end] on a 24h clock (window may wrap midnight)."""
    return (hour - start) % 24 <= (end - start) % 24
