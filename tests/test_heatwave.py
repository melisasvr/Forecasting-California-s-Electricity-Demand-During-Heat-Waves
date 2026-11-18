import pandas as pd
import pytest

from src.features import heatwave as hw

DAYS = pd.date_range("2021-01-01", "2023-12-31", freq="D")


def series(hot_days):
    s = pd.Series(20.0, index=DAYS)
    for d in hot_days:
        s[pd.Timestamp(d)] = 40.0
    return s


def test_three_or_more_consecutive_hot_days_form_a_wave():
    s = series(["2023-07-10", "2023-07-11", "2023-07-12", "2023-07-13"])
    out = hw.label_heat_waves(s, 0.90, 3)
    assert out["heat_wave_day"].sum() == 4
    assert out["wave_id"].dropna().nunique() == 1


def test_two_hot_days_are_not_a_wave():
    s = series(["2022-07-10", "2022-07-11"])
    out = hw.label_heat_waves(s, 0.90, 3)
    assert out["hot_day"].sum() == 2
    assert out["heat_wave_day"].sum() == 0


def test_missing_day_breaks_a_run():
    s = series([f"2023-07-{d}" for d in range(10, 15)])
    s = s.drop(pd.Timestamp("2023-07-12"))
    out = hw.label_heat_waves(s, 0.90, 3)
    assert out["heat_wave_day"].sum() == 0  # split into runs of 2 and 2


def test_two_separate_waves_get_two_ids():
    s = series([f"2023-07-{d}" for d in (5, 6, 7)] + [f"2023-08-{d}" for d in (20, 21, 22, 23)])
    out = hw.label_heat_waves(s, 0.90, 3)
    assert out["wave_id"].dropna().nunique() == 2
    summary = hw.summarize_waves(out)
    assert summary["days"].tolist() == [3, 4]


def test_threshold_is_per_calendar_month():
    s = pd.Series(10.0, index=DAYS)  # cold baseline
    s[s.index.month == 7] = 30.0     # July is always warm
    for d in ("2023-01-10", "2023-01-11", "2023-01-12"):
        s[pd.Timestamp(d)] = 20.0    # warm FOR JANUARY, cooler than any July day
    out = hw.label_heat_waves(s, 0.90, 3)
    assert out.loc["2023-01-11", "heat_wave_day"]
    assert not out.loc["2023-07-11", "heat_wave_day"]


def test_daily_index_is_population_weighted_daily_max():
    idx = pd.date_range("2024-06-10", periods=48, freq="h", tz="UTC")
    ds = pd.DataFrame(index=idx)
    ds["era5_temperature_2m__a_city"] = 10.0
    ds["era5_temperature_2m__b_town"] = 20.0
    ds.loc[idx[5], "era5_temperature_2m__a_city"] = 30.0   # 10 June 2024, 22:00 local the day before
    locs = [{"name": "A City", "weight": 0.75}, {"name": "B Town", "weight": 0.25}]
    out = hw.daily_temperature_index(ds, locs, "UTC", min_hours=24)
    assert out.iloc[0] == pytest.approx(0.75 * 30 + 0.25 * 20)
    assert out.iloc[1] == pytest.approx(0.75 * 10 + 0.25 * 20)


def test_partial_days_are_dropped():
    idx = pd.date_range("2024-06-10 12:00", periods=30, freq="h", tz="UTC")
    ds = pd.DataFrame({"era5_temperature_2m__a_city": 10.0}, index=idx)
    out = hw.daily_temperature_index(ds, [{"name": "A City", "weight": 1.0}], "UTC", min_hours=23)
    assert len(out) == 0 or (out.index >= pd.Timestamp("2024-06-11")).all()


def test_cooling_season_flag_and_wave_kinds():
    s = series([f"2023-07-{d}" for d in (10, 11, 12)]            # inside May-Oct
               + [f"2023-01-{d}" for d in (10, 11, 12)]          # off-season
               + ["2023-10-30", "2023-10-31", "2023-11-01", "2023-11-02"])  # straddles
    out = hw.label_heat_waves(s, 0.90, 3)
    assert out.loc["2023-07-11", "cooling_season_heat_wave_day"]
    assert out.loc["2023-01-11", "heat_wave_day"] and not out.loc["2023-01-11", "cooling_season_heat_wave_day"]
    assert out.loc["2023-10-31", "cooling_season_heat_wave_day"]
    assert not out.loc["2023-11-01", "cooling_season_heat_wave_day"]
    kinds = hw.summarize_waves(out).set_index("start")["kind"]
    assert kinds[pd.Timestamp("2023-07-10")] == "cooling-season"
    assert kinds[pd.Timestamp("2023-01-10")] == "off-season"
    assert kinds[pd.Timestamp("2023-10-30")] == "straddles season edge"


def test_october_and_may_count_but_april_does_not():
    s = series([f"2023-10-{d}" for d in (5, 6, 7)]
               + [f"2023-05-{d}" for d in (8, 9, 10)]
               + [f"2023-04-{d}" for d in (8, 9, 10)])
    out = hw.label_heat_waves(s, 0.90, 3)
    assert out.loc["2023-10-06", "cooling_season_heat_wave_day"]
    assert out.loc["2023-05-09", "cooling_season_heat_wave_day"]
    assert out.loc["2023-04-09", "heat_wave_day"] and not out.loc["2023-04-09", "cooling_season_heat_wave_day"]


def test_headline_months_are_configurable():
    s = series([f"2023-01-{d}" for d in (10, 11, 12)])
    out = hw.label_heat_waves(s, 0.90, 3, headline_months=[1])
    assert out["cooling_season_heat_wave_day"].sum() == 3


def test_summary_of_no_waves_has_expected_columns():
    out = hw.label_heat_waves(series([]), 0.90, 3)
    assert list(hw.summarize_waves(out).columns)[-1] == "kind"
