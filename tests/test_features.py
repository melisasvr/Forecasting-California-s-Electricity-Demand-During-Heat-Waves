import numpy as np
import pandas as pd
import pytest

from src.features import build_features as bf

CFG = {
    "evaluation": {"horizon_hours": 24, "local_timezone": "America/Los_Angeles"},
    "features": {
        "demand_lags_hours": [24, 48, 72, 168, 336],
        "rolling_windows_hours": [24, 72, 168],
        "degree_base_c": 18.3,
        "temp_bin_edges_c": [10, 15, 20, 25, 30, 35, 40],
        "holiday_cap_days": 7,
    },
}


@pytest.fixture(scope="module")
def ds():
    idx = pd.date_range("2023-11-01", periods=24 * 160, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    h = idx.hour.values
    d = pd.DataFrame(index=idx)
    d["demand_mwh"] = 20000.0 + np.arange(len(idx))  # unique per hour, easy to trace
    d["fc_temperature_2m__ca_pw"] = 20 + 8 * np.sin(2 * np.pi * (h - 16) / 24)
    d["fc_relative_humidity_2m__ca_pw"] = 50 + rng.normal(0, 3, len(idx))
    d["era5_temperature_2m__ca_pw"] = 20 + 8 * np.sin(2 * np.pi * (h - 16) / 24) + rng.normal(0, 1, len(idx))
    d["era5_relative_humidity_2m__ca_pw"] = 50 + rng.normal(0, 3, len(idx))
    return d


@pytest.fixture(scope="module")
def built(ds):
    return bf.build_features(ds, CFG)


def test_origins_are_local_midnight_and_24_rows_each(built):
    out, _, _ = built
    local = pd.DatetimeIndex(out["origin_utc"].unique()).tz_convert("America/Los_Angeles")
    assert (local.hour == 0).all() and (local.minute == 0).all()
    assert (out.groupby("origin_utc").size() == 24).all()
    assert set(out["horizon_step"]) == set(range(24))


def test_targets_follow_origin_consecutively_through_dst(built):
    out, _, _ = built
    one = out[out["origin_utc"] == pd.Timestamp("2024-03-10 08:00", tz="UTC")]  # spring-forward day
    assert len(one) == 24
    assert ((one["target_utc"] - one["origin_utc"]).dt.total_seconds() / 3600).tolist() == list(range(24))
    nxt = pd.Timestamp("2024-03-11 07:00", tz="UTC")  # next local midnight, now PDT
    assert (nxt in set(out["origin_utc"]))


@pytest.mark.parametrize("L", [24, 48, 72, 168, 336])
def test_lag_features_equal_demand_at_target_minus_lag(ds, built, L):
    out, _, _ = built
    expected = ds["demand_mwh"].reindex(pd.DatetimeIndex(out["target_utc"] - pd.Timedelta(hours=L)))
    assert np.allclose(out[f"demand_lag_{L}h"].to_numpy(), expected.to_numpy())


def test_all_demand_lags_are_older_than_the_origin(built):
    out, _, _ = built
    for L in CFG["features"]["demand_lags_hours"]:
        used = out["target_utc"] - pd.Timedelta(hours=L)
        assert (used <= out["origin_utc"] - pd.Timedelta(hours=1)).all()


def test_rolling_mean_uses_only_the_24h_before_origin(ds, built):
    out, _, _ = built
    o = out["origin_utc"].iloc[500]
    row = out[(out["origin_utc"] == o) & (out["horizon_step"] == 0)].iloc[0]
    window = ds["demand_mwh"].loc[o - pd.Timedelta(hours=24): o - pd.Timedelta(hours=1)]
    assert len(window) == 24
    assert row["demand_roll_mean_24h"] == pytest.approx(window.mean())
    assert row["demand_roll_std_24h"] == pytest.approx(window.std())


def test_no_future_demand_leaks_into_features(ds, built):
    """Wreck all demand from T0 onward. Features of origins <= T0 must not move."""
    out, cols, _ = built
    T0 = pd.Timestamp("2024-02-10 08:00", tz="UTC")
    ds2 = ds.copy()
    ds2.loc[ds2.index >= T0, "demand_mwh"] = 1e6
    out2, cols2, _ = bf.build_features(ds2, CFG)
    a = out.loc[out["origin_utc"] <= T0, cols].reset_index(drop=True)
    b = out2.loc[out2["origin_utc"] <= T0, cols2].reset_index(drop=True)
    assert len(a) > 1000
    pd.testing.assert_frame_equal(a, b)


def test_operational_features_never_touch_era5(ds, built):
    out, cols, _ = built
    assert not any(c.startswith("era5_") for c in cols)
    ds2 = ds.copy()
    for c in [c for c in ds2.columns if c.startswith("era5_")]:
        ds2[c] = ds2[c] * 1000
    out2, cols2, _ = bf.build_features(ds2, CFG)
    pd.testing.assert_frame_equal(out[cols], out2[cols2])


def test_perfect_weather_build_uses_era5_only(ds):
    out, cols, info = bf.build_features(ds, CFG, weather="era5")
    assert not any(c.startswith("fc_") for c in cols)
    assert info["weather_source"] == "era5"


def test_feature_matrix_is_finite_with_no_nans(built):
    out, cols, _ = built
    assert not out[cols + ["y"]].isna().any().any()
    assert np.isfinite(out[cols].to_numpy(dtype=float)).all()


def test_holiday_flags_and_proximity(built):
    out, _, _ = built
    day = lambda s: out[out["target_local_date"] == pd.Timestamp(s)].iloc[0]
    assert day("2023-12-25")["is_holiday"] == 1
    assert day("2023-12-24")["days_to_next_holiday"] == 1
    assert day("2023-12-26")["days_since_prev_holiday"] == 1


def test_lags_shorter_than_horizon_are_rejected(ds):
    bad = {**CFG, "features": {**CFG["features"], "demand_lags_hours": [12, 24]}}
    with pytest.raises(ValueError):
        bf.build_features(ds, bad)
