"""Run the dashboard on fake data and make sure it neither crashes nor prints stray output."""
from pathlib import Path

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "src" / "app" / "streamlit_app.py")


def fake_dataset():
    idx = pd.date_range("2021-01-01", "2023-12-31 23:00", freq="h", tz="UTC")
    rng = np.random.default_rng(1)
    doy, h = idx.dayofyear.values, idx.hour.values
    base = 18 - 8 * np.cos(2 * np.pi * (doy - 15) / 365) + 6 * np.sin(2 * np.pi * (h - 16) / 24)
    shock = ((idx >= "2022-09-02") & (idx < "2022-09-10")).astype(float) * 9   # summer wave
    shock += ((idx >= "2022-01-10") & (idx < "2022-01-16")).astype(float) * 8  # off-season spell
    ds, tot = pd.DataFrame(index=idx), 0
    cities = zip(["los_angeles", "san_diego", "fresno", "sacramento", "bay_area"],
                 [0.45, 0.15, 0.07, 0.11, 0.22])
    for name, wt in cities:
        t = base + rng.normal(0, 0.5, len(idx)) + shock
        ds[f"era5_temperature_2m__{name}"] = t
        ds[f"fc_temperature_2m__{name}"] = t + rng.normal(0, 1.2, len(idx))
        tot = tot + wt * t
    ds["era5_temperature_2m__ca_pw"] = tot
    ds["fc_temperature_2m__ca_pw"] = tot + rng.normal(0, 1, len(idx))
    ds["demand_mwh"] = 24000 + 700 * np.maximum(tot - 20, 0) + 3000 * np.sin(2 * np.pi * (h - 1) / 24)
    ds.loc[ds.index[100:104], "demand_mwh"] = np.nan
    return ds


def test_dashboard_runs_cleanly_on_fake_data(tmp_path, monkeypatch):
    fake_dataset().to_parquet(tmp_path / "dataset.parquet")
    monkeypatch.setenv("CAISO_PROCESSED_DIR", str(tmp_path))
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.get("help_info")) == 0, "stray object printed to the page (a bare expression?)"
    labels = [m.label for m in at.metric]
    assert "Cooling-season heat waves" in labels


def test_stray_output_detector_actually_fires(tmp_path):
    """Self-test: the exact pattern that bit us must be caught by the check above."""
    script = tmp_path / "bad_app.py"
    script.write_text(
        "import streamlit as st, pandas as pd\n"
        "df = pd.DataFrame({'a': [1]})\n"
        "st.dataframe(df) if True else st.dataframe(df)\n"
    )
    at = AppTest.from_file(str(script)).run()
    assert len(at.get("help_info")) == 1
