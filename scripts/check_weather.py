import pandas as pd

VARS = ["temperature_2m", "apparent_temperature", "relative_humidity_2m",
        "dew_point_2m", "wind_speed_10m", "cloud_cover", "precipitation"]

era5 = pd.read_parquet("data/raw/weather_era5.parquet")
fc = pd.read_parquet("data/raw/weather_forecast_archive.parquet")

for name, df in [("era5", era5), ("forecast_archive", fc)]:
    print(f"\n{name}: missing values per variable, by year")
    print(df[VARS].isna().groupby(df["time"].dt.year).sum())

m = era5.merge(fc, on=["time", "location"], suffixes=("_era5", "_fc"))
m["abs_diff"] = (m["temperature_2m_era5"] - m["temperature_2m_fc"]).abs()
print("\nMean abs temperature difference, ERA5 vs archive, by year (C):")
print(m.groupby(m["time"].dt.year)["abs_diff"].mean())
