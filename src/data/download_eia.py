"""Download CAISO hourly demand from the EIA-930 API (v2) into data/raw.

Raw rows are saved exactly as returned (strings included), plus the retrieval
time, so revisions can be tracked by re-running and comparing snapshots.
"""
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import yaml

URL = "https://api.eia.gov/v2/electricity/rto/region-data/data/"


def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def fetch_demand(api_key, respondent, dtype, start, end=None, page=5000):
    session = requests.Session()
    rows, offset = [], 0
    while True:
        params = {
            "api_key": api_key,
            "frequency": "hourly",
            "data[0]": "value",
            "facets[respondent][]": respondent,
            "facets[type][]": dtype,
            "start": start,
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "offset": offset,
            "length": page,
        }
        if end:
            params["end"] = end
        r = session.get(URL, params=params, timeout=60)
        r.raise_for_status()
        resp = r.json()["response"]
        data = resp.get("data", [])
        rows.extend(data)
        total = int(resp.get("total", 0))
        offset += page
        print(f"  fetched {min(offset, total)}/{total}")
        if not data or offset >= total:
            break
        time.sleep(0.25)
    return pd.DataFrame(rows)


def main():
    cfg = load_config()
    key = os.environ.get(cfg["eia"]["api_key_env"])
    if not key:
        raise SystemExit(f"Set the {cfg['eia']['api_key_env']} environment variable first.")
    df = fetch_demand(key, cfg["eia"]["respondent"], cfg["eia"]["type"], cfg["eia"]["start"])
    df["source"] = "EIA-930 region-data"
    df["retrieved_at_utc"] = datetime.now(timezone.utc).isoformat()
    out = Path(cfg["paths"]["raw"]) / "eia_caiso_demand.parquet"
    df.to_parquet(out, index=False)
    print(f"Saved {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
