"""
update_dashboard_data.py
========================
End-to-end pipeline to refresh the Streamlit dashboard with the latest
hotspot data from NASA FIRMS.

Steps:
  1. Download new hotspot data from NASA FIRMS API (2025 -> today)
  2. Merge with the existing raw CSV  (data/raw/perlu_diolah_untukTA.csv)
  3. Save the merged result back to the raw CSV (in-place update)
  4. Re-run data_processing pipeline to regenerate:
       - data/processed/daily_hotspot_mapping_2014_<current_year>.csv
       - monthly_hotspot_sum.csv  (root-level file read by the dashboard)

Run from the PROJECT ROOT:
    python src/update_dashboard_data.py

Requirements:
    pip install requests pandas
"""

import sys
import os
import shutil
from pathlib import Path
from datetime import date, timedelta
import time

import requests
import pandas as pd

# ── Ensure we can import local src modules ──────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent   # project root
SRC  = ROOT / "src"
sys.path.insert(0, str(SRC))

from data_processing import create_daily_hotspot_mapping, create_monthly_hotspot_data

# ── CONFIG ───────────────────────────────────────────────────────────────────
MAP_KEY = "2934538155d848c5fb2b71e428c61406"

# Paths (all relative to project root)
RAW_CSV           = ROOT / "data" / "raw"  / "perlu_diolah_untukTA.csv"
TILE_CSV          = ROOT / "pontianak_tile_boundaries.csv"
DAILY_OUTPUT      = ROOT / "data" / "processed" / "daily_hotspot_mapping_2014_now.csv"
MONTHLY_PROCESSED = ROOT / "data" / "processed" / "monthly_hotspot_sum.csv"
MONTHLY_ROOT      = ROOT / "monthly_hotspot_sum.csv"   # file the dashboard actually reads

# Bounding box: Kubu Raya, Kalimantan Barat
W, S, E, N = 109.0, -1.5, 110.5, 0.5

# Download range: 2025-01-01 -> today
START_DATE = date(2025, 1, 1)
END_DATE   = date.today()

SATELLITES = [
    "VIIRS_SNPP_NRT",
    "VIIRS_NOAA20_NRT",
    "MODIS_NRT",
]
# ─────────────────────────────────────────────────────────────────────────────


def fetch_firms_area(satellite: str, start: date, end: date) -> pd.DataFrame:
    """Download hotspot data from NASA FIRMS (max 10 days per request, chunked)."""
    all_frames = []
    chunk_start = start

    while chunk_start <= end:
        chunk_end = min(chunk_start + timedelta(days=4), end)
        days      = (chunk_end - chunk_start).days + 1
        date_str  = chunk_start.strftime("%Y-%m-%d")

        url = (
            f"https://firms.modaps.eosdis.nasa.gov/api/area/csv"
            f"/{MAP_KEY}/{satellite}/{W},{S},{E},{N}/{days}/{date_str}"
        )

        print(f"  Fetching {satellite} | {chunk_start} -> {chunk_end} ...", end=" ")
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200 and len(r.text) > 50:
                from io import StringIO
                df = pd.read_csv(StringIO(r.text))
                all_frames.append(df)
                print(f"OK {len(df)} rows")
            else:
                print(f"FAILED HTTP {r.status_code}")
        except Exception as e:
            print(f"ERROR: {e}")

        chunk_start = chunk_end + timedelta(days=1)
        time.sleep(0.5)

    return pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()


def standardize_firms(df: pd.DataFrame, satellite: str) -> pd.DataFrame:
    """Normalize FIRMS columns to match the existing raw CSV format."""
    if df.empty:
        return df

    out = pd.DataFrame()

    if "acq_date" in df.columns:
        # Convert FIRMS date (YYYY-MM-DD) -> DD-MM-YYYY to match existing CSV
        out["Tanggal"] = pd.to_datetime(df["acq_date"]).dt.strftime("%d-%m-%Y")

    out["Latitude"]  = df.get("latitude",  pd.Series(dtype=float))
    out["Longitude"] = df.get("longitude", pd.Series(dtype=float))

    return out


def download_new_data() -> pd.DataFrame:
    """Download and standardize new hotspot data from all satellites."""
    print(f"\n{'='*60}")
    print(f"Downloading Kubu Raya hotspot data: {START_DATE} -> {END_DATE}")
    print(f"{'='*60}\n")

    all_new = []
    for sat in SATELLITES:
        print(f"\n[{sat}]")
        df_raw = fetch_firms_area(sat, START_DATE, END_DATE)
        if not df_raw.empty:
            df_std = standardize_firms(df_raw, sat)
            all_new.append(df_std)

    if not all_new:
        print("\nNo data downloaded. Check your API key and internet connection.")
        return pd.DataFrame()

    new_data = pd.concat(all_new, ignore_index=True)
    new_data.drop_duplicates(subset=["Tanggal", "Latitude", "Longitude"], inplace=True)
    new_data.sort_values("Tanggal", inplace=True)
    print(f"\nDownloaded {len(new_data)} unique new hotspot records.")
    return new_data


def merge_with_existing(new_data: pd.DataFrame) -> int:
    """
    Merge new hotspot rows into the existing raw CSV.
    Returns the number of new rows actually added.
    """
    print(f"\nMerging with existing raw CSV: {RAW_CSV}")
    existing = pd.read_csv(RAW_CSV)
    print(f"  Existing rows : {len(existing)}")

    # Normalise column names to match
    existing.columns = existing.columns.str.strip()
    new_data.columns = new_data.columns.str.strip()

    merged = pd.concat([existing, new_data[["Tanggal", "Latitude", "Longitude"]]], ignore_index=True)
    merged.drop_duplicates(subset=["Tanggal", "Latitude", "Longitude"], inplace=True)
    merged.sort_values("Tanggal", inplace=True)

    rows_added = len(merged) - len(existing)
    merged.to_csv(RAW_CSV, index=False)
    print(f"  New rows added: {rows_added}")
    print(f"  Total rows    : {len(merged)}")
    print(f"  Raw CSV updated in-place: {RAW_CSV}")
    return rows_added


def rebuild_aggregations():
    """Re-run the data processing pipeline to regenerate monthly_hotspot_sum.csv."""
    current_year = date.today().year
    print(f"\n{'='*60}")
    print(f"Rebuilding aggregations (2014 -> {current_year}) ...")
    print(f"{'='*60}\n")

    # Ensure output directories exist
    DAILY_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    MONTHLY_PROCESSED.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: Daily mapping
    create_daily_hotspot_mapping(
        raw_csv=str(RAW_CSV),
        tile_csv=str(TILE_CSV),
        output_csv=str(DAILY_OUTPUT),
        start_year=2014,
        end_year=current_year,
    )

    # Step 2: Monthly aggregation → data/processed/
    create_monthly_hotspot_data(
        daily_csv=str(DAILY_OUTPUT),
        output_csv=str(MONTHLY_PROCESSED),
    )

    # Step 3: Copy to root-level so dashboard can read it
    shutil.copy2(MONTHLY_PROCESSED, MONTHLY_ROOT)
    print(f"\nDashboard file updated: {MONTHLY_ROOT}")


def main():
    if MAP_KEY == "YOUR_FIRMS_API_KEY_HERE":
        print("WARNING: Please set your NASA FIRMS API key in MAP_KEY!")
        print("   Get one free at: https://firms.modaps.eosdis.nasa.gov/api/")
        sys.exit(1)

    # 1. Download
    new_data = download_new_data()

    # 2. Merge (even if 0 new rows, re-run aggregation in case raw CSV was edited)
    if new_data.empty:
        print("\n⚠️  No new data to merge. Re-running aggregation on existing raw CSV anyway.")
        rows_added = 0
    else:
        rows_added = merge_with_existing(new_data)

    # 3. Rebuild aggregations
    rebuild_aggregations()

    print(f"\n{'='*60}")
    print(f"Dashboard data update complete!")
    print(f"   New hotspot records added : {rows_added}")
    print(f"   Dashboard file            : {MONTHLY_ROOT}")
    print(f"   Restart the Streamlit app to see updated data.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
