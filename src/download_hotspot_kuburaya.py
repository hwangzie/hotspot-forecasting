"""
Download hotspot data for Kubu Raya, Kalimantan Barat (2025 - present)
from NASA FIRMS API and append to your existing CSV.

Requirements:
    pip install requests pandas

Steps:
    1. Get a free NASA FIRMS API key at:
       https://firms.modaps.eosdis.nasa.gov/api/
    2. Set your API key below (MAP_KEY)
    3. Run: python download_hotspot_kuburaya.py
"""

import requests
import pandas as pd
from datetime import date, timedelta
import time
import os
from pathlib import Path

# Resolve project root (two levels up from this file: src/ → project root)
_ROOT = Path(__file__).resolve().parent.parent

# ── CONFIG ──────────────────────────────────────────────────────────────────
MAP_KEY = "2934538155d848c5fb2b71e428c61406"
EXISTING_CSV = str(_ROOT / "data" / "raw" / "perlu_diolah_untukTA.csv")
OUTPUT_CSV   = str(_ROOT / "data" / "raw" / "hotspot_kuburaya_2025_now.csv")
MERGED_CSV   = str(_ROOT / "data" / "raw" / "perlu_diolah_untukTA_updated.csv")

# Kubu Raya, Kalimantan Barat bounding box
W, S, E, N = 108.9, -1.0, 110.0, 0.1  # lon_min, lat_min, lon_max, lat_max

# Date range: Jan 1 2025 to today
START_DATE = date(2025, 1, 1)
END_DATE   = date.today()

SATELLITES = [
    "MODIS_NRT",
]
# ────────────────────────────────────────────────────────────────────────────


def fetch_firms_area(satellite: str, start: date, end: date) -> pd.DataFrame:
    """
    Fetch hotspot data from NASA FIRMS area API.
    Max 10 days per request — we chunk automatically.
    """
    all_frames = []
    chunk_start = start

    while chunk_start <= end:
        chunk_end = min(chunk_start + timedelta(days=4), end)
        days = (chunk_end - chunk_start).days + 1
        date_str = chunk_start.strftime("%Y-%m-%d")

        url = (
            f"https://firms.modaps.eosdis.nasa.gov/api/area/csv"
            f"/{MAP_KEY}/{satellite}/{W},{S},{E},{N}/{days}/{date_str}"
        )

        print(f"  Fetching {satellite} | {chunk_start} → {chunk_end} ...", end=" ")
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200 and len(r.text) > 50:
                from io import StringIO
                df = pd.read_csv(StringIO(r.text))
                all_frames.append(df)
                print(f"✓ {len(df)} rows")
            else:
                print(f"✗ HTTP {r.status_code}")
        except Exception as e:
            print(f"✗ Error: {e}")

        chunk_start = chunk_end + timedelta(days=1)
        time.sleep(0.5)   # be polite to the API

    return pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()


def standardize_firms(df: pd.DataFrame, satellite: str) -> pd.DataFrame:
    """
    Normalize FIRMS columns to match your existing CSV format.
    Adjust the column mapping below if your CSV has different names.
    """
    if df.empty:
        return df

    # FIRMS columns differ slightly by satellite; handle both MODIS & VIIRS
    out = pd.DataFrame()

    # Date
    if "acq_date" in df.columns:
        out["tanggal"] = pd.to_datetime(df["acq_date"]).dt.strftime("%Y-%m-%d")
    
    out["latitude"]    = df.get("latitude", pd.Series(dtype=float))
    out["longitude"]   = df.get("longitude", pd.Series(dtype=float))
    out["satelit"]     = satellite.split("_")[0] + " " + satellite.split("_")[1]
    
    # Confidence: VIIRS = 'l/n/h', MODIS = 0-100
    if "confidence" in df.columns:
        conf = df["confidence"].astype(str).str.lower()
        mapping = {"l": "Low", "n": "Medium", "h": "High"}
        out["kepercayaan"] = conf.map(mapping).fillna(
            pd.to_numeric(df["confidence"], errors="coerce")
            .apply(lambda x: "High" if x >= 80 else ("Medium" if x >= 50 else "Low"))
        )
    
    out["kabupaten"] = "KUBU RAYA"
    out["provinsi"]  = "KALIMANTAN BARAT"

    # Extra FIRMS fields (keep for reference)
    for col in ["bright_ti4", "bright_ti5", "frp", "daynight", "scan", "track"]:
        if col in df.columns:
            out[col] = df[col]

    return out


def main():
    print(f"\n{'='*60}")
    print(f"Downloading Kubu Raya hotspot data: {START_DATE} → {END_DATE}")
    print(f"{'='*60}\n")

    if MAP_KEY == "YOUR_FIRMS_API_KEY_HERE":
        print("⚠️  Please set your NASA FIRMS API key in MAP_KEY!")
        print("   Get one free at: https://firms.modaps.eosdis.nasa.gov/api/")
        return

    all_new = []
    for sat in SATELLITES:
        print(f"\n[{sat}]")
        df_raw = fetch_firms_area(sat, START_DATE, END_DATE)
        if not df_raw.empty:
            df_std = standardize_firms(df_raw, sat)
            all_new.append(df_std)

    if not all_new:
        print("\n❌ No data downloaded. Check your API key and internet connection.")
        return

    new_data = pd.concat(all_new, ignore_index=True)
    new_data.drop_duplicates(subset=["tanggal", "latitude", "longitude", "satelit"], inplace=True)
    new_data.sort_values("tanggal", inplace=True)

    new_data.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Saved {len(new_data)} new rows → {OUTPUT_CSV}")

    # ── Merge with existing CSV ──────────────────────────────────────────
    if os.path.exists(EXISTING_CSV):
        print(f"\nMerging with existing file: {EXISTING_CSV}")
        existing = pd.read_csv(EXISTING_CSV)
        print(f"  Existing rows : {len(existing)}")

        # Standardize column names (lowercase, strip spaces)
        existing.columns = existing.columns.str.strip().str.lower()
        new_data.columns  = new_data.columns.str.strip().str.lower()

        merged = pd.concat([existing, new_data], ignore_index=True)
        merged.drop_duplicates(subset=["tanggal", "latitude", "longitude", "satelit"], inplace=True)
        merged.sort_values("tanggal", inplace=True)
        merged.to_csv(MERGED_CSV, index=False)
        print(f"  New rows added: {len(merged) - len(existing)}")
        print(f"  Total rows    : {len(merged)}")
        print(f"  ✅ Merged file → {MERGED_CSV}")
    else:
        print(f"\n⚠️  Existing CSV not found at '{EXISTING_CSV}'. Skipping merge.")
        print(f"    Rename {OUTPUT_CSV} manually or update EXISTING_CSV path.")

    print("\nDone! 🎉")


if __name__ == "__main__":
    main()