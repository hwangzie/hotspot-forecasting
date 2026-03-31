"""
Data Processing Module
======================
Convert raw hotspot data → daily mapping → monthly aggregation.
"""

import pandas as pd
import numpy as np
import argparse
import math
from pathlib import Path


def point_in_tile(lat, lon, tile_bounds):
    """Check if a point (lat, lon) is within tile boundaries."""
    return (tile_bounds['lat_bottom_right'] <= lat <= tile_bounds['lat_top_left'] and
            tile_bounds['lon_top_left'] <= lon <= tile_bounds['lon_bottom_right'])


def create_daily_hotspot_mapping(raw_csv, tile_csv, output_csv, start_year=2014, end_year=2024):
    """
    Create daily hotspot mapping for each tile.

    Args:
        raw_csv: path to raw hotspot CSV (perlu_diolah_untukTA.csv).
        tile_csv: path to tile boundaries CSV (pontianak_tile_boundaries.csv).
        output_csv: output path for daily mapping CSV.
        start_year: first year.
        end_year: last year.
    """
    print("Loading tile boundaries...")
    tiles_df = pd.read_csv(tile_csv)

    print("Loading hotspot data...")
    hotspots_df = pd.read_csv(raw_csv)
    hotspots_df['Tanggal'] = pd.to_datetime(hotspots_df['Tanggal'], format='%d-%m-%Y')

    start_date = pd.Timestamp(start_year, 1, 1)
    end_date = pd.Timestamp(end_year, 12, 31)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    tile_ids = tiles_df['id'].unique()
    print(f"Processing {len(date_range)} days for {len(tile_ids)} tiles...")

    result_data = []
    for i, current_date in enumerate(date_range):
        if i % 365 == 0:
            print(f"  Processing year: {current_date.year}")

        daily_hotspots = hotspots_df[hotspots_df['Tanggal'] == current_date]
        row = {'date': current_date.strftime('%Y-%m-%d')}

        for tile_id in tile_ids:
            tile_info = tiles_df[tiles_df['id'] == tile_id].iloc[0]
            has_hotspot = 0
            for _, hotspot in daily_hotspots.iterrows():
                if point_in_tile(hotspot['Latitude'], hotspot['Longitude'], tile_info):
                    has_hotspot = 1
                    break
            row[f'tile_{tile_id}'] = has_hotspot
        result_data.append(row)

    result_df = pd.DataFrame(result_data)
    result_df.to_csv(output_csv, index=False)
    print(f"✓ Saved daily mapping to: {output_csv} (shape: {result_df.shape})")
    return result_df


def create_monthly_hotspot_data(daily_csv, output_csv):
    """
    Aggregate daily hotspot mapping to monthly sums.

    Args:
        daily_csv: path to daily mapping CSV.
        output_csv: output path for monthly CSV.
    """
    print("Reading daily hotspot data...")
    daily_df = pd.read_csv(daily_csv)
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    daily_df['year_month'] = daily_df['date'].dt.strftime('%Y-%m')

    tile_columns = [col for col in daily_df.columns if col.startswith('tile_')]
    print(f"Found {len(tile_columns)} tile columns")

    monthly_df = daily_df.groupby('year_month')[tile_columns].sum().reset_index()
    monthly_df = monthly_df.sort_values('year_month')

    monthly_df.to_csv(output_csv, index=False)
    print(f"✓ Saved monthly data to: {output_csv} ({len(monthly_df)} months)")
    return monthly_df


def main():
    parser = argparse.ArgumentParser(description='Data Processing Pipeline')
    parser.add_argument('--raw-csv', type=str, default='data/raw/perlu_diolah_untukTA.csv',
                        help='Path to raw hotspot CSV')
    parser.add_argument('--tile-csv', type=str, default='data/processed/pontianak_tile_boundaries.csv',
                        help='Path to tile boundaries CSV')
    parser.add_argument('--daily-output', type=str, default='data/processed/daily_hotspot_mapping_2014_2024.csv',
                        help='Output path for daily mapping')
    parser.add_argument('--monthly-output', type=str, default='data/processed/monthly_hotspot_sum.csv',
                        help='Output path for monthly aggregation')
    parser.add_argument('--start-year', type=int, default=2014)
    parser.add_argument('--end-year', type=int, default=2024)

    args = parser.parse_args()

    # Step 1: Create daily mapping
    create_daily_hotspot_mapping(
        raw_csv=args.raw_csv,
        tile_csv=args.tile_csv,
        output_csv=args.daily_output,
        start_year=args.start_year,
        end_year=args.end_year
    )

    # Step 2: Aggregate to monthly
    create_monthly_hotspot_data(
        daily_csv=args.daily_output,
        output_csv=args.monthly_output
    )

    print("\n✓ Data processing complete!")


if __name__ == '__main__':
    main()
