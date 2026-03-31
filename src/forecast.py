"""
Forecast Pipeline
=================
Load trained LSTM models and generate monthly hotspot forecasts.
Adapted from the existing ETL forecast pipeline.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

from src.model import r_squared


class HotspotForecaster:
    """Generate hotspot forecasts using trained LSTM models."""

    def __init__(self, monthly_csv, models_dir='models', output_dir='data/forecasts'):
        self.monthly_csv = monthly_csv
        self.models_dir = Path(models_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.df = None
        self.models = {}
        self.tile_data = {}
        self.tile_columns = []

    def extract(self):
        """Load monthly hotspot data."""
        print("=" * 60)
        print("STEP 1: EXTRACT - Loading Data")
        print("=" * 60)

        self.df = pd.read_csv(self.monthly_csv)
        self.df['date'] = pd.to_datetime(self.df['year_month'] + '-01')
        self.df.set_index('date', inplace=True)
        self.df.drop('year_month', axis=1, inplace=True)
        self.tile_columns = [col for col in self.df.columns if col.startswith('tile_')]

        print(f"✓ Loaded data: {self.df.shape}")
        print(f"✓ Date range: {self.df.index.min().strftime('%Y-%m')} to {self.df.index.max().strftime('%Y-%m')}")
        print(f"✓ Tiles: {len(self.tile_columns)}")
        return self

    def transform(self, lookback_months=12):
        """Scale and prepare features for each tile."""
        print("\n" + "=" * 60)
        print("STEP 2: TRANSFORM - Preprocessing Data")
        print("=" * 60)

        self.lookback_months = lookback_months

        for tile_name in self.tile_columns:
            tile_values = self.df[tile_name].values.reshape(-1, 1)

            features = pd.DataFrame({
                'month': self.df.index.month,
                'year': self.df.index.year - self.df.index.year.min(),
                'time_trend': range(len(self.df)),
                'month_sin': np.sin(2 * np.pi * self.df.index.month / 12),
                'month_cos': np.cos(2 * np.pi * self.df.index.month / 12)
            })

            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(tile_values).flatten()

            feature_scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_features = feature_scaler.fit_transform(features)

            self.tile_data[tile_name] = {
                'scaled_data': scaled_data,
                'features': features,
                'scaler': scaler,
                'feature_scaler': feature_scaler,
                'raw_data': tile_values,
            }
            print(f"  ✓ {tile_name} processed")

        return self

    def load_models(self):
        """Load all trained .h5 models."""
        print("\n" + "=" * 60)
        print("STEP 3: LOAD - Loading Trained Models")
        print("=" * 60)

        loaded = 0
        for tile_name in self.tile_columns:
            tile_num = tile_name.replace('tile_', '')
            model_path = self.models_dir / f"best_model_tile_{tile_num}.h5"

            if model_path.exists():
                try:
                    self.models[tile_name] = load_model(
                        str(model_path),
                        custom_objects={'r_squared': r_squared}
                    )
                    loaded += 1
                    print(f"  ✓ {tile_name}")
                except Exception as e:
                    print(f"  ✗ {tile_name}: {e}")
            else:
                print(f"  ✗ {tile_name}: not found at {model_path}")

        print(f"\n✓ Loaded {loaded}/{len(self.tile_columns)} models")
        return self

    def _forecast_tile(self, tile_name, model, start_date, months_ahead=12):
        """Rolling forecast for one tile."""
        tile_info = self.tile_data[tile_name]
        scaler = tile_info['scaler']
        feature_scaler = tile_info['feature_scaler']
        scaled_data = tile_info['scaled_data']
        features = tile_info['features']
        scaled_features = feature_scaler.transform(features)

        current_seq = scaled_data[-12:].copy()
        current_feats = scaled_features[-12:].copy()

        future_dates = pd.date_range(start=start_date, periods=months_ahead, freq='MS')
        forecasts = []

        for i, date in enumerate(future_dates):
            fut = pd.DataFrame({
                'month': [date.month],
                'year': [date.year - self.df.index.year.min()],
                'time_trend': [len(self.df) + i],
                'month_sin': [np.sin(2 * np.pi * date.month / 12)],
                'month_cos': [np.cos(2 * np.pi * date.month / 12)]
            })
            scaled_fut = feature_scaler.transform(fut)

            inp = []
            for j in range(len(current_seq)):
                inp.append([
                    current_seq[j],
                    current_feats[j, 0],
                    current_feats[j, 1],
                    current_feats[j, 2],
                    current_feats[j, 3],
                    current_feats[j, 4]
                ])
            inp = np.array(inp).reshape(1, len(current_seq), 6)

            pred = model.predict(inp, verbose=0)[0, 0]
            forecasts.append(pred)

            current_seq = np.append(current_seq[1:], pred)
            current_feats = np.vstack([current_feats[1:], scaled_fut])

        result = scaler.inverse_transform(np.array(forecasts).reshape(-1, 1)).flatten()
        return np.maximum(0, result)

    def generate_forecasts(self, start_year=2025, months_ahead=12):
        """Generate forecasts for all tiles."""
        print("\n" + "=" * 60)
        print(f"STEP 4: FORECAST - Generating {start_year} Forecasts")
        print("=" * 60)

        start_date = f"{start_year}-01-01"
        forecast_dates = pd.date_range(start=start_date, periods=months_ahead, freq='MS')
        months = [d.strftime('%Y-%m') for d in forecast_dates]

        all_fc = {}
        for tile in self.models:
            fc = self._forecast_tile(tile, self.models[tile], start_date, months_ahead)
            all_fc[tile] = fc
            print(f"  ✓ {tile}: total={fc.sum():.2f}, avg={fc.mean():.2f}")

        df_fc = pd.DataFrame(all_fc, index=months)
        df_fc.index.name = 'year_month'
        df_fc['total'] = df_fc.sum(axis=1)

        print(f"\n✓ Forecasts generated for {len(all_fc)} tiles")
        print(f"✓ Total predicted hotspots: {df_fc['total'].sum():.2f}")
        return df_fc

    def save_results(self, forecast_df, year=2025):
        """Save forecast CSV and summary JSON."""
        print("\n" + "=" * 60)
        print("STEP 5: SAVE - Saving Results")
        print("=" * 60)

        out_csv = self.output_dir / f"monthly_hotspot_forecasts_{year}.csv"
        forecast_df.to_csv(out_csv)
        print(f"✓ Saved forecasts to: {out_csv}")

        summary = {
            'forecast_year': year,
            'total_predicted_hotspots': float(forecast_df['total'].sum()),
            'monthly_average': float(forecast_df['total'].mean()),
            'peak_month': forecast_df['total'].idxmax(),
            'lowest_month': forecast_df['total'].idxmin(),
            'tiles': {}
        }
        for col in forecast_df.columns:
            if col != 'total':
                summary['tiles'][col] = {
                    'total': float(forecast_df[col].sum()),
                    'average': float(forecast_df[col].mean()),
                    'max_month': forecast_df[col].idxmax(),
                    'max_value': float(forecast_df[col].max()),
                }

        summary_path = self.output_dir / f"forecast_summary_{year}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Saved summary to: {summary_path}")

        print("\n" + "=" * 60)
        print("FORECAST PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        return summary


def main():
    parser = argparse.ArgumentParser(description='Generate monthly hotspot forecasts')
    parser.add_argument('--monthly-csv', type=str, default='data/processed/monthly_hotspot_sum.csv')
    parser.add_argument('--models-dir', type=str, default='models')
    parser.add_argument('--output-dir', type=str, default='data/forecasts')
    parser.add_argument('--year', type=int, default=2025)
    parser.add_argument('--months', type=int, default=12)

    args = parser.parse_args()

    fc = HotspotForecaster(
        monthly_csv=args.monthly_csv,
        models_dir=args.models_dir,
        output_dir=args.output_dir
    )

    fc.extract()
    fc.transform()
    fc.load_models()
    forecast_df = fc.generate_forecasts(start_year=args.year, months_ahead=args.months)
    fc.save_results(forecast_df, year=args.year)


if __name__ == '__main__':
    main()
