"""
Training Module
===============
Train per-tile LSTM models using optimized hyperparameters.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from src.model import build_lstm_model, load_hyperparameters, prepare_tile_data, r_squared


def load_monthly_data(csv_path):
    """Load monthly hotspot CSV and return DataFrame with datetime index."""
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['year_month'] + '-01')
    df.set_index('date', inplace=True)
    df.drop('year_month', axis=1, inplace=True)
    return df


def train_all_tiles(monthly_csv, hyperparams_path, models_dir,
                    epochs=150, batch_size=16, patience=20,
                    lookback_months=12, test_size=24):
    """
    Train one LSTM model per tile using shared hyperparameters.

    Args:
        monthly_csv: path to monthly hotspot CSV.
        hyperparams_path: path to best_hyperparameters.json.
        models_dir: directory to save trained .h5 models.
        epochs / batch_size / patience: training config.
        lookback_months: LSTM input window size.
        test_size: months held out for testing.
    """
    df = load_monthly_data(monthly_csv)
    tile_columns = [col for col in df.columns if col.startswith('tile_')]
    hyperparams = load_hyperparameters(hyperparams_path)

    os.makedirs(models_dir, exist_ok=True)

    results = {}
    print(f"Training models for {len(tile_columns)} tiles...")
    print(f"Hyperparameters: {json.dumps(hyperparams, indent=2)}")
    print("=" * 60)

    for idx, tile_name in enumerate(tile_columns):
        print(f"\n{'=' * 60}")
        print(f"Training model for {tile_name} ({idx + 1}/{len(tile_columns)})")
        print(f"{'=' * 60}")

        tile_total = df[tile_name].sum()
        print(f"Total activity in {tile_name}: {tile_total} hotspots")

        # Prepare data
        data = prepare_tile_data(df, tile_name, lookback_months, test_size)

        # Build model
        model = build_lstm_model(hyperparams, lookback_months, n_features=6)
        print(f"\nModel architecture for {tile_name}:")
        model.summary()

        # Callbacks
        tile_num = tile_name.replace('tile_', '')
        model_path = os.path.join(models_dir, f"best_model_tile_{tile_num}.h5")

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=patience // 2, min_lr=1e-7),
            ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, verbose=1)
        ]

        # Train
        print(f"\nTraining {tile_name} model...")
        print(f"Training data shape: {data['X_train'].shape}")
        print(f"Test data shape: {data['X_test'].shape}")

        history = model.fit(
            data['X_train'], data['y_train'],
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(data['X_test'], data['y_test']),
            callbacks=callbacks,
            verbose=1
        )

        # Evaluate
        test_loss, test_mae, test_r2 = model.evaluate(
            data['X_test'], data['y_test'], verbose=0
        )

        results[tile_name] = {
            'mse': float(test_loss),
            'mae': float(test_mae),
            'r2': float(test_r2),
            'tile_total_hotspots': int(tile_total),
            'model_path': model_path,
            'epochs_trained': len(history.history['loss']),
        }

        print(f"\n[OK] {tile_name} - MSE: {test_loss:.6f}, MAE: {test_mae:.6f}, R2: {test_r2:.6f}")
        print(f"[OK] Model saved to: {model_path}")

    # Save results summary
    results_path = os.path.join(models_dir, "training_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Training results saved to: {results_path}")
    print(f"\n{'=' * 60}")
    print("ALL TRAINING COMPLETE!")
    print(f"{'=' * 60}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Train LSTM models for all tiles')
    parser.add_argument('--monthly-csv', type=str, default='data/processed/monthly_hotspot_sum.csv',
                        help='Path to monthly hotspot CSV')
    parser.add_argument('--hyperparams', type=str, default='configs/best_hyperparameters.json',
                        help='Path to hyperparameters JSON')
    parser.add_argument('--models-dir', type=str, default='models',
                        help='Directory to save trained models')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--lookback', type=int, default=12)
    parser.add_argument('--test-size', type=int, default=24)

    args = parser.parse_args()

    train_all_tiles(
        monthly_csv=args.monthly_csv,
        hyperparams_path=args.hyperparams,
        models_dir=args.models_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        lookback_months=args.lookback,
        test_size=args.test_size
    )


if __name__ == '__main__':
    main()
