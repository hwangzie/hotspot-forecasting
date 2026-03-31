"""
Hyperparameter Tuning Module
=============================
Run Bayesian optimization to find optimal LSTM architecture.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import keras_tuner as kt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from src.model import r_squared, prepare_tile_data


def build_model_for_tuning(hp):
    """
    Build LSTM model with tunable hyperparameters.

    Args:
        hp: Keras Tuner HyperParameters object.

    Returns:
        Compiled Keras model.
    """
    model = Sequential()
    model.add(Input(shape=(12, 6)))  # lookback_months=12, n_features=6

    n_lstm_layers = hp.Int('n_lstm_layers', min_value=1, max_value=3, default=1)

    for i in range(n_lstm_layers):
        lstm_units = hp.Int(f'lstm_units_{i}', min_value=32, max_value=256, step=16, default=64)
        return_sequences = (i < n_lstm_layers - 1)

        model.add(LSTM(
            units=lstm_units,
            return_sequences=return_sequences,
            kernel_regularizer=tf.keras.regularizers.l2(
                hp.Float('l2_reg', min_value=1e-5, max_value=1e-2, sampling='log', default=1e-4)
            )
        ))

        dropout_rate = hp.Float(f'dropout_{i}', min_value=0.1, max_value=0.5, step=0.1, default=0.2)
        model.add(Dropout(dropout_rate))

        if hp.Boolean(f'batch_norm_{i}', default=False):
            model.add(BatchNormalization())

    if hp.Boolean('add_dense_layer', default=False):
        dense_units = hp.Int('dense_units', min_value=16, max_value=128, step=16, default=32)
        model.add(Dense(dense_units, activation='relu'))
        model.add(Dropout(hp.Float('dense_dropout', min_value=0.1, max_value=0.3, step=0.1, default=0.2)))

    model.add(Dense(1, activation='linear'))

    learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log', default=1e-3)

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae', r_squared]
    )

    return model


def run_tuning(monthly_csv, output_json, max_trials=30,
               tuning_epochs=50, tuning_patience=10,
               batch_size=16, lookback_months=12, test_size=24):
    """
    Run Bayesian hyperparameter tuning on the most-active tile.

    Args:
        monthly_csv: path to monthly hotspot CSV.
        output_json: path to save best hyperparameters JSON.
        max_trials: number of HP combinations to explore.
        tuning_epochs / tuning_patience / batch_size: tuning config.
        lookback_months / test_size: data windowing.
    """
    # Load data
    df = pd.read_csv(monthly_csv)
    df['date'] = pd.to_datetime(df['year_month'] + '-01')
    df.set_index('date', inplace=True)
    df.drop('year_month', axis=1, inplace=True)

    tile_columns = [col for col in df.columns if col.startswith('tile_')]

    # Select most active tile
    tile_totals = df[tile_columns].sum(axis=0).sort_values(ascending=False)
    representative_tile = tile_totals.index[0]
    print(f"Running hyperparameter tuning on: {representative_tile}")
    print(f"Total activity: {tile_totals[representative_tile]} hotspots")
    print("=" * 60)

    # Prepare data
    tuning_data = prepare_tile_data(df, representative_tile, lookback_months, test_size)

    # Create tuner
    tuner = kt.BayesianOptimization(
        build_model_for_tuning,
        objective=kt.Objective('val_loss', direction='min'),
        max_trials=max_trials,
        num_initial_points=5,
        directory='lstm_tuning',
        project_name='hotspot_forecasting',
        overwrite=True
    )

    print("\nHyperparameter Search Space:")
    tuner.search_space_summary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=tuning_patience, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=tuning_patience // 2, min_lr=1e-7)
    ]

    # Run search
    print("\nStarting hyperparameter search...")
    tuner.search(
        tuning_data['X_train'],
        tuning_data['y_train'],
        epochs=tuning_epochs,
        batch_size=batch_size,
        validation_data=(tuning_data['X_test'], tuning_data['y_test']),
        callbacks=callbacks,
        verbose=1
    )

    print("\nHyperparameter tuning completed!")

    # Extract best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_hyperparameters = {
        'n_lstm_layers': best_hps.get('n_lstm_layers'),
        'lstm_units': [best_hps.get(f'lstm_units_{i}') for i in range(best_hps.get('n_lstm_layers'))],
        'dropout_rates': [best_hps.get(f'dropout_{i}') for i in range(best_hps.get('n_lstm_layers'))],
        'batch_norm': [best_hps.get(f'batch_norm_{i}') for i in range(best_hps.get('n_lstm_layers'))],
        'l2_reg': best_hps.get('l2_reg'),
        'add_dense_layer': best_hps.get('add_dense_layer'),
        'dense_units': best_hps.get('dense_units') if best_hps.get('add_dense_layer') else None,
        'dense_dropout': best_hps.get('dense_dropout') if best_hps.get('add_dense_layer') else None,
        'learning_rate': best_hps.get('learning_rate')
    }

    os.makedirs(os.path.dirname(output_json) or '.', exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(best_hyperparameters, f, indent=2)

    print(f"\n✓ Best hyperparameters saved to: {output_json}")
    print(json.dumps(best_hyperparameters, indent=2))

    return best_hyperparameters


def main():
    parser = argparse.ArgumentParser(description='Run hyperparameter tuning')
    parser.add_argument('--monthly-csv', type=str, default='data/processed/monthly_hotspot_sum.csv')
    parser.add_argument('--output-json', type=str, default='configs/best_hyperparameters.json')
    parser.add_argument('--max-trials', type=int, default=30)
    parser.add_argument('--tuning-epochs', type=int, default=50)
    parser.add_argument('--tuning-patience', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lookback', type=int, default=12)
    parser.add_argument('--test-size', type=int, default=24)

    args = parser.parse_args()

    run_tuning(
        monthly_csv=args.monthly_csv,
        output_json=args.output_json,
        max_trials=args.max_trials,
        tuning_epochs=args.tuning_epochs,
        tuning_patience=args.tuning_patience,
        batch_size=args.batch_size,
        lookback_months=args.lookback,
        test_size=args.test_size,
    )


if __name__ == '__main__':
    main()
