"""
LSTM Model Builder for Hotspot Forecasting
==========================================
Build LSTM models from hyperparameters configuration.
"""

import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler


def r_squared(y_true, y_pred):
    """Custom R² metric for Keras."""
    from tensorflow.keras import backend as K
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())


def load_hyperparameters(path):
    """Load hyperparameters from a JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def build_lstm_model(hyperparams, lookback_months=12, n_features=6):
    """
    Build an LSTM model from a hyperparameters dict.

    Args:
        hyperparams: dict with keys like n_lstm_layers, lstm_units, dropout_rates, etc.
        lookback_months: number of months in the input window.
        n_features: number of features per timestep.

    Returns:
        A compiled Keras Sequential model.
    """
    model = Sequential()
    model.add(Input(shape=(lookback_months, n_features)))

    n_layers = hyperparams['n_lstm_layers']

    for i in range(n_layers):
        return_sequences = (i < n_layers - 1)
        model.add(LSTM(
            units=hyperparams['lstm_units'][i],
            return_sequences=return_sequences,
            kernel_regularizer=l2(hyperparams['l2_reg'])
        ))
        model.add(Dropout(hyperparams['dropout_rates'][i]))

        if hyperparams['batch_norm'][i]:
            model.add(BatchNormalization())

    if hyperparams.get('add_dense_layer'):
        model.add(Dense(hyperparams['dense_units'], activation='relu'))
        model.add(Dropout(hyperparams['dense_dropout']))

    model.add(Dense(1, activation='linear'))

    model.compile(
        optimizer=Adam(learning_rate=hyperparams['learning_rate']),
        loss='mse',
        metrics=['mae', r_squared]
    )

    return model


def prepare_tile_data(df, tile_name, lookback_months=12, test_size=24):
    """
    Prepare data for a single tile: scale, add temporal features, create sequences.

    Args:
        df: DataFrame with date index and tile columns.
        tile_name: column name of the tile.
        lookback_months: how many months to look back.
        test_size: number of months for test set.

    Returns:
        dict with X_train, y_train, X_test, y_test, scaler, features, feature_scaler.
    """
    tile_values = df[tile_name].values.reshape(-1, 1)

    # Create temporal features
    features = np.column_stack([
        df.index.month,
        df.index.year - df.index.year.min(),
        np.arange(len(df)),
        np.sin(2 * np.pi * df.index.month / 12),
        np.cos(2 * np.pi * df.index.month / 12)
    ])

    # Scale hotspot data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(tile_values).flatten()

    # Scale temporal features
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = feature_scaler.fit_transform(features)

    # Create sequences
    X, y = [], []
    for i in range(lookback_months, len(scaled_data)):
        seq = []
        for j in range(i - lookback_months, i):
            seq.append([
                scaled_data[j],
                scaled_features[j, 0],
                scaled_features[j, 1],
                scaled_features[j, 2],
                scaled_features[j, 3],
                scaled_features[j, 4]
            ])
        X.append(seq)
        y.append(scaled_data[i])

    X = np.array(X)
    y = np.array(y)

    # Split
    X_train = X[:-test_size]
    y_train = y[:-test_size]
    X_test = X[-test_size:]
    y_test = y[-test_size:]

    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'scaler': scaler,
        'feature_scaler': feature_scaler,
        'features': features,
        'scaled_data': scaled_data,
        'scaled_features': scaled_features,
    }
