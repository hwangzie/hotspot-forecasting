"""
Unit tests for model.py
"""

import json
import tempfile
import numpy as np
import pandas as pd
import pytest

from src.model import (
    build_lstm_model,
    load_hyperparameters,
    prepare_tile_data,
    r_squared,
)


@pytest.fixture
def sample_hyperparams():
    """Minimal hyperparameters dict."""
    return {
        "n_lstm_layers": 1,
        "lstm_units": [32],
        "dropout_rates": [0.2],
        "batch_norm": [False],
        "l2_reg": 0.001,
        "add_dense_layer": False,
        "dense_units": None,
        "dense_dropout": None,
        "learning_rate": 0.01
    }


@pytest.fixture
def sample_monthly_df():
    """Create a tiny monthly DataFrame for testing."""
    dates = pd.date_range('2020-01-01', periods=48, freq='MS')
    rng = np.random.RandomState(42)
    data = {
        'tile_1': rng.randint(0, 10, size=48),
        'tile_2': rng.randint(0, 5, size=48),
    }
    df = pd.DataFrame(data, index=dates)
    return df


class TestBuildModel:
    def test_build_model_shape(self, sample_hyperparams):
        """Model should accept (batch, 12, 6) input and output (batch, 1)."""
        model = build_lstm_model(sample_hyperparams, lookback_months=12, n_features=6)
        assert model.input_shape == (None, 12, 6)
        assert model.output_shape == (None, 1)

    def test_build_model_with_dense_layer(self, sample_hyperparams):
        """Model with optional dense layer should also compile."""
        hp = sample_hyperparams.copy()
        hp['add_dense_layer'] = True
        hp['dense_units'] = 16
        hp['dense_dropout'] = 0.1
        model = build_lstm_model(hp)
        assert model.output_shape == (None, 1)

    def test_build_model_multi_lstm(self, sample_hyperparams):
        """Multi-layer LSTM should compile."""
        hp = sample_hyperparams.copy()
        hp['n_lstm_layers'] = 2
        hp['lstm_units'] = [32, 16]
        hp['dropout_rates'] = [0.2, 0.2]
        hp['batch_norm'] = [False, False]
        model = build_lstm_model(hp)
        assert model.output_shape == (None, 1)


class TestLoadHyperparameters:
    def test_load_round_trip(self, sample_hyperparams):
        """Save then load hyperparameters."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_hyperparams, f)
            f.flush()
            loaded = load_hyperparameters(f.name)
        assert loaded == sample_hyperparams


class TestPrepareTileData:
    def test_output_keys(self, sample_monthly_df):
        """prepare_tile_data should return expected keys."""
        result = prepare_tile_data(sample_monthly_df, 'tile_1', lookback_months=6, test_size=6)
        expected_keys = {
            'X_train', 'y_train', 'X_test', 'y_test',
            'scaler', 'feature_scaler', 'features',
            'scaled_data', 'scaled_features',
        }
        assert set(result.keys()) == expected_keys

    def test_output_shapes(self, sample_monthly_df):
        """X and y dimensions should be consistent."""
        result = prepare_tile_data(sample_monthly_df, 'tile_1', lookback_months=6, test_size=6)
        assert result['X_test'].shape[0] == 6
        assert result['X_test'].shape[1] == 6  # lookback
        assert result['X_test'].shape[2] == 6  # features
        assert result['y_test'].shape[0] == 6

    def test_scaling_bounds(self, sample_monthly_df):
        """Scaled data should be in [0, 1]."""
        result = prepare_tile_data(sample_monthly_df, 'tile_2', lookback_months=6, test_size=6)
        assert result['scaled_data'].min() >= 0.0
        assert result['scaled_data'].max() <= 1.0


class TestRSquared:
    def test_perfect_prediction(self):
        """R² should be ~1 for perfect prediction."""
        import tensorflow as tf
        y = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0])
        val = r_squared(y, y)
        assert float(val.numpy()) == pytest.approx(1.0, abs=1e-5)
