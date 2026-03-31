# 🔥 LSTM Hotspot Forecasting — CI/CD Pipeline

Automated monthly hotspot forecasting using per-tile LSTM models with CI/CD retraining on GitHub Actions.

---

## 📁 Repository Structure

```
hotspot-forecasting/
├── .github/workflows/
│   ├── test.yml          # Run tests on push / PR
│   ├── retrain.yml       # Monthly retrain + tune + forecast
│   └── forecast.yml      # Manual-dispatch forecast only
├── src/
│   ├── __init__.py
│   ├── model.py          # LSTM builder & data preparation
│   ├── data_processing.py # Raw → daily → monthly pipeline
│   ├── train.py          # Train per-tile models
│   ├── tune.py           # Bayesian hyperparameter tuning
│   └── forecast.py       # Generate rolling forecasts
├── configs/
│   └── best_hyperparameters.json
├── data/
│   ├── raw/              # Raw hotspot CSVs (manual upload)
│   ├── processed/        # Monthly aggregated data
│   └── forecasts/        # Forecast outputs
├── models/               # Trained .h5 models (Git LFS)
├── tests/
│   └── test_model.py
├── notebooks/            # Original Jupyter notebooks
├── requirements.txt
├── .gitattributes        # Git LFS rules
└── .gitignore
```

## 🚀 Quick Start

### 1. Clone & Install

```bash
git lfs install
git clone <your-repo-url>
cd hotspot-forecasting
pip install -r requirements.txt
```

### 2. Add Your Data

Place your raw hotspot CSV (`perlu_diolah_untukTA.csv`) into `data/raw/` and
the monthly aggregated CSV (`monthly_hotspot_sum.csv`) into `data/processed/`.

### 3. Run Locally

```bash
# Data processing (raw → daily → monthly)
python -m src.data_processing \
  --raw-csv data/raw/perlu_diolah_untukTA.csv \
  --tile-csv data/processed/pontianak_tile_boundaries.csv

# Hyperparameter tuning
python -m src.tune \
  --monthly-csv data/processed/monthly_hotspot_sum.csv \
  --output-json configs/best_hyperparameters.json

# Train all tile models
python -m src.train \
  --monthly-csv data/processed/monthly_hotspot_sum.csv \
  --hyperparams configs/best_hyperparameters.json \
  --models-dir models

# Generate forecasts
python -m src.forecast \
  --monthly-csv data/processed/monthly_hotspot_sum.csv \
  --models-dir models \
  --year 2025
```

### 4. Run Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

---

## ⚙️ CI/CD Workflows

| Workflow | Trigger | What It Does |
|---|---|---|
| **test.yml** | Push / PR to `main` | Runs unit tests |
| **retrain.yml** | Monthly cron • data push • manual | Tune → Train → Forecast → Commit |
| **forecast.yml** | Manual dispatch only | Forecast using existing models |

### Triggering a Manual Retrain

1. Go to **Actions** → **Monthly Retrain** → **Run workflow**
2. Optionally override `max_trials` and `epochs`
3. Results are auto-committed back to the repo

### Adding New Data

1. Place new CSV in `data/raw/` or update `data/processed/monthly_hotspot_sum.csv`
2. Push to `main`
3. The **retrain** workflow triggers automatically

---

## 🗂️ Git LFS

Large files (`.h5` models, `.csv` datasets) are tracked with [Git LFS](https://git-lfs.com/).

```bash
git lfs install   # one-time setup
git lfs track "*.h5"
git lfs track "*.csv"
```

The `.gitattributes` file already includes these rules.

---

## 📊 Model Architecture

The LSTM architecture is determined by Bayesian hyperparameter tuning (30 trials) on the most-active tile:

- **Tuned parameters**: # LSTM layers, units per layer, dropout, L2 regularization, optional dense layer, learning rate
- **Default best**: 1 LSTM layer → 144 units → 0.2 dropout → Dense(1)
- **Training**: EarlyStopping + ReduceLROnPlateau + ModelCheckpoint
- **One model per tile**: each tile gets its own trained `.h5` file

---

## 📝 License

This project is part of a thesis (skripsi). All rights reserved.
