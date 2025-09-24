# NFL Use Case: Home Win Prediction (Starter Kit)

This project is a **from-scratch, direct-run** template for an NFL analytics use case.
It ships with tiny sample data so you can run end-to-end immediately (no internet required).

## What it does
- Builds expanding historical features from `data/raw/sample_games.csv`
- Trains a simple **Logistic Regression** model to predict whether the **home team wins**
- Evaluates and saves metrics + a confusion matrix
- Produces a quick plot and creates a CSV of predictions

## Quick start
```bash
# 1) (Optional) Create and activate a virtualenv
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install deps
pip install -r requirements.txt

# 3) Run the full pipeline
bash scripts/run_all.sh
# or
make run
```

Outputs:
- `models/model.joblib` — trained model
- `data/processed/features.csv` — features used for training
- `reports/metrics.json` — accuracy, precision, recall, f1
- `reports/figures/confusion_matrix.png` — confusion matrix
- `reports/predictions.csv` — per-game predictions

## Project structure
```
nfl-usecase/
  data/
    raw/
      sample_games.csv
      sample_plays.csv
    processed/
  models/
  reports/
    figures/
  scripts/
    run_all.sh
  src/
    config.py
    ingest.py
    features.py
    train.py
    evaluate.py
    predict.py
    utils.py
  tests/
    test_features.py
  requirements.txt
  README.md
  Makefile
```

## Extend it
- Replace the sample data with real NFL data in `data/raw`
- Add new features in `src/features.py` (e.g., rolling EPA, pass/run splits)
- Swap model in `src/train.py` (RandomForest, XGBoost, etc.)
