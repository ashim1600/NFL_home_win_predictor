import json
import sys, pathlib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from joblib import dump
from src.config import FEATURES_CSV, MODEL_FILE, REPORTS_DIR

def main():
    df = pd.read_csv(FEATURES_CSV, parse_dates=['date'])
    X = df[['pf_mean_diff', 'pa_mean_diff']]
    y = df['home_win']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y if y.nunique() > 1 else None)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    dump(model, MODEL_FILE)
    print(f"Saved model to {MODEL_FILE}")

if __name__ == "__main__":
    main()
