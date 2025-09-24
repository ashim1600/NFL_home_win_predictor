import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from joblib import load
from src.config import FEATURES_CSV, MODEL_FILE, METRICS_JSON, CONFUSION_PNG
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
def main():
    df = pd.read_csv(FEATURES_CSV, parse_dates=['date'])
    X = df[['pf_mean_diff', 'pa_mean_diff']]
    y = df['home_win']

    model = load(MODEL_FILE)
    y_pred = model.predict(X)

    metrics = {
        'accuracy': float(accuracy_score(y, y_pred)),
        'precision': float(precision_score(y, y_pred, zero_division=0)),
        'recall': float(recall_score(y, y_pred, zero_division=0)),
        'f1': float(f1_score(y, y_pred, zero_division=0)),
        'n_samples': int(len(y))
    }
    with open(METRICS_JSON, 'w') as f:
        json.dump(metrics, f, indent=2)

    # Plot confusion matrix
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y, y_pred, ax=ax)
    fig.tight_layout()
    fig.savefig(CONFUSION_PNG, dpi=150)
    print(f"Saved metrics to {METRICS_JSON} and confusion matrix to {CONFUSION_PNG}")

if __name__ == "__main__":
    main()
