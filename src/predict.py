import pandas as pd
import sys, pathlib
from joblib import load
from src.config import FEATURES_CSV, MODEL_FILE, PREDICTIONS_CSV

def main():
    df = pd.read_csv(FEATURES_CSV, parse_dates=['date'])
    X = df[['pf_mean_diff', 'pa_mean_diff']]
    model = load(MODEL_FILE)
    proba = model.predict_proba(X)[:,1]
    preds = (proba >= 0.5).astype(int)

    out = df[['game_id','date','week','home_team','away_team','home_score','away_score']].copy()
    out['pred_home_win_prob'] = proba
    out['pred_home_win'] = preds
    out.to_csv(PREDICTIONS_CSV, index=False)
    print(f"Saved predictions to {PREDICTIONS_CSV}")

if __name__ == "__main__":
    main()
