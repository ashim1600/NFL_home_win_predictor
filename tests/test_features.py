import pandas as pd
from src.features import build_features

def test_build_features_minimal():
    csv = """game_id,date,week,home_team,away_team,home_score,away_score
g1,2023-09-10,1,KC,DET,20,21
g2,2023-09-17,2,KC,JAX,17,9
"""
    df = pd.read_csv(pd.compat.StringIO(csv), parse_dates=['date'])
    feats = build_features(df)
    assert 'pf_mean_diff' in feats.columns
    assert 'pa_mean_diff' in feats.columns
