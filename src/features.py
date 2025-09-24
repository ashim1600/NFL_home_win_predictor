import pandas as pd
import sys, pathlib
from src.config import RAW_GAMES, FEATURES_CSV

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    # Sort by date so expanding stats make sense
    df = df.sort_values('date').reset_index(drop=True)

    # Label: home team win?
    df['home_win'] = (df['home_score'] > df['away_score']).astype(int)

    # Build per-team rolling (expanding) points for past games
    def team_stats(team_col, score_col, opp_score_col, prefix):
        rows = []
        teams = pd.unique(df[team_col])
        for t in teams:
            mask_home = df['home_team'] == t
            mask_away = df['away_team'] == t
            team_games = df[mask_home | mask_away].copy()
            team_games['points_for'] = team_games.apply(
                lambda r: r[score_col] if r[team_col] == t else r[opp_score_col], axis=1
            )
            team_games['points_against'] = team_games.apply(
                lambda r: r[opp_score_col] if r[team_col] == t else r[score_col], axis=1
            )
            # expanding means (shifted so current game doesn't leak info)
            team_games[f'{prefix}_pf_exp_mean'] = team_games['points_for'].expanding().mean().shift(1)
            team_games[f'{prefix}_pa_exp_mean'] = team_games['points_against'].expanding().mean().shift(1)
            team_games['game_id'] = team_games['game_id']
            rows.append(team_games[['game_id', f'{prefix}_pf_exp_mean', f'{prefix}_pa_exp_mean']])
        stats = pd.concat(rows).drop_duplicates(subset=['game_id'], keep='last')
        return stats

    # For home and away teams separately
    home_stats = team_stats('home_team', 'home_score', 'away_score', 'home_team')
    away_stats = team_stats('away_team', 'away_score', 'home_score', 'away_team')

    out = df.merge(home_stats, on='game_id', how='left').merge(away_stats, on='game_id', how='left')

    # Simple feature engineering: differences
    out['pf_mean_diff'] = out['home_team_pf_exp_mean'] - out['away_team_pf_exp_mean']
    out['pa_mean_diff'] = out['home_team_pa_exp_mean'] - out['away_team_pa_exp_mean']

    # Drop rows with all-na features (first appearances)
    out_feat = out_feat = out.fillna(0).copy()

    return out_feat

def main():
    games = pd.read_csv(RAW_GAMES, parse_dates=['date'])
    feats = build_features(games)
    feats.to_csv(FEATURES_CSV, index=False)
    print(f"Saved features to {FEATURES_CSV} with shape {feats.shape}")

if __name__ == "__main__":
    main()
