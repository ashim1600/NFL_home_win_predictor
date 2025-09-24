import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.config import PREDICTIONS_CSV, METRICS_JSON, CONFUSION_PNG
import json

st.set_page_config(page_title="NFL Home Win Predictor", layout="wide")

st.title("🏈 NFL Home Win Predictor Dashboard")

# --- Team logos mapping ---
TEAM_LOGOS = {
    "KC": "https://a.espncdn.com/i/teamlogos/nfl/500/scoreboard/kc.png",
    "DET": "https://a.espncdn.com/i/teamlogos/nfl/500/scoreboard/det.png",
    "PHI": "https://a.espncdn.com/i/teamlogos/nfl/500/scoreboard/phi.png",
    "NE": "https://a.espncdn.com/i/teamlogos/nfl/500/scoreboard/ne.png",
    "SEA": "https://a.espncdn.com/i/teamlogos/nfl/500/scoreboard/sea.png",
    "JAX": "https://a.espncdn.com/i/teamlogos/nfl/500/scoreboard/jax.png",
    "CHI": "https://a.espncdn.com/i/teamlogos/nfl/500/scoreboard/chi.png",
    "ATL": "https://a.espncdn.com/i/teamlogos/nfl/500/scoreboard/atl.png",
    "NYJ": "https://a.espncdn.com/i/teamlogos/nfl/500/scoreboard/nyj.png",
    "GB": "https://a.espncdn.com/i/teamlogos/nfl/500/scoreboard/gb.png",
    "MIN": "https://a.espncdn.com/i/teamlogos/nfl/500/scoreboard/min.png",
    "CAR": "https://a.espncdn.com/i/teamlogos/nfl/500/scoreboard/car.png",
}

# --- Load data ---
@st.cache_data
def load_predictions():
    df = pd.read_csv(PREDICTIONS_CSV, parse_dates=["date"])
    # Add logo URLs
    df["home_logo"] = df["home_team"].map(TEAM_LOGOS)
    df["away_logo"] = df["away_team"].map(TEAM_LOGOS)
    return df

@st.cache_data
def load_metrics():
    with open(METRICS_JSON) as f:
        return json.load(f)

preds = load_predictions()
metrics = load_metrics()

# --- Show metrics ---
st.subheader("📊 Model Performance")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{metrics['accuracy']:.2f}")
col2.metric("Precision", f"{metrics['precision']:.2f}")
col3.metric("Recall", f"{metrics['recall']:.2f}")
col4.metric("F1 Score", f"{metrics['f1']:.2f}")

st.image(str(CONFUSION_PNG), caption="Confusion Matrix")

# --- Sidebar Filters ---
st.sidebar.header("🔍 Filters")
team_filter = st.sidebar.selectbox(
    "Team", ["All"] + sorted(set(preds["home_team"]) | set(preds["away_team"]))
)
week_filter = st.sidebar.multiselect(
    "Weeks", sorted(preds["week"].unique()), default=sorted(preds["week"].unique())
)

filtered = preds.copy()
if team_filter != "All":
    filtered = filtered[
        (filtered["home_team"] == team_filter) | (filtered["away_team"] == team_filter)
    ]
if week_filter:
    filtered = filtered[filtered["week"].isin(week_filter)]

# --- Predictions Table with Logos ---
st.subheader("🔮 Predictions")

display_df = filtered.copy()
display_df = display_df[
    [
        "date", "week", "home_team", "home_logo", "away_team", "away_logo",
        "home_score", "away_score", "pred_home_win_prob", "pred_home_win"
    ]
]

display_df = display_df.rename(columns={
    "date": "Date",
    "week": "Week",
    "home_team": "Home Team",
    "home_logo": "Home Logo",
    "away_team": "Away Team",
    "away_logo": "Away Logo",
    "home_score": "Home Score",
    "away_score": "Away Score",
    "pred_home_win_prob": "Pred Home Win Prob",
    "pred_home_win": "Pred Home Win"
})

st.data_editor(
    display_df,
    column_config={
        "Home Logo": st.column_config.ImageColumn("🏠 Logo"),
        "Away Logo": st.column_config.ImageColumn("✈️ Logo"),
        "Pred Home Win Prob": st.column_config.ProgressColumn(
            "Home Win Probability",
            format="%.2f",
            min_value=0,
            max_value=1,
        ),
    },
    hide_index=True,
    use_container_width=True,
)

# --- Visualization ---
st.subheader("📈 Predicted Home Win Probabilities")
fig, ax = plt.subplots()
ax.plot(filtered["date"], filtered["pred_home_win_prob"], marker="o")
ax.set_ylabel("Probability")
ax.set_xlabel("Date")
ax.set_title("Home Win Probability Over Time")
st.pyplot(fig)

# --- Summary stats ---
st.subheader("📑 Summary")
st.write(f"Total games: {len(filtered)}")
st.write(f"Predicted Home Wins: {filtered['pred_home_win'].sum()}")