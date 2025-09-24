from pathlib import Path
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROC_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIG_DIR = REPORTS_DIR / "figures"

RAW_GAMES = RAW_DIR / "sample_games.csv"
RAW_PLAYS = RAW_DIR / "sample_plays.csv"
FEATURES_CSV = PROC_DIR / "features.csv"
MODEL_FILE = MODELS_DIR / "model.joblib"
METRICS_JSON = REPORTS_DIR / "metrics.json"
PREDICTIONS_CSV = REPORTS_DIR / "predictions.csv"
CONFUSION_PNG = FIG_DIR / "confusion_matrix.png"

for d in [PROC_DIR, MODELS_DIR, REPORTS_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)
