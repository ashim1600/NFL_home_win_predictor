#!/usr/bin/env bash
set -euo pipefail

# Optionally run the full pipeline before launching the app
if [ "${RUN_PIPELINE_ON_START:-true}" = "true" ]; then
  python -m src.ingest
  python -m src.features
  python -m src.train
  python -m src.evaluate
  python -m src.predict
fi

# Start Streamlit
exec streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0