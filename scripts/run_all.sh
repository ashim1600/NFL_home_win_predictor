#!/usr/bin/env bash
set -euo pipefail

# Install (if not already)
python -m pip install -r requirements.txt

# Pipeline
python -m src.ingest
python -m src.features
python -m src.train
python -m src.evaluate
python -m src.predict

echo "✅ Pipeline complete. See reports/ and models/."