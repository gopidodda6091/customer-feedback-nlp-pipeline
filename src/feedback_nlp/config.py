"""Paths and labels shared across the pipeline."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.getenv("FEEDBACK_ROOT", Path(__file__).resolve().parents[2]))
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SEEDS_DIR = DATA_DIR / "seeds"
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", ROOT / "output"))
CLEAN_PATH = OUTPUT_DIR / "feedback_clean.parquet"
ENRICHED_PATH = OUTPUT_DIR / "feedback_enriched"
REPORTS_DIR = OUTPUT_DIR / "reports"
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", (ROOT / "mlruns").as_uri())

SENTIMENTS = ("negative", "neutral", "positive")
TOPICS = ("billing", "bug", "feature_request", "performance", "support", "usability")
