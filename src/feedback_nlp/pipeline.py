"""End-to-end batch job: clean -> enrich (Spark or pandas) -> evaluate -> MLflow."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import pandas as pd

from feedback_nlp import config
from feedback_nlp.clean import clean_frame, load_raw
from feedback_nlp.embedder import get_embedder
from feedback_nlp.enrich import build_spark, enrich_pandas, enrich_spark, write_enriched
from feedback_nlp.evaluate import evaluate_task, join_holdout, log_to_mlflow


def run_clean(raw_dir: Path, out_path: Path) -> dict[str, int]:
    clean, stats = clean_frame(load_raw(raw_dir))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(out_path, index=False)
    print(f"clean: {stats.input_rows} -> {stats.output_rows} rows, dropped {stats.dropped}")
    return {"input_rows": stats.input_rows, "output_rows": stats.output_rows, **stats.dropped}


def run_enrich(engine: str, clean_path: Path, out_path: Path) -> None:
    if out_path.exists():
        shutil.rmtree(out_path)
    if engine == "spark":
        spark = build_spark()
        try:
            write_enriched(enrich_spark(spark.read.parquet(str(clean_path))), out_path)
        finally:
            spark.stop()
    else:
        enriched = enrich_pandas(pd.read_parquet(clean_path))
        enriched.to_parquet(out_path, partition_cols=["source"], index=False)
    print(f"enrich[{engine}]: wrote {out_path}")


def run_evaluate(
    enriched_path: Path, holdout_path: Path, reports_dir: Path, params: dict[str, str]
) -> dict[str, float]:
    enriched = pd.read_parquet(enriched_path)
    enriched["source"] = enriched["source"].astype(str)
    joined = join_holdout(enriched, pd.read_csv(holdout_path, dtype=str))
    metrics = {"holdout_rows_evaluated": float(len(joined))}
    metrics |= evaluate_task(joined, "sentiment", config.SENTIMENTS, reports_dir)
    metrics |= evaluate_task(joined, "topic", config.TOPICS, reports_dir)
    (reports_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    run_id = log_to_mlflow(metrics, params, reports_dir, config.MLFLOW_URI)
    print(f"evaluate: {len(joined)} hold-out rows, mlflow run {run_id}")
    for key in ("sentiment_macro_f1", "topic_macro_f1", "sentiment_accuracy", "topic_accuracy"):
        print(f"  {key}: {metrics[key]:.3f}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engine", choices=["spark", "pandas"], default=os.getenv("ENRICH_ENGINE", "spark")
    )
    args = parser.parse_args()
    clean_stats = run_clean(config.RAW_DIR, config.CLEAN_PATH)
    run_enrich(args.engine, config.CLEAN_PATH, config.ENRICHED_PATH)
    params = {
        "engine": args.engine,
        "embedder": get_embedder().name,
        "topic_method": "nearest_centroid_seeds",
        "sentiment_method": "lexicon",
        **{f"clean_{k}": str(v) for k, v in clean_stats.items()},
    }
    run_evaluate(
        config.ENRICHED_PATH,
        config.DATA_DIR / "labelled" / "holdout.csv",
        config.REPORTS_DIR,
        params,
    )


if __name__ == "__main__":
    main()
