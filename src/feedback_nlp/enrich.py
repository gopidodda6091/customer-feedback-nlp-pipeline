"""Batch enrichment. The same pandas functions run directly (pandas engine) or as
PySpark pandas UDFs (spark engine), so both engines produce identical rows."""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from feedback_nlp import entities, sentiment
from feedback_nlp.config import SEEDS_DIR
from feedback_nlp.embedder import Embedder, get_embedder
from feedback_nlp.topics import CentroidClassifier, fit_from_seeds

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession

SEED_PATH = SEEDS_DIR / "topic_seeds.csv"
SENTIMENT_DDL = "struct<label:string,score:double>"
TOPIC_DDL = "struct<label:string,score:double>"
ENTITY_DDL = "struct<products:array<string>,features:array<string>,plans:array<string>>"
OUTPUT_COLUMNS = [
    "record_id",
    "source",
    "created_at",
    "rating",
    "text",
    "text_hash",
    "sentiment_label",
    "sentiment_score",
    "topic_label",
    "topic_score",
    "products",
    "features",
    "plans",
    "embedding",
    "embedder",
]


@lru_cache(maxsize=4)
def _classifier(seed_path: str) -> CentroidClassifier:
    return fit_from_seeds(Path(seed_path), get_embedder())


# ---- pure pandas building blocks (also the bodies of the Spark UDFs) -----------------------


def sentiment_frame(texts: pd.Series) -> pd.DataFrame:
    results = [sentiment.classify(t) for t in texts]
    return pd.DataFrame(
        {"label": [r.label for r in results], "score": [float(r.score) for r in results]}
    )


def embedding_series(texts: pd.Series, embedder: Embedder) -> pd.Series:
    vectors = embedder.embed(texts.fillna("").tolist())
    return pd.Series([row.tolist() for row in vectors], index=texts.index)


def topic_frame(embeddings: pd.Series, clf: CentroidClassifier) -> pd.DataFrame:
    if len(embeddings) == 0:
        return pd.DataFrame({"label": pd.Series(dtype=str), "score": pd.Series(dtype=float)})
    labels, scores = clf.predict_vectors(np.vstack(embeddings.to_numpy()).astype(np.float32))
    return pd.DataFrame({"label": labels, "score": [round(float(s), 4) for s in scores]})


def entity_frame(texts: pd.Series) -> pd.DataFrame:
    found = [entities.extract(t) for t in texts]
    return pd.DataFrame(
        {
            "products": [e.products for e in found],
            "features": [e.features for e in found],
            "plans": [e.plans for e in found],
        }
    )


def enrich_pandas(clean: pd.DataFrame, seed_path: Path = SEED_PATH) -> pd.DataFrame:
    embedder = get_embedder()
    df = clean.reset_index(drop=True).copy()
    df["embedding"] = embedding_series(df["text"], embedder)
    sent = sentiment_frame(df["text"])
    topic = topic_frame(df["embedding"], _classifier(str(seed_path)))
    ents = entity_frame(df["text"])
    df["sentiment_label"], df["sentiment_score"] = sent["label"], sent["score"]
    df["topic_label"], df["topic_score"] = topic["label"], topic["score"]
    df["products"], df["features"], df["plans"] = ents["products"], ents["features"], ents["plans"]
    df["embedder"] = embedder.name
    return df[OUTPUT_COLUMNS]


# ---- Spark engine ---------------------------------------------------------------------------


def build_spark(app_name: str = "feedback-nlp") -> SparkSession:
    from pyspark.sql import SparkSession

    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    return (
        SparkSession.builder.master(os.getenv("SPARK_MASTER", "local[2]"))
        .appName(app_name)
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", "512")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )


def enrich_spark(df: DataFrame, seed_path: Path = SEED_PATH) -> DataFrame:
    from pyspark.sql import functions as F
    from pyspark.sql.functions import pandas_udf

    seeds = str(seed_path)

    @pandas_udf("array<float>")
    def embed_udf(texts: pd.Series) -> pd.Series:
        return embedding_series(texts, get_embedder())

    @pandas_udf(SENTIMENT_DDL)
    def sentiment_udf(texts: pd.Series) -> pd.DataFrame:
        return sentiment_frame(texts)

    @pandas_udf(TOPIC_DDL)
    def topic_udf(vectors: pd.Series) -> pd.DataFrame:
        return topic_frame(vectors, _classifier(seeds))

    @pandas_udf(ENTITY_DDL)
    def entity_udf(texts: pd.Series) -> pd.DataFrame:
        return entity_frame(texts)

    @pandas_udf("string")
    def embedder_name_udf(texts: pd.Series) -> pd.Series:
        return pd.Series([get_embedder().name] * len(texts))

    out = (
        df.withColumn("embedding", embed_udf("text"))
        .withColumn("_s", sentiment_udf("text"))
        .withColumn("_t", topic_udf("embedding"))
        .withColumn("_e", entity_udf("text"))
        .withColumn("embedder", embedder_name_udf("text"))
    )
    return out.select(
        "record_id",
        "source",
        "created_at",
        F.col("rating").cast("int").alias("rating"),
        "text",
        "text_hash",
        F.col("_s.label").alias("sentiment_label"),
        F.col("_s.score").alias("sentiment_score"),
        F.col("_t.label").alias("topic_label"),
        F.col("_t.score").alias("topic_score"),
        "_e.products",
        "_e.features",
        "_e.plans",
        "embedding",
        "embedder",
    )


def write_enriched(df: DataFrame, path: Path) -> None:
    """Overwrite the whole output so re-runs are idempotent."""
    df.write.mode("overwrite").partitionBy("source").parquet(str(path))
