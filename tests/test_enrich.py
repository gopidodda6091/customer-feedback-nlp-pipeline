"""UDF contract tests: schema, row preservation and Spark/pandas parity."""

import numpy as np
import pandas as pd
import pytest

from feedback_nlp.embedder import get_embedder
from feedback_nlp.enrich import OUTPUT_COLUMNS, enrich_pandas, enrich_spark
from feedback_nlp.semantic_search import SemanticIndex


def test_pandas_enrichment_contract(clean_sample: pd.DataFrame) -> None:
    out = enrich_pandas(clean_sample)
    assert list(out.columns) == OUTPUT_COLUMNS
    assert len(out) == len(clean_sample)
    assert set(out["sentiment_label"]) <= {"positive", "neutral", "negative"}
    assert out.loc[0, "products"] == ["Prairie CRM"] and out.loc[0, "features"] == ["dashboard"]
    assert out.loc[0, "sentiment_label"] == "negative"
    assert all(len(v) == 384 for v in out["embedding"])


def test_pandas_enrichment_is_deterministic(clean_sample: pd.DataFrame) -> None:
    pd.testing.assert_frame_equal(enrich_pandas(clean_sample), enrich_pandas(clean_sample))


def test_semantic_search_ranks_relevant_first(clean_sample: pd.DataFrame) -> None:
    index = SemanticIndex(enrich_pandas(clean_sample), get_embedder())
    hits = index.search("refunds for the subscription", k=2)
    assert hits.loc[0, "record_id"] == "T-2"
    assert hits["similarity"].is_monotonic_decreasing


@pytest.mark.spark
def test_spark_udfs_match_pandas(spark, clean_sample: pd.DataFrame) -> None:
    sdf = spark.createDataFrame(clean_sample.astype({"rating": "float"}))
    result = enrich_spark(sdf)
    assert result.columns == OUTPUT_COLUMNS
    types = dict(result.dtypes)
    assert types["embedding"] == "array<float>"
    assert types["products"] == "array<string>"
    assert types["sentiment_score"] == "double"

    got = result.toPandas().sort_values("record_id").reset_index(drop=True)
    expected = enrich_pandas(clean_sample)
    for col in ("sentiment_label", "topic_label", "embedder"):
        assert got[col].tolist() == expected[col].tolist()
    assert [list(v) for v in got["products"]] == expected["products"].tolist()
    np.testing.assert_allclose(
        np.vstack(got["embedding"]), np.vstack(expected["embedding"]), atol=1e-6
    )
    np.testing.assert_allclose(got["topic_score"], expected["topic_score"], atol=1e-3)
