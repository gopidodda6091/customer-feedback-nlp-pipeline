"""Cosine-similarity search over the enriched table's embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from feedback_nlp.config import ENRICHED_PATH
from feedback_nlp.embedder import Embedder, get_embedder


class SemanticIndex:
    def __init__(self, frame: pd.DataFrame, embedder: Embedder) -> None:
        if frame.empty:
            raise ValueError("cannot index an empty frame")
        self.frame = frame.reset_index(drop=True)
        self.matrix = np.vstack(self.frame["embedding"].to_numpy()).astype(np.float32)
        self.embedder = embedder

    @classmethod
    def from_parquet(cls, path: Path, embedder: Embedder | None = None) -> SemanticIndex:
        return cls(pd.read_parquet(path), embedder or get_embedder())

    def search(self, query: str, k: int = 5, topic: str | None = None) -> pd.DataFrame:
        scores = self.matrix @ self.embedder.embed([query])[0]
        result = self.frame.assign(similarity=scores)
        if topic:
            result = result[result["topic_label"] == topic]
        cols = ["record_id", "source", "similarity", "sentiment_label", "topic_label", "text"]
        return result.nlargest(k, "similarity")[cols].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Search enriched feedback")
    parser.add_argument("query")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--topic")
    parser.add_argument("--path", type=Path, default=ENRICHED_PATH)
    args = parser.parse_args()
    hits = SemanticIndex.from_parquet(args.path).search(args.query, args.k, args.topic)
    with pd.option_context("display.max_colwidth", 90, "display.width", 200):
        print(hits.to_string(index=False))


if __name__ == "__main__":
    main()
