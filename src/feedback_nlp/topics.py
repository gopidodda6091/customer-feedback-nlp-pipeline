"""Topic classification: embed labelled seed examples, classify by nearest centroid."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from feedback_nlp.embedder import Embedder, l2_normalize


@dataclass(frozen=True)
class CentroidClassifier:
    labels: tuple[str, ...]
    centroids: np.ndarray  # shape (n_labels, dim), L2-normalised

    @classmethod
    def fit(
        cls, texts: Sequence[str], labels: Sequence[str], embedder: Embedder
    ) -> CentroidClassifier:
        if len(texts) != len(labels) or not texts:
            raise ValueError("texts and labels must be non-empty and the same length")
        vectors = embedder.embed(texts)
        names = tuple(sorted(set(labels)))
        label_arr = np.asarray(labels)
        centroids = np.vstack([vectors[label_arr == name].mean(axis=0) for name in names])
        return cls(names, l2_normalize(centroids))

    def predict_vectors(self, vectors: np.ndarray) -> tuple[list[str], np.ndarray]:
        """Return the best label and its cosine similarity for each row."""
        sims = vectors @ self.centroids.T
        best = sims.argmax(axis=1)
        return [self.labels[i] for i in best], sims[np.arange(len(best)), best]


def load_seeds(path: Path) -> pd.DataFrame:
    seeds = pd.read_csv(path, dtype=str)
    missing = {"topic", "text"} - set(seeds.columns)
    if missing:
        raise ValueError(f"seed file missing columns: {missing}")
    return seeds


def fit_from_seeds(path: Path, embedder: Embedder) -> CentroidClassifier:
    seeds = load_seeds(path)
    return CentroidClassifier.fit(seeds["text"].tolist(), seeds["topic"].tolist(), embedder)
