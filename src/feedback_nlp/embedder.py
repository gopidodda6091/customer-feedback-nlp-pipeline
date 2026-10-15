"""Embedder interface with a deterministic hashing fallback.

`HashingEmbedder` needs no downloads and is used by default (and always in tests).
Set ``USE_ST_MODEL=1`` to use sentence-transformers ``all-MiniLM-L6-v2`` instead.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from collections.abc import Sequence
from functools import lru_cache
from itertools import pairwise
from typing import Protocol

import numpy as np

LOG = logging.getLogger(__name__)
ST_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_TOKEN = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    """Anything that turns texts into L2-normalised float32 vectors."""

    name: str
    dim: int

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


def _features(text: str) -> list[str]:
    words = _TOKEN.findall(text.lower())
    bigrams = [f"{a}_{b}" for a, b in pairwise(words)]
    return words + bigrams


def _bucket(feature: str, dim: int) -> tuple[int, float]:
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "little")
    return value % dim, 1.0 if (value >> 63) & 1 else -1.0


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class HashingEmbedder:
    """Signed feature hashing of words and word bigrams. Deterministic across processes."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        self.name = f"hashing-{dim}"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feature in _features(text or ""):
                index, sign = _bucket(feature, self.dim)
                out[row, index] += sign
        return l2_normalize(out)


class SentenceTransformerEmbedder:
    """Wrapper around sentence-transformers; imported lazily so it stays optional."""

    def __init__(self, model_name: str = ST_MODEL_NAME) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device="cpu")
        self.dim = int(self._model.get_sentence_embedding_dimension())
        self.name = model_name

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._model.encode(list(texts), batch_size=64, show_progress_bar=False)
        return l2_normalize(np.asarray(vectors, dtype=np.float32))


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """Return the configured embedder, falling back to hashing if the model is unavailable."""
    if os.getenv("USE_ST_MODEL", "0") == "1":
        try:
            return SentenceTransformerEmbedder()
        except Exception as exc:
            LOG.warning("sentence-transformers unavailable (%s); using HashingEmbedder", exc)
    return HashingEmbedder()


def download_model() -> None:
    """Best-effort model pre-download used by `make setup`; never fails the build."""
    try:
        SentenceTransformerEmbedder()
        print(f"Downloaded {ST_MODEL_NAME}")
    except Exception as exc:
        print(f"Model download skipped ({type(exc).__name__}); HashingEmbedder will be used.")


if __name__ == "__main__":
    download_model()
