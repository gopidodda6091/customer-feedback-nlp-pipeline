from __future__ import annotations

import re
import shutil
import subprocess

import pandas as pd
import pytest

from feedback_nlp.embedder import get_embedder


@pytest.fixture(autouse=True)
def hashing_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests never download models: force the deterministic HashingEmbedder."""
    monkeypatch.setenv("USE_ST_MODEL", "0")
    get_embedder.cache_clear()


def java_major() -> int | None:
    """Major version of the `java` on PATH, or None when there is no usable JVM."""
    if shutil.which("java") is None:
        return None
    proc = subprocess.run(["java", "-version"], capture_output=True, text=True)
    match = re.search(r'version "(\d+)(?:\.(\d+))?', proc.stderr)
    if proc.returncode != 0 or not match:
        return None
    major = int(match.group(1))
    return int(match.group(2) or 0) if major == 1 else major


@pytest.fixture(scope="session")
def spark():
    major = java_major()
    if major not in (8, 11, 17):
        pytest.skip(f"Spark 3.5 Arrow UDFs need Java 8/11/17 (found {major}); pandas engine tested")
    from feedback_nlp.enrich import build_spark

    session = build_spark("feedback-nlp-tests")
    yield session
    session.stop()


@pytest.fixture
def clean_sample() -> pd.DataFrame:
    texts = [
        "Very disappointed. The dashboard in Prairie CRM crashes when I click save.",
        "Love it! Setting up the SSO login was intuitive and easy. Thanks a lot!",
        "Quick question. How do refunds work for the Pro subscription? Thanks.",
        "Not happy at all. Mesa Helpdesk is slow during peak hours.",
    ]
    return pd.DataFrame(
        {
            "record_id": [f"T-{i}" for i in range(len(texts))],
            "source": ["review", "review", "ticket", "ticket"],
            "created_at": ["2026-01-0%dT00:00:00" % (i + 1) for i in range(len(texts))],
            "rating": pd.array([1, 5, None, None], dtype="Int64"),
            "text": texts,
            "text_hash": [f"h{i}" for i in range(len(texts))],
        }
    )
