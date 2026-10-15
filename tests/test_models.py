import numpy as np
import pytest

from feedback_nlp.config import SEEDS_DIR
from feedback_nlp.embedder import HashingEmbedder, get_embedder
from feedback_nlp.entities import extract
from feedback_nlp.sentiment import classify
from feedback_nlp.topics import CentroidClassifier, fit_from_seeds


def test_hashing_embedder_is_deterministic_and_normalised() -> None:
    emb = HashingEmbedder(dim=64)
    a, b = emb.embed(["slow dashboard"]), HashingEmbedder(dim=64).embed(["slow dashboard"])
    assert a.shape == (1, 64) and a.dtype == np.float32
    np.testing.assert_array_equal(a, b)
    assert np.isclose(np.linalg.norm(a[0]), 1.0)
    assert np.all(emb.embed([""])[0] == 0)


def test_default_embedder_is_hashing() -> None:
    assert get_embedder().name.startswith("hashing")


@pytest.mark.parametrize(
    ("text", "label"),
    [
        ("Love it! Setup was intuitive and easy.", "positive"),
        ("Very disappointed, the page crashes.", "negative"),
        ("Not happy at all.", "negative"),
        ("Quick question about the invoice page.", "neutral"),
    ],
)
def test_sentiment(text: str, label: str) -> None:
    assert classify(text).label == label


def test_centroid_classifier_from_seeds() -> None:
    clf = fit_from_seeds(SEEDS_DIR / "topic_seeds.csv", HashingEmbedder())
    vectors = HashingEmbedder().embed(
        ["We were charged twice on our invoice", "The report page takes forever to load"]
    )
    labels, scores = clf.predict_vectors(vectors)
    assert labels == ["billing", "performance"]
    assert np.all((scores > 0) & (scores <= 1.0001))


def test_centroid_classifier_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        CentroidClassifier.fit([], [], HashingEmbedder())


def test_entity_extraction() -> None:
    found = extract("Prairie CRM CSV export fails on the Enterprise plan; Pro plan works")
    assert found.products == ["Prairie CRM"]
    assert found.features == ["export"]
    assert found.plans == ["Enterprise", "Pro"]
    assert extract("pro tips for a free afternoon").plans == []
