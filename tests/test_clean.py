import pandas as pd

from feedback_nlp.clean import clean_frame, dedupe_key, is_english, normalize_text


def test_normalize_collapses_whitespace_and_controls() -> None:
    assert normalize_text("  Hello\t\x07 world \n") == "Hello world"
    assert normalize_text(None) == ""
    assert normalize_text("ﬁle") == "file"  # NFKC ligature


def test_dedupe_key_ignores_case_and_spacing() -> None:
    assert dedupe_key("The App  is SLOW") == dedupe_key("the app is slow ")
    assert dedupe_key("app is slow") != dedupe_key("app is fast")


def test_language_heuristic() -> None:
    assert is_english("The invoice for my account is wrong and I want a refund")
    assert not is_english("Die Rechnung ist falsch, bitte korrigieren Sie das.")
    assert not is_english("La aplicación es muy lenta y no funciona bien.")


def test_clean_frame_drops_short_foreign_and_duplicates() -> None:
    raw = pd.DataFrame(
        {
            "record_id": ["A", "B", "C", "D", "E"],
            "source": ["review"] * 5,
            "created_at": ["2026-01-02", "2026-01-01", "2026-01-03", "2026-01-04", "2026-01-05"],
            "rating": ["5", "5", "", "3", "1"],
            "text": [
                "The export is great for our team",
                "  THE EXPORT IS GREAT FOR OUR TEAM ",
                "Le tableau de bord ne se charge pas depuis hier.",
                "ok",
                "The search is slow for us",
            ],
        }
    )
    clean, stats = clean_frame(raw)
    assert stats.dropped == {"too_short": 1, "non_english": 1, "duplicate": 1}
    assert clean["record_id"].tolist() == ["B", "E"]  # earliest duplicate wins
    assert clean["text"].iloc[0] == "THE EXPORT IS GREAT FOR OUR TEAM"
    assert str(clean["rating"].dtype) == "Int64"


def test_clean_is_idempotent() -> None:
    raw = pd.DataFrame(
        {
            "record_id": ["A"],
            "source": ["ticket"],
            "created_at": ["2026-01-01"],
            "rating": [""],
            "text": ["The API returns an error for my account"],
        }
    )
    once, _ = clean_frame(raw)
    twice, _ = clean_frame(once.drop(columns="text_hash").astype({"rating": str}))
    assert once["text_hash"].tolist() == twice["text_hash"].tolist()
