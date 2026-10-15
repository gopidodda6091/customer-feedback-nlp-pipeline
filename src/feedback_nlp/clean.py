"""Normalise, language-filter and de-duplicate raw feedback."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

_WS = re.compile(r"\s+")
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_WORD = re.compile(r"[a-zà-ÿ']+")
EN_STOPWORDS = frozenset(
    "a an the and or but if is are was were be been to of in on for with at by from it this "
    "that we our you your i my me not no can do does did have has had will would please "
    "when after so too very all just during about how what they there more up".split()
)
FOREIGN_STOPWORDS = frozenset(
    "la el los las es y muy bien una que de del die der das und ist bitte sie nicht le les "
    "des est ne pas il non ha alla mia che per".split()
)
MIN_TOKENS = 3


@dataclass
class CleanStats:
    input_rows: int = 0
    dropped: dict[str, int] = field(default_factory=dict)
    output_rows: int = 0


def normalize_text(text: str | None) -> str:
    """NFKC-normalise, strip control characters and collapse whitespace."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = _CONTROL.sub(" ", text)
    return _WS.sub(" ", text).strip()


def dedupe_key(text: str) -> str:
    """Case- and whitespace-insensitive content hash."""
    return hashlib.sha1(normalize_text(text).lower().encode("utf-8")).hexdigest()


def is_english(text: str) -> bool:
    """Cheap heuristic: more English than foreign stopwords, and little non-ASCII text."""
    tokens = _WORD.findall(text.lower())
    if not tokens:
        return False
    non_ascii = sum(1 for ch in text if ord(ch) > 127) / max(len(text), 1)
    english = sum(1 for t in tokens if t in EN_STOPWORDS)
    foreign = sum(1 for t in tokens if t in FOREIGN_STOPWORDS)
    return english > foreign and non_ascii < 0.05


def load_raw(raw_dir: Path) -> pd.DataFrame:
    frames = []
    for source in ("review", "ticket"):
        frame = pd.read_csv(raw_dir / f"{source}s.csv", dtype=str, keep_default_na=False)
        frames.append(frame.assign(source=source))
    return pd.concat(frames, ignore_index=True)


def clean_frame(raw: pd.DataFrame) -> tuple[pd.DataFrame, CleanStats]:
    """Return cleaned rows (one per distinct text, earliest kept) and drop counts."""
    stats = CleanStats(input_rows=len(raw))
    df = raw.copy()
    df["text"] = df["text"].map(normalize_text)

    too_short = df["text"].str.split().str.len().fillna(0) < MIN_TOKENS
    stats.dropped["too_short"] = int(too_short.sum())
    df = df[~too_short]

    english = df["text"].map(is_english)
    stats.dropped["non_english"] = int((~english).sum())
    df = df[english]

    df = df.assign(text_hash=df["text"].map(dedupe_key)).sort_values(["created_at", "record_id"])
    dupes = df.duplicated("text_hash", keep="first")
    stats.dropped["duplicate"] = int(dupes.sum())
    df = df[~dupes]

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").astype("Int64")
    stats.output_rows = len(df)
    columns = ["record_id", "source", "created_at", "rating", "text", "text_hash"]
    return df[columns].reset_index(drop=True), stats
