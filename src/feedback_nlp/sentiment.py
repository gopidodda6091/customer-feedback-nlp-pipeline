"""Lexicon sentiment scorer with negation and intensifier handling (CPU-only, no downloads)."""

from __future__ import annotations

import re
from dataclasses import dataclass

_TOKEN = re.compile(r"[a-z0-9']+|[!?]")
POSITIVE = {
    "great": 2.0,
    "love": 2.0,
    "happy": 1.5,
    "good": 1.0,
    "easy": 1.0,
    "intuitive": 1.5,
    "clean": 0.5,
    "simple": 0.5,
    "faster": 1.0,
    "resolved": 1.0,
    "recommend": 1.5,
    "thanks": 0.5,
    "correct": 0.5,
    "stars": 1.0,
    "works": 0.5,
    "clear": 0.5,
}
NEGATIVE = {
    "disappointed": 2.0,
    "frustrating": 2.0,
    "terrible": 2.5,
    "poor": 1.5,
    "unacceptable": 2.5,
    "crashes": 1.5,
    "error": 1.0,
    "slow": 1.5,
    "forever": 1.0,
    "confusing": 1.5,
    "wrong": 1.5,
    "twice": 1.0,
    "disappears": 1.5,
    "cancelling": 2.0,
    "lags": 1.0,
    "nobody": 1.0,
    "fix": 0.5,
    "transferred": 1.0,
}
NEGATIONS = {"not", "no", "never", "isn't", "don't", "doesn't", "can't", "won't"}
INTENSIFIERS = {"very": 1.5, "really": 1.5, "honestly": 1.2, "extremely": 2.0, "too": 1.2}
NEUTRAL_BAND = 0.75


@dataclass(frozen=True)
class SentimentResult:
    label: str
    score: float


def raw_score(text: str) -> float:
    """Sum of lexicon weights; negation flips the next sentiment word, intensifiers scale it."""
    score, flip, boost = 0.0, False, 1.0
    for token in _TOKEN.findall((text or "").lower()):
        if token in NEGATIONS:
            flip = True
            continue
        if token in INTENSIFIERS:
            boost = INTENSIFIERS[token]
            continue
        weight = POSITIVE.get(token, 0.0) - NEGATIVE.get(token, 0.0)
        if weight:
            score += (-weight if flip else weight) * boost
            flip, boost = False, 1.0
    return score


def classify(text: str) -> SentimentResult:
    score = raw_score(text)
    if score > NEUTRAL_BAND:
        label = "positive"
    elif score < -NEUTRAL_BAND:
        label = "negative"
    else:
        label = "neutral"
    return SentimentResult(label, round(score, 3))
