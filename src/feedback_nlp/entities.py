"""Rule/gazetteer entity extraction for products, features and plans."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

GAZETTEER: dict[str, dict[str, tuple[str, ...]]] = {
    "product": {
        "Prairie CRM": ("prairie crm", "prairie"),
        "Mesa Helpdesk": ("mesa helpdesk", "mesa"),
        "Pecos Insights": ("pecos insights", "pecos"),
    },
    "feature": {
        "dashboard": ("dashboard",),
        "export": ("csv export", "export", "exports"),
        "sso": ("sso login", "sso", "single sign-on"),
        "mobile_app": ("mobile app",),
        "api": ("rest api", "api"),
        "invoice": ("invoice page", "invoice", "invoices"),
        "notifications": ("notifications", "notification"),
        "search": ("search",),
        "reports": ("report builder", "reports"),
    },
    "plan": {
        "Free": ("free plan", "free tier", "free subscription"),
        "Starter": ("starter",),
        "Pro": ("pro plan", "pro tier", "pro subscription"),
        "Enterprise": ("enterprise",),
    },
}
# "Free"/"Pro" are common words, so they only match with plan/tier context (see aliases above).


@dataclass
class Entities:
    products: list[str] = field(default_factory=list)
    features: list[str] = field(default_factory=list)
    plans: list[str] = field(default_factory=list)


def _compile() -> dict[str, list[tuple[re.Pattern[str], str]]]:
    compiled: dict[str, list[tuple[re.Pattern[str], str]]] = {}
    for kind, entries in GAZETTEER.items():
        compiled[kind] = [
            (re.compile(rf"\b{re.escape(alias)}\b"), canonical)
            for canonical, aliases in entries.items()
            for alias in aliases
        ]
    return compiled


_PATTERNS = _compile()


def _match(kind: str, text: str) -> list[str]:
    found: list[str] = []
    for pattern, canonical in _PATTERNS[kind]:
        if canonical not in found and pattern.search(text):
            found.append(canonical)
    return sorted(found)


def extract(text: str) -> Entities:
    lowered = (text or "").lower()
    return Entities(_match("product", lowered), _match("feature", lowered), _match("plan", lowered))
