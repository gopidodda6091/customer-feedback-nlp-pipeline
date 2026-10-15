"""Generate synthetic Bluebonnet Software reviews and support tickets.

Writes data/raw/{reviews,tickets}.csv plus data/labelled/holdout.csv (true sentiment and
topic for a random hold-out sample). Noise is planted on purpose: non-English rows, exact
and near duplicates, near-empty texts and off-topic distractor sentences.
"""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

PRODUCTS = ["Prairie CRM", "Mesa Helpdesk", "Pecos Insights"]
FEATURES = [
    "dashboard",
    "CSV export",
    "SSO login",
    "mobile app",
    "REST API",
    "invoice page",
    "notifications",
    "search",
    "report builder",
]
PLANS = ["Free", "Starter", "Pro", "Enterprise"]

OPENERS = {
    "positive": [
        "Really happy with {product}.",
        "Great experience overall.",
        "Love it!",
        "Five stars from our team.",
        "Works great for us.",
    ],
    "negative": [
        "Very disappointed.",
        "This is frustrating.",
        "Honestly a terrible experience.",
        "Not happy at all.",
        "We are considering cancelling {product}.",
    ],
    "neutral": ["Quick question.", "Hi team,", "FYI.", "Following up on our account.", ""],
}
CLOSERS = {
    "positive": ["Thanks a lot!", "Keep up the good work.", "Highly recommend.", ""],
    "negative": ["Please fix this asap.", "Really poor.", "This is unacceptable.", ""],
    "neutral": ["Thanks.", "Let me know.", "Regards.", ""],
}
# (template, sentiment it is compatible with or "any")
TOPIC_SENTENCES: dict[str, list[tuple[str, str]]] = {
    "billing": [
        ("We were charged twice on the {plan} plan this month.", "negative"),
        ("The invoice for {product} shows the wrong amount.", "negative"),
        ("Upgrading from {plan} changed our billing date.", "any"),
        ("How do refunds work for the {plan} subscription?", "neutral"),
        ("Pricing for the {plan} tier is clear and the invoice was correct.", "positive"),
    ],
    "bug": [
        ("The {feature} in {product} crashes when I click save.", "negative"),
        ("The {feature} throws an error after the latest update.", "any"),
        ("Data disappears from the {feature} after refresh.", "negative"),
        ("Getting a 500 error when using the {feature}.", "any"),
    ],
    "feature_request": [
        ("It would be nice if {product} supported dark mode in the {feature}.", "any"),
        ("Please add bulk edit to the {feature}.", "any"),
        ("Can you add a Slack integration for the {feature}?", "neutral"),
        ("We would like scheduled exports from the {feature} on the {plan} plan.", "any"),
    ],
    "performance": [
        ("The {feature} takes forever to load in {product}.", "negative"),
        ("{product} is slow during peak hours.", "negative"),
        ("Load time for the {feature} is much faster now.", "positive"),
        ("Sync with the {feature} lags by several minutes.", "any"),
    ],
    "support": [
        ("The support agent resolved my ticket within an hour.", "positive"),
        ("Nobody from support has replied in three days.", "negative"),
        ("Chat support explained the {feature} settings.", "any"),
        ("I was transferred between support agents four times.", "negative"),
    ],
    "usability": [
        ("The {feature} screen in {product} is confusing to navigate.", "negative"),
        ("Setting up the {feature} was intuitive and easy.", "positive"),
        ("Too many clicks to find the {feature} settings.", "any"),
        ("The new layout of the {feature} is clean and simple.", "positive"),
    ],
}
NON_ENGLISH = [
    "La aplicación es muy lenta y no funciona bien.",
    "Die Rechnung ist falsch, bitte korrigieren Sie das.",
    "Le tableau de bord ne se charge pas depuis hier.",
    "Il supporto non ha ancora risposto alla mia richiesta.",
]
CONTEXT = [
    "We have {seats} seats on the {plan} plan.",
    "Our {team} team uses {product} every day.",
    "This started on {weekday}.",
    "Account {acct}.",
]
TEAMS = ["sales", "finance", "support", "operations", "marketing", "IT"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
SENTIMENT_WEIGHTS = {"positive": 0.35, "negative": 0.4, "neutral": 0.25}


@dataclass
class Record:
    record_id: str
    source: str
    created_at: str
    rating: str
    text: str
    sentiment: str = ""
    topic: str = ""


def fill(template: str, rng: random.Random) -> str:
    return template.format(
        product=rng.choice(PRODUCTS),
        feature=rng.choice(FEATURES),
        plan=rng.choice(PLANS),
        seats=rng.randint(2, 500),
        team=rng.choice(TEAMS),
        weekday=rng.choice(WEEKDAYS),
        acct=f"BB-{rng.randint(10000, 99999)}",
    )


def compose(topic: str, sentiment: str, rng: random.Random) -> str:
    options = [t for t, s in TOPIC_SENTENCES[topic] if s in (sentiment, "any")]
    parts = [fill(rng.choice(OPENERS[sentiment]), rng), fill(rng.choice(options), rng)]
    if rng.random() < 0.7:  # neutral context adds variety (and plan mentions)
        parts.insert(rng.randint(0, len(parts)), fill(rng.choice(CONTEXT), rng))
    if rng.random() < 0.1:  # off-topic distractor makes topic classification non-trivial
        other = rng.choice([t for t in TOPIC_SENTENCES if t != topic])
        parts.append(fill(rng.choice(TOPIC_SENTENCES[other])[0], rng))
    parts.append(rng.choice(CLOSERS[sentiment]))
    return " ".join(p for p in parts if p)


def rating_for(sentiment: str, rng: random.Random) -> str:
    return str(
        {"positive": rng.choice([4, 5]), "neutral": 3, "negative": rng.choice([1, 2])}[sentiment]
    )


def make_records(n: int, rng: random.Random) -> list[Record]:
    start = datetime(2026, 1, 1)
    records: list[Record] = []
    for i in range(n):
        source = "review" if rng.random() < 0.55 else "ticket"
        sentiment = rng.choices(list(SENTIMENT_WEIGHTS), list(SENTIMENT_WEIGHTS.values()))[0]
        topic = rng.choice(list(TOPIC_SENTENCES))
        ts = (start + timedelta(minutes=rng.randint(0, 60 * 24 * 270))).isoformat()
        rating = rating_for(sentiment, rng) if source == "review" else ""
        records.append(
            Record(
                f"FB-{i:06d}", source, ts, rating, compose(topic, sentiment, rng), sentiment, topic
            )
        )
    return records


def add_noise(records: list[Record], rng: random.Random) -> list[Record]:
    noisy: list[Record] = []
    n = len(records)
    for j in range(int(n * 0.03)):
        noisy.append(
            Record(f"FB-X{j:05d}", "ticket", records[j].created_at, "", rng.choice(NON_ENGLISH))
        )
    for j, base in enumerate(rng.sample(records, int(n * 0.04))):
        text = base.text if j % 2 else "  " + base.text.upper() + "  "
        noisy.append(Record(f"FB-D{j:05d}", base.source, base.created_at, base.rating, text))
    for j in range(int(n * 0.01)):
        noisy.append(Record(f"FB-E{j:05d}", "review", records[j].created_at, "3", "ok"))
    return noisy


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--holdout-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("data"))
    args = parser.parse_args()

    rng = random.Random(args.seed)
    records = make_records(args.n, rng)
    everything = records + add_noise(records, rng)
    rng.shuffle(everything)
    for source in ("review", "ticket"):
        rows = [
            {
                "record_id": r.record_id,
                "created_at": r.created_at,
                "rating": r.rating,
                "text": r.text,
            }
            for r in everything
            if r.source == source
        ]
        write_csv(args.out / "raw" / f"{source}s.csv", rows)
    holdout = rng.sample(records, int(len(records) * args.holdout_frac))
    write_csv(
        args.out / "labelled" / "holdout.csv",
        [
            {"record_id": r.record_id, "sentiment": r.sentiment, "topic": r.topic}
            for r in sorted(holdout, key=lambda r: r.record_id)
        ],
    )
    print(f"wrote {len(everything)} raw records and {len(holdout)} labelled hold-out rows")


if __name__ == "__main__":
    main()
