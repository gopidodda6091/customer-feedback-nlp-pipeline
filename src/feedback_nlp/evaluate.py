"""Evaluate enriched predictions against the labelled hold-out and log to MLflow."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class LabelMetrics:
    label: str
    precision: float
    recall: float
    f1: float
    support: int


def confusion_matrix(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]
) -> pd.DataFrame:
    """Rows are true labels, columns are predicted labels."""
    matrix = pd.DataFrame(0, index=list(labels), columns=list(labels), dtype=int)
    for truth, pred in zip(y_true, y_pred, strict=True):
        if truth in matrix.index and pred in matrix.columns:
            matrix.loc[truth, pred] += 1
    matrix.index.name, matrix.columns.name = "true", "predicted"
    return matrix


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def per_label_metrics(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]
) -> list[LabelMetrics]:
    cm = confusion_matrix(y_true, y_pred, labels)
    out = []
    for label in labels:
        tp = int(cm.loc[label, label])
        precision = _safe_div(tp, int(cm[label].sum()))
        recall = _safe_div(tp, int(cm.loc[label].sum()))
        f1 = _safe_div(2 * precision * recall, precision + recall)
        out.append(LabelMetrics(label, precision, recall, f1, int(cm.loc[label].sum())))
    return out


def summary_metrics(
    rows: list[LabelMetrics], y_true: Sequence[str], y_pred: Sequence[str]
) -> dict[str, float]:
    n = len(rows) or 1
    correct = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == p)
    return {
        "accuracy": _safe_div(correct, len(y_true)),
        "macro_precision": sum(r.precision for r in rows) / n,
        "macro_recall": sum(r.recall for r in rows) / n,
        "macro_f1": sum(r.f1 for r in rows) / n,
    }


def error_analysis(joined: pd.DataFrame, task: str) -> pd.DataFrame:
    """Misclassified rows with text and confidence, least confident first."""
    wrong = joined[joined[f"{task}_true"] != joined[f"{task}_label"]]
    cols = ["record_id", "source", "text", f"{task}_true", f"{task}_label", f"{task}_score"]
    return wrong[cols].sort_values(f"{task}_score").reset_index(drop=True)


def join_holdout(enriched: pd.DataFrame, holdout: pd.DataFrame) -> pd.DataFrame:
    renamed = holdout.rename(columns={"sentiment": "sentiment_true", "topic": "topic_true"})
    return enriched.merge(renamed, on="record_id", how="inner")


def evaluate_task(
    joined: pd.DataFrame, task: str, labels: Sequence[str], out_dir: Path
) -> dict[str, float]:
    """Compute metrics for one task and write its CSV artifacts; returns flat metrics."""
    y_true, y_pred = joined[f"{task}_true"].tolist(), joined[f"{task}_label"].tolist()
    rows = per_label_metrics(y_true, y_pred, labels)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([r.__dict__ for r in rows]).to_csv(out_dir / f"{task}_per_label.csv", index=False)
    confusion_matrix(y_true, y_pred, labels).to_csv(out_dir / f"{task}_confusion_matrix.csv")
    error_analysis(joined, task).to_csv(out_dir / f"{task}_errors.csv", index=False)
    flat = {f"{task}_{k}": v for k, v in summary_metrics(rows, y_true, y_pred).items()}
    for r in rows:
        flat.update(
            {
                f"{task}_{r.label}_precision": r.precision,
                f"{task}_{r.label}_recall": r.recall,
                f"{task}_{r.label}_f1": r.f1,
            }
        )
    return flat


def log_to_mlflow(
    metrics: dict[str, float],
    params: dict[str, str],
    artifacts_dir: Path,
    tracking_uri: str,
    experiment: str = "feedback-nlp-eval",
) -> str:
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run() as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_artifacts(str(artifacts_dir), artifact_path="evaluation")
        return run.info.run_id
