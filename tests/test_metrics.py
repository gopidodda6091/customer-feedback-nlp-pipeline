import pandas as pd
import pytest

from feedback_nlp.evaluate import (
    confusion_matrix,
    error_analysis,
    evaluate_task,
    per_label_metrics,
    summary_metrics,
)

Y_TRUE = ["a", "a", "a", "b", "b", "c"]
Y_PRED = ["a", "a", "b", "b", "c", "c"]


def test_confusion_matrix() -> None:
    cm = confusion_matrix(Y_TRUE, Y_PRED, ["a", "b", "c"])
    assert cm.loc["a", "a"] == 2 and cm.loc["a", "b"] == 1 and cm.loc["b", "c"] == 1
    assert int(cm.to_numpy().sum()) == len(Y_TRUE)


def test_per_label_precision_recall_f1() -> None:
    rows = {r.label: r for r in per_label_metrics(Y_TRUE, Y_PRED, ["a", "b", "c"])}
    assert rows["a"].precision == pytest.approx(1.0)
    assert rows["a"].recall == pytest.approx(2 / 3)
    assert rows["a"].f1 == pytest.approx(0.8)
    assert rows["c"].precision == pytest.approx(0.5) and rows["c"].recall == pytest.approx(1.0)
    assert rows["b"].support == 2


def test_zero_division_is_safe() -> None:
    rows = per_label_metrics(["a"], ["a"], ["a", "z"])
    assert rows[1].precision == 0.0 and rows[1].f1 == 0.0


def test_summary_metrics() -> None:
    rows = per_label_metrics(Y_TRUE, Y_PRED, ["a", "b", "c"])
    summary = summary_metrics(rows, Y_TRUE, Y_PRED)
    assert summary["accuracy"] == pytest.approx(4 / 6)
    assert summary["macro_f1"] == pytest.approx(sum(r.f1 for r in rows) / 3)


def test_evaluate_task_writes_artifacts(tmp_path) -> None:
    joined = pd.DataFrame(
        {
            "record_id": list("123456"),
            "source": ["review"] * 6,
            "text": ["t"] * 6,
            "topic_true": Y_TRUE,
            "topic_label": Y_PRED,
            "topic_score": [0.9, 0.8, 0.2, 0.7, 0.1, 0.6],
        }
    )
    metrics = evaluate_task(joined, "topic", ["a", "b", "c"], tmp_path)
    assert metrics["topic_accuracy"] == pytest.approx(4 / 6)
    errors = pd.read_csv(tmp_path / "topic_errors.csv")
    assert errors["record_id"].astype(str).tolist() == ["5", "3"]  # least confident first
    assert (tmp_path / "topic_confusion_matrix.csv").exists()
    assert len(error_analysis(joined, "topic")) == 2
