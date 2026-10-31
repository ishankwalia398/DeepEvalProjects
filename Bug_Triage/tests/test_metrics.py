"""Metrics on tiny fixtures with hand-computed answers."""
import pytest

from src.metrics import PARSE_ERROR, evaluate_run, field_metrics, flipped_items, severity_misses
from tests.conftest import make_item, make_result


def test_field_metrics_known_values():
    gold = ["critical", "critical", "minor", "minor"]
    pred = ["critical", "minor", "minor", None]
    m = field_metrics(gold, pred, ("critical", "minor"))
    assert m["accuracy"] == 0.5
    assert m["per_class"]["critical"] == pytest.approx({"precision": 1.0, "recall": 0.5, "f1": 2 / 3, "support": 2})
    assert m["per_class"]["minor"] == pytest.approx({"precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 2})
    assert m["macro_f1"] == pytest.approx((2 / 3 + 0.5) / 2)
    assert m["confusion"].loc["critical"].tolist() == [1, 1, 0]
    assert m["confusion"].loc["minor", PARSE_ERROR] == 1


def test_macro_f1_ignores_classes_absent_from_gold():
    m = field_metrics(["minor", "minor"], ["minor", "minor"], ("critical", "minor", "trivial"))
    assert m["macro_f1"] == 1.0


def test_severity_misses_only_counts_costly_downgrades():
    items = [make_item("BUG-001", "critical"), make_item("BUG-002", "major"),
             make_item("BUG-003", "major"), make_item("BUG-004", "minor")]
    results = [make_result("BUG-001", "trivial"), make_result("BUG-002", "critical"),
               make_result("BUG-003", "minor"), make_result("BUG-004", "trivial")]
    assert [m["item_id"] for m in severity_misses(items, results)] == ["BUG-001", "BUG-003"]


def test_evaluate_run_parse_errors_and_tricky_split():
    items = [make_item("BUG-001", tricky=True), make_item("BUG-002"), make_item("BUG-003")]
    results = [make_result("BUG-001", None), make_result("BUG-002"), make_result("BUG-003", "minor")]
    m = evaluate_run(items, results)
    assert m["parse_error_rate"] == pytest.approx(1 / 3)
    assert m["exact_match"] == pytest.approx(1 / 3)
    assert m["tricky_split"]["severity"] == {"tricky": 0.0, "non_tricky": 0.5}


def test_flipped_items_lists_disagreements():
    a = [make_result("BUG-001"), make_result("BUG-002")]
    b = [make_result("BUG-001", "minor"), make_result("BUG-002")]
    assert flipped_items({"a": a, "b": b}) == [
        {"item_id": "BUG-001", "field": "severity", "a": "major", "b": "minor"}
    ]
