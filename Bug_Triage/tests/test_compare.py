"""Bootstrap CI, recommendation rule and an end-to-end compare on a scratch database."""
import numpy as np

from src import store
from src.compare import bootstrap_diff_ci, compare_runs, recommend
from tests.conftest import make_item, make_result


def test_bootstrap_ci_is_deterministic_and_brackets_the_diff():
    a = np.array([0, 0, 1, 1, 0, 1, 0, 1], dtype=float)
    b = np.array([1, 1, 1, 1, 0, 1, 1, 1], dtype=float)
    ci = bootstrap_diff_ci(a, b)
    assert ci == bootstrap_diff_ci(a, b)
    assert ci[0] <= b.mean() - a.mean() <= ci[1]


def test_recommendation_rule():
    assert recommend((0.05, 0.30), [], 50) == "ship B"
    assert recommend((-0.30, -0.02), [], 50) == "keep A"
    assert recommend((-0.05, 0.20), [], 50).startswith("inconclusive: sample too small")
    assert "adds severity misses" in recommend((0.05, 0.30), ["BUG-001"], 50)


def test_compare_runs_lists_regressions(tmp_path):
    items = [make_item("BUG-001"), make_item("BUG-002")]
    run_a = [make_result("BUG-001"), make_result("BUG-002", "minor")]
    run_b = [make_result("BUG-001", "trivial"), make_result("BUG-002")]
    with store.connect(tmp_path / "t.db") as conn:
        store.sync_items(conn, items)
        a = store.save_triage_run(conn, "x", "mock", items, run_a)
        b = store.save_triage_run(conn, "y", "mock", items, run_b)
        c = compare_runs(conn, a, b, items)
    assert c["regressed"] == [{"item_id": "BUG-001", "fields": ["severity"]}]
    assert c["improved"] == [{"item_id": "BUG-002", "fields": ["severity"]}]
    assert c["new_severity_misses"] == ["BUG-001"]
