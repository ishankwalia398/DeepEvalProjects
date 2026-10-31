"""Deterministic citation checks."""
import pytest

from src.groundedness import check_citations, groundedness_report
from tests.conftest import make_item, make_result

LOG = "2026-01-01T00:00:00Z ERROR [drm] license denied status=401\n2026-01-01T00:00:01Z INFO [player] stop"


def test_whitespace_is_normalised():
    check = check_citations("BUG-001", LOG, ["ERROR   [drm]  license denied\tstatus=401"])
    assert check.valid and not check.invalid


def test_fabricated_and_empty_citations_are_invalid():
    check = check_citations("BUG-001", LOG, ["ERROR [cdn] timeout", ""])
    assert check.invalid == ["ERROR [cdn] timeout", ""]


def test_report_rates_and_worst_offenders():
    items = [make_item("BUG-001", log=LOG), make_item("BUG-002", log=LOG), make_item("BUG-003", log="")]
    results = [
        make_result("BUG-001", cited=["license denied status=401", "made up line"]),
        make_result("BUG-002", cited=["made up 1", "made up 2"]),
        make_result("BUG-003", cited=[]),
    ]
    g = groundedness_report(items, results)
    assert g["citations_total"] == 4 and g["citations_invalid"] == 3
    assert g["hallucinated_citation_rate"] == pytest.approx(0.75)
    assert g["zero_valid_fraction"] == pytest.approx(2 / 3)
    assert [w["item_id"] for w in g["worst_offenders"]] == ["BUG-002", "BUG-001"]


def test_parse_errors_are_skipped():
    g = groundedness_report([make_item("BUG-001", log=LOG)], [make_result("BUG-001", None)])
    assert g["citations_total"] == 0 and g["hallucinated_citation_rate"] == 0.0
