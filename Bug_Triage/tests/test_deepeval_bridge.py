"""The DeepEval citation metric on known cases (skipped when deepeval is not installed)."""
import pytest

pytest.importorskip("deepeval")

from deepeval.test_case import LLMTestCase  # noqa: E402

from src.deepeval_bridge import CitationGroundednessMetric  # noqa: E402

LOG = "2026-01-01T00:00:00Z ERROR [drm] license denied status=401"


def _case(cited: list[str], log: str = LOG) -> LLMTestCase:
    return LLMTestCase(name="BUG-001", input="i", actual_output="o", retrieval_context=[log],
                       metadata={"cited_log_lines": cited})


def test_metric_scores_share_of_valid_citations():
    metric = CitationGroundednessMetric()
    assert metric.measure(_case(["license denied status=401", "invented line"])) == 0.5
    assert not metric.is_successful()
    assert metric.measure(_case(["license denied status=401"])) == 1.0
    assert metric.is_successful()


def test_no_citations_fails_unless_the_log_is_empty():
    metric = CitationGroundednessMetric()
    assert metric.measure(_case([])) == 0.0 and not metric.is_successful()
    assert metric.measure(_case([], log="")) == 1.0 and metric.is_successful()
