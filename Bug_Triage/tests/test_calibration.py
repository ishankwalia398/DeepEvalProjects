"""Kappa, bias statistics and the mock judge on hand-computed examples."""
import json

import pytest

from src.calibration import kappa_binary, kappa_weighted, partial_spearman, position_outcome, score_confusion
from src.llm import MockLLM
from src.prompts import judge_prompt

LOG = "2026-01-01T00:00:00Z ERROR [drm] license denied status=401 reason=token_expired"


def test_cohen_kappa_known_example():
    # observed agreement 0.75, chance agreement 0.5 -> kappa 0.5
    assert kappa_binary([True, True, False, False], [True, False, False, False]) == pytest.approx(0.5)


def test_weighted_kappa_known_examples():
    assert kappa_weighted([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == pytest.approx(1.0)
    # every pair swapped: sum(w*O)=1, sum(w*E)=0.5 -> 1 - 1/0.5 = -1
    assert kappa_weighted([1, 2], [2, 1]) == pytest.approx(-1.0)


def test_score_confusion_shape_and_counts():
    cm = score_confusion([5, 5, 1], [5, 4, 1])
    assert cm.shape == (5, 5)
    assert cm.loc[5, 4] == 1 and cm.loc[5, 5] == 1 and cm.loc[1, 1] == 1


def test_partial_spearman():
    control = [1, 2, 3, 4, 5]
    x = [5, 1, 4, 2, 3]
    assert partial_spearman(x, x, control) == pytest.approx(1.0)
    assert partial_spearman(control, x, control) == 0.0


def test_position_outcome():
    assert position_outcome("A", "B") == {"pairwise_flip": 0, "pairwise_agree": 1}
    assert position_outcome("A", "A") == {"pairwise_flip": 1, "pairwise_agree": 0}
    assert position_outcome("tie", "tie") == {"pairwise_flip": 0, "pairwise_agree": 0}
    assert position_outcome(None, "B")["pairwise_flip"] == 1


def test_mock_judge_v2_catches_invented_quote_v1_does_not():
    fake = ('The license call fails with "ERROR [drm] license cache overflow, terminating" so the DRM token '
            'expired and the cache grew without bound until the whole application stopped working for everyone.')
    v1 = json.loads(MockLLM().complete(judge_prompt("v1", LOG, fake)))
    v2 = json.loads(MockLLM().complete(judge_prompt("v2", LOG, fake)))
    assert v1["grounded"] is True and v1["score"] >= 4
    assert v2["grounded"] is False and v2["score"] <= 2
