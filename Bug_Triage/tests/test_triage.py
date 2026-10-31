"""Strict parsing, paraphrase transform and MockLLM determinism."""
import json

import pytest

from src.llm import LLMError, MockLLM, get_judge_llm, get_llm
from src.prompts import triage_prompt
from src.schemas import load_bug_reports
from src.triage import paraphrase, parse_triage

VALID = {"severity": "major", "component": "cdn", "root_cause_category": "network_timeout",
         "explanation": "x", "cited_log_lines": ["a"]}


def test_parse_valid_and_fenced_json():
    assert parse_triage(json.dumps(VALID))[0].component == "cdn"
    assert parse_triage("```json\n" + json.dumps(VALID) + "\n```")[0] is not None


def test_parse_rejects_malformed_unknown_labels_and_extra_fields():
    assert parse_triage("Sure! severity is major")[0] is None
    assert parse_triage(json.dumps({**VALID, "severity": "blocker"}))[1]
    assert parse_triage(json.dumps({**VALID, "confidence": 0.9}))[1]


def test_paraphrase_is_deterministic_and_changes_text():
    text = "First sentence. Second one! Third?"
    assert paraphrase(text) == paraphrase(text) == "second one!  third?  first sentence."


def test_mock_is_deterministic():
    item = load_bug_reports()[0]
    a, b = MockLLM(), MockLLM()
    assert a.complete(triage_prompt("v2_structured", item)) == b.complete(triage_prompt("v2_structured", item))


def test_mock_v1_obeys_trailing_injection_but_v3_does_not():
    item = next(i for i in load_bug_reports() if i.id == "BUG-039")
    mock = MockLLM()
    assert json.loads(mock.complete(triage_prompt("v1_basic", item)))["severity"] == "trivial"
    assert json.loads(mock.complete(triage_prompt("v3_cot_light", item)))["severity"] != "trivial"


def test_default_provider_is_offline_mock(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    assert get_llm().name == "mock"


def test_judge_uses_judge_provider_and_model(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("JUDGE_PROVIDER", "anthropic")
    monkeypatch.setenv("JUDGE_MODEL", "claude-opus-5-5")
    assert get_judge_llm().name == "anthropic:claude-opus-5-5"
    assert get_judge_llm("mock").name == "mock"


def test_judge_model_does_not_leak_into_another_provider(monkeypatch):
    monkeypatch.setenv("JUDGE_PROVIDER", "anthropic")
    monkeypatch.setenv("JUDGE_MODEL", "claude-opus-5-5")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "")
    assert get_judge_llm("openrouter").model == "openrouter/auto"


def test_mock_refuses_tracker_tickets():
    from src.schemas import BugReport

    ticket = BugReport(id="PROJ-1", title="t", description="d", log_snippet="", platform="unknown", source="jira")
    with pytest.raises(LLMError, match="real provider"):
        MockLLM().complete(triage_prompt("v3_cot_light", ticket))
