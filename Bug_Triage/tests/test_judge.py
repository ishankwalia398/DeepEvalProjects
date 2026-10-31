"""Judge failure reasons and the description context for tracker tickets (no network)."""
import json

from src.judge import score_explanation
from src.llm import LLMError, MockLLM
from src.prompts import judge_prompt
from src.schemas import BugReport, TriageOutput, TriageResult
from src.triage import judge_results

LOG = "2026-01-01 ERROR [player] manifest parse failed"
EXPLANATION = 'The log shows "ERROR [player] manifest parse failed". Manifest error in the player.'


class FixedReply:
    name = "fixed"

    def __init__(self, reply: str | None = None):
        self.reply, self.prompts = reply, []

    def complete(self, prompt: str, system: str | None = None) -> str:
        self.prompts.append(prompt)
        if self.reply is None:
            raise LLMError("connection refused")
        return self.reply


def test_failure_reason_is_returned_for_provider_and_parse_errors():
    assert score_explanation(FixedReply(), "v2", LOG, EXPLANATION) == (None, "llm error: connection refused")
    score, error = score_explanation(FixedReply("I would give it a 4."), "v2", LOG, EXPLANATION)
    assert score is None and error.startswith("invalid judge output") and "I would give it a 4." in error


def test_description_is_added_only_when_given():
    assert "<<<DESCRIPTION" not in judge_prompt("v2", LOG, EXPLANATION)
    prompt = judge_prompt("v2", LOG, EXPLANATION, "Seen in 12.2.0 as well, so not a regression.")
    assert "<<<DESCRIPTION\nSeen in 12.2.0 as well, so not a regression.\nDESCRIPTION>>>" in prompt
    assert prompt.index("<<<DESCRIPTION") < prompt.index("<<<LOG")
    assert json.loads(MockLLM().complete(prompt))["score"] >= 1  # mock still reads the LOG fence


def _item(source: str) -> BugReport:
    return BugReport(id="PROJ-1", title="t", description="the description", log_snippet=LOG,
                     platform="unknown", source=source)


def _result() -> TriageResult:
    out = TriageOutput(severity="major", component="player", root_cause_category="manifest_error",
                       explanation=EXPLANATION, cited_log_lines=[LOG])
    return TriageResult(item_id="PROJ-1", output=out, parse_error=None)


def test_judge_results_sends_description_for_tracker_tickets_only():
    reply = json.dumps({"score": 4, "grounded": True, "reason": "ok"})
    for source, expected in (("jira", True), ("file", False)):
        judge = FixedReply(reply)
        scores, errors = judge_results(judge, [_item(source)], [_result()])
        assert scores["PROJ-1"].score == 4 and errors == {}
        assert ("<<<DESCRIPTION\nthe description" in judge.prompts[0]) is expected


def test_judge_results_reports_why_a_score_is_missing():
    scores, errors = judge_results(FixedReply(), [_item("jira")], [_result()])
    assert scores == {"PROJ-1": None}
    assert errors == {"PROJ-1": "llm error: connection refused"}
