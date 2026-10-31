"""Tiny hand-made fixtures shared by the unit tests."""
from src.schemas import BugReport, Label, TriageOutput, TriageResult


def make_item(item_id: str, severity: str = "major", component: str = "player",
              cause: str = "manifest_error", log: str = "", tricky: bool = False) -> BugReport:
    return BugReport(
        id=item_id, title="t", description="d", log_snippet=log, platform="web",
        gold=Label(severity=severity, component=component, root_cause_category=cause),
        tricky=tricky, tricky_reason="r" if tricky else None, rationale="r",
    )


def make_result(item_id: str, severity: str | None = "major", component: str = "player",
                cause: str = "manifest_error", cited: list[str] | None = None) -> TriageResult:
    if severity is None:
        return TriageResult(item_id=item_id, output=None, parse_error="invalid output")
    out = TriageOutput(severity=severity, component=component, root_cause_category=cause,
                       explanation="e", cited_log_lines=cited or [])
    return TriageResult(item_id=item_id, output=out)
