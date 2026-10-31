"""Deterministic citation check: every cited log line must appear in the input log."""
from pydantic import BaseModel

from src.schemas import BugReport, TriageResult


def normalize(text: str) -> str:
    """Collapse all whitespace runs to single spaces."""
    return " ".join(text.split())


class CitationCheck(BaseModel):
    """Valid and invalid citations for one triage output."""

    item_id: str
    valid: list[str]
    invalid: list[str]


def check_citations(item_id: str, log: str, cited: list[str]) -> CitationCheck:
    """Split cited lines into those found verbatim (whitespace-normalised) in the log and the rest."""
    nlog = normalize(log)
    valid, invalid = [], []
    for line in cited:
        (valid if normalize(line) and normalize(line) in nlog else invalid).append(line)
    return CitationCheck(item_id=item_id, valid=valid, invalid=invalid)


def groundedness_report(items: list[BugReport], results: list[TriageResult]) -> dict:
    """Hallucinated-citation rate, zero-valid-citation fraction and worst offenders for one run."""
    logs = {i.id: i.log_snippet for i in items}
    checks = [
        check_citations(r.item_id, logs[r.item_id], r.output.cited_log_lines)
        for r in results if r.output is not None
    ]
    total = sum(len(c.valid) + len(c.invalid) for c in checks)
    invalid = sum(len(c.invalid) for c in checks)
    worst = sorted((c for c in checks if c.invalid), key=lambda c: (-len(c.invalid), c.item_id))
    return {
        "citations_total": total,
        "citations_invalid": invalid,
        "hallucinated_citation_rate": invalid / total if total else 0.0,
        "zero_valid_fraction": sum(not c.valid for c in checks) / len(checks) if checks else 0.0,
        "worst_offenders": [c.model_dump() for c in worst[:10]],
        "checks": {c.item_id: c for c in checks},
    }
