"""Bug-source interface and the tracker-neutral label convention for gold labels."""
import re
from typing import Protocol

from src.schemas import COMPONENTS, ROOT_CAUSES, SEVERITIES, BugReport, Label

PLATFORMS = ("web", "ios", "android", "smart_tv")
_LABEL = re.compile(r"^(severity|component|root_cause|platform)[:=_-](.+)$", re.IGNORECASE)
_ALLOWED = {"severity": SEVERITIES, "component": COMPONENTS, "root_cause": ROOT_CAUSES, "platform": PLATFORMS}


class SourceError(RuntimeError):
    """Raised when a bug source cannot return the requested reports."""


class BugSource(Protocol):
    """Anything that returns BugReports, optionally filtered to specific ids."""

    name: str

    def fetch(self, ids: list[str] | None = None) -> list[BugReport]: ...


def parse_triage_labels(labels: list[str]) -> tuple[Label | None, str]:
    """Read gold labels and platform from tracker labels such as 'severity:major' or 'root_cause-token_expiry'.

    Gold is returned only when a human has set all three triage labels; otherwise None.
    Platform is 'unknown' unless a platform label is present.
    """
    found: dict[str, str] = {}
    for raw in labels:
        match = _LABEL.match(raw.strip())
        if match:
            key, value = match.group(1).lower(), match.group(2).lower().replace("-", "_")
            if value in _ALLOWED[key]:
                found[key] = value
    gold = None
    if {"severity", "component", "root_cause"} <= found.keys():
        gold = Label(severity=found["severity"], component=found["component"],
                     root_cause_category=found["root_cause"])
    return gold, found.get("platform", "unknown")
