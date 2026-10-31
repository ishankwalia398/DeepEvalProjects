"""Pydantic models and data loaders shared across the benchmark."""
import json
from pathlib import Path
from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
BUG_REPORTS_PATH = DATA_DIR / "bug_reports.json"
HUMAN_JUDGEMENTS_PATH = DATA_DIR / "human_judgements.json"

Severity = Literal["critical", "major", "minor", "trivial"]
Component = Literal["player", "drm", "cdn", "api", "ui", "analytics"]
RootCause = Literal[
    "token_expiry", "manifest_error", "network_timeout", "ui_regression",
    "config_error", "memory_leak", "data_mismatch", "unknown",
]
Platform = Literal["web", "ios", "android", "smart_tv", "unknown"]
ITEM_ID_PATTERN = r"^[A-Za-z][A-Za-z0-9_]*-\d+$"

SEVERITIES: tuple[str, ...] = get_args(Severity)
COMPONENTS: tuple[str, ...] = get_args(Component)
ROOT_CAUSES: tuple[str, ...] = get_args(RootCause)
LABEL_FIELDS: dict[str, tuple[str, ...]] = {
    "severity": SEVERITIES,
    "component": COMPONENTS,
    "root_cause_category": ROOT_CAUSES,
}


class Label(BaseModel):
    """The three triage labels, used for gold answers and predictions."""

    model_config = ConfigDict(extra="forbid")
    severity: Severity
    component: Component
    root_cause_category: RootCause


class BugReport(BaseModel):
    """One bug report: a labelled benchmark item, or a tracker ticket (gold may be None)."""

    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=ITEM_ID_PATTERN)
    title: str
    description: str
    log_snippet: str
    platform: Platform
    gold: Label | None = None
    tricky: bool = False
    tricky_reason: str | None = None
    rationale: str = ""
    source: str = "file"
    url: str | None = None


class TriageOutput(Label):
    """Structured answer the LLM must return for a triage prompt."""

    explanation: str
    cited_log_lines: list[str]


class TriageResult(BaseModel):
    """Outcome of one triage call; output is None when parsing failed."""

    item_id: str
    output: TriageOutput | None
    parse_error: str | None = None
    raw: str = ""


class JudgeScore(BaseModel):
    """Pointwise judge verdict on one explanation."""

    model_config = ConfigDict(extra="forbid")
    score: int = Field(ge=1, le=5)
    grounded: bool
    reason: str


class PairwiseVerdict(BaseModel):
    """Pairwise judge verdict between explanation A and B."""

    model_config = ConfigDict(extra="forbid")
    winner: Literal["A", "B", "tie"]
    reason: str


class HumanJudgement(BaseModel):
    """Human score for a reference explanation, optionally with a pairwise alternative."""

    model_config = ConfigDict(extra="forbid")
    item_id: str = Field(pattern=ITEM_ID_PATTERN)
    explanation: str
    human_score: int = Field(ge=1, le=5)
    grounded: bool
    quality_note: str
    alt_explanation: str | None = None
    alt_human_score: int | None = Field(default=None, ge=1, le=5)


def load_bug_reports(path: Path = BUG_REPORTS_PATH) -> list[BugReport]:
    """Load and validate the labelled bug reports."""
    return [BugReport.model_validate(x) for x in json.loads(path.read_text(encoding="utf-8"))]


def load_human_judgements(path: Path = HUMAN_JUDGEMENTS_PATH) -> list[HumanJudgement]:
    """Load and validate the human explanation judgements."""
    return [HumanJudgement.model_validate(x) for x in json.loads(path.read_text(encoding="utf-8"))]
