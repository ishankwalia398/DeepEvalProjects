"""LLM judge: pointwise explanation scores and pairwise comparisons."""
from pydantic import ValidationError

from src.llm import LLM, LLMError, strip_code_fence
from src.prompts import judge_prompt, pairwise_prompt
from src.schemas import JudgeScore, PairwiseVerdict


def score_explanation(llm: LLM, version: str, log: str, explanation: str,
                      description: str | None = None) -> tuple[JudgeScore | None, str | None]:
    """Score one explanation 1-5 with a grounded flag; returns (score, None) or (None, why it failed)."""
    try:
        raw = llm.complete(judge_prompt(version, log, explanation, description))
    except LLMError as e:
        return None, f"llm error: {e}"
    try:
        return JudgeScore.model_validate_json(strip_code_fence(raw)), None
    except ValidationError as e:
        return None, f"invalid judge output: {e.errors()[0]['msg']} (reply starts: {raw[:120]!r})"


def judge_explanation(llm: LLM, version: str, log: str, explanation: str) -> JudgeScore | None:
    """Score one explanation 1-5 with a grounded flag; None if the call or the parse fails."""
    return score_explanation(llm, version, log, explanation)[0]


def judge_pairwise(llm: LLM, version: str, log: str, a: str, b: str) -> str | None:
    """Return 'A', 'B' or 'tie'; None if the call or the parse fails."""
    try:
        raw = llm.complete(pairwise_prompt(version, log, a, b))
        return PairwiseVerdict.model_validate_json(strip_code_fence(raw)).winner
    except (LLMError, ValidationError):
        return None
