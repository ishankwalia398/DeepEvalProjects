"""Optional DeepEval layer: the deterministic citation check as a DeepEval metric (offline),
plus a GEval groundedness judge that runs only with a real provider."""
import argparse
import json
import os

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, CacheConfig, DisplayConfig
from deepeval.metrics import BaseMetric, GEval
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from deepeval.test_case.llm_test_case import SingleTurnParams

from src import store
from src.groundedness import check_citations
from src.llm import LLM, get_judge_llm
from src.schemas import BugReport, load_bug_reports


class CitationGroundednessMetric(BaseMetric):
    """Share of cited log lines found verbatim in the log; needs no LLM.

    With an empty log, citing nothing is the correct answer and scores 1.
    """

    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        cited, log = test_case.metadata["cited_log_lines"], test_case.retrieval_context[0]
        check = check_citations(test_case.name, log, cited)
        if cited:
            self.score = len(check.valid) / len(cited)
        else:
            self.score = 0.0 if log.strip() else 1.0
        self.success = self.score >= self.threshold
        self.reason = f"invalid citations: {check.invalid}" if check.invalid else (
            "all citations found in log" if cited else "no citations")
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return bool(self.success)

    @property
    def __name__(self) -> str:
        return "Citation Groundedness"


class BenchmarkLLM(DeepEvalBaseLLM):
    """Adapts this project's LLM interface to DeepEval's model interface."""

    def __init__(self, llm: LLM):
        self.llm = llm

    def load_model(self) -> LLM:
        return self.llm

    def generate(self, prompt: str, *args, **kwargs) -> str:
        return self.llm.complete(prompt)

    async def a_generate(self, prompt: str, *args, **kwargs) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return self.llm.name


def geval_groundedness(llm: LLM) -> GEval:
    """GEval judge with the same intent as judge_v2. Requires a real provider."""
    return GEval(
        name="Explanation Groundedness",
        criteria=("Check every claim and quoted log line in the actual output against the retrieval context (the log). "
                  "Penalise invented log lines, unsupported root causes, and vague answers."),
        evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.RETRIEVAL_CONTEXT],
        model=BenchmarkLLM(llm),
    )


def test_cases_for_run(run_id: int, items: list[BugReport], db_path=store.DB_PATH) -> list[LLMTestCase]:
    """One LLMTestCase per parsed prediction of a stored triage run."""
    logs = {i.id: i for i in items}
    with store.connect(db_path) as conn:
        _, results = store.load_triage_run(conn, run_id)
    return [
        LLMTestCase(
            name=r.item_id, input=f"{logs[r.item_id].title}\n{logs[r.item_id].description}",
            actual_output=r.output.explanation, retrieval_context=[logs[r.item_id].log_snippet],
            metadata={"cited_log_lines": r.output.cited_log_lines},
        )
        for r in results if r.output is not None
    ]


def run(run_id: int, use_geval: bool = False, provider: str | None = None) -> dict:
    """Evaluate a stored run with DeepEval; returns pass counts per metric."""
    cases = test_cases_for_run(run_id, load_bug_reports())
    metrics: list[BaseMetric] = [CitationGroundednessMetric()]
    if use_geval:
        llm = get_judge_llm(provider)
        if llm.name == "mock":
            raise SystemExit("GEval needs a real judge provider: set JUDGE_PROVIDER=anthropic|openrouter or pass --provider.")
        metrics.append(geval_groundedness(llm))
    result = evaluate(
        test_cases=cases, metrics=metrics,
        async_config=AsyncConfig(run_async=False),
        display_config=DisplayConfig(print_results=False, show_indicator=False),
        cache_config=CacheConfig(write_cache=False, use_cache=False),
    )
    summary: dict[str, dict] = {}
    for tr in result.test_results:
        for md in tr.metrics_data or []:
            s = summary.setdefault(md.name, {"passed": 0, "failed": [], "total": 0})
            s["total"] += 1
            if md.success:
                s["passed"] += 1
            else:
                s["failed"].append(tr.name)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-check a triage run with DeepEval.")
    parser.add_argument("--run", type=int, required=True)
    parser.add_argument("--geval", action="store_true", help="also run the GEval judge (real provider only)")
    parser.add_argument("--provider", choices=["anthropic", "openrouter"], default=None)
    args = parser.parse_args()
    print(json.dumps(run(args.run, args.geval, args.provider), indent=2))


if __name__ == "__main__":
    main()
