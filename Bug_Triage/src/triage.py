"""Run a triage prompt variant over bug reports from any source, parse outputs strictly, and store the run."""
import argparse
import re
from pathlib import Path

from pydantic import ValidationError

from src import store
from src.groundedness import check_citations, groundedness_report
from src.judge import score_explanation
from src.llm import LLM, LLMError, get_judge_llm, get_llm, strip_code_fence
from src.metrics import evaluate_run
from src.prompts import TRIAGE_VARIANTS, triage_prompt
from src.schemas import BugReport, JudgeScore, TriageOutput, TriageResult
from src.sources import SOURCES, SourceError, get_source

JUDGE_VERSION = "v2"


def parse_triage(raw: str) -> tuple[TriageOutput | None, str | None]:
    """Validate the reply against TriageOutput; a single surrounding ```json fence is tolerated."""
    try:
        return TriageOutput.model_validate_json(strip_code_fence(raw)), None
    except ValidationError as e:
        return None, f"invalid output: {e.errors()[0]['msg']}"


def paraphrase(text: str) -> str:
    """Deterministic rewording: rotate sentence order, lowercase, and double the spaces."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if len(sentences) > 1:
        sentences = sentences[1:] + sentences[:1]
    return "  ".join(sentences).lower()


def run_triage(llm: LLM, items: list[BugReport], variant: str, paraphrased: bool = False) -> list[TriageResult]:
    """Triage every item; provider and parse failures are recorded, never raised."""
    results = []
    for item in items:
        prompt = triage_prompt(variant, item, paraphrase(item.description) if paraphrased else None)
        try:
            raw = llm.complete(prompt)
        except LLMError as e:
            results.append(TriageResult(item_id=item.id, output=None, parse_error=f"llm error: {e}"))
            continue
        output, error = parse_triage(raw)
        results.append(TriageResult(item_id=item.id, output=output, parse_error=error, raw=raw))
    return results


def judge_results(judge: LLM, items: list[BugReport], results: list[TriageResult],
                  ) -> tuple[dict[str, JudgeScore | None], dict[str, str]]:
    """Score each parsed explanation with judge_v2; returns (scores, failure reasons).

    Tracker tickets also give the judge their description; the labelled benchmark does not.
    """
    by_id = {i.id: i for i in items}
    scores, errors = {}, {}
    for r in results:
        if r.output is None:
            continue
        item = by_id[r.item_id]
        description = item.description if item.source != "file" else None
        scores[r.item_id], error = score_explanation(
            judge, JUDGE_VERSION, item.log_snippet, r.output.explanation, description)
        if error:
            errors[r.item_id] = error
    return scores, errors


def execute(variant: str, provider: str | None = None, paraphrased: bool = False,
            db_path: Path = store.DB_PATH, source: str = "file", ids: list[str] | None = None,
            path: str | None = None, judge_provider: str | None = None, use_judge: bool = False,
            ) -> tuple[int, list[BugReport], list[TriageResult], dict[str, JudgeScore | None], dict[str, str]]:
    """Fetch reports, run one variant, optionally judge, and persist;
    returns (run_id, items, results, judge scores, judge failure reasons)."""
    items = get_source(source, path).fetch(ids)
    llm = get_llm(provider)
    if llm.name == "mock" and any(i.gold is None for i in items):
        raise SourceError("MockLLM can only triage the labelled synthetic dataset; pass --provider anthropic|openrouter")
    results = run_triage(llm, items, variant, paraphrased)
    verdicts: dict[str, JudgeScore | None] = {}
    judge_errors: dict[str, str] = {}
    model = llm.name
    if use_judge:
        judge = get_judge_llm(judge_provider)
        verdicts, judge_errors = judge_results(judge, items, results)
        model += f" | judge: {judge.name}"
    label = variant + ("+paraphrase" if paraphrased else "") + ("" if source == "file" and not path else f"@{source}")
    with store.connect(db_path) as conn:
        store.sync_items(conn, items)
        run_id = store.save_triage_run(conn, label, model, items, results, verdicts)
    return run_id, items, results, verdicts, judge_errors


def _print_tickets(items: list[BugReport], results: list[TriageResult], verdicts: dict[str, JudgeScore | None],
                   judge_errors: dict[str, str]) -> None:
    by_id = {i.id: i for i in items}
    for r in results:
        item = by_id[r.item_id]
        print(f"\n{item.id}  {item.title}" + (f"  ({item.url})" if item.url else ""))
        if r.output is None:
            print(f"  ERROR: {r.parse_error}")
            continue
        o = r.output
        check = check_citations(item.id, item.log_snippet, o.cited_log_lines)
        print(f"  severity={o.severity} component={o.component} root_cause={o.root_cause_category}")
        if item.gold:
            g = item.gold
            print(f"  gold:    severity={g.severity} component={g.component} root_cause={g.root_cause_category}")
        print(f"  explanation: {o.explanation}")
        print(f"  citations: {len(check.valid)} valid, {len(check.invalid)} not found in log {check.invalid or ''}")
        verdict = verdicts.get(item.id)
        if verdict:
            print(f"  judge_v2: score={verdict.score} grounded={verdict.grounded} ({verdict.reason})")
        elif item.id in judge_errors:
            print(f"  judge_v2: no score, {judge_errors[item.id]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one triage prompt variant on a dataset file or tracker tickets.")
    parser.add_argument("--prompt", choices=TRIAGE_VARIANTS, default="v3_cot_light")
    parser.add_argument("--provider", choices=["mock", "anthropic", "openrouter"], default=None)
    parser.add_argument("--source", choices=sorted(SOURCES), default="file")
    parser.add_argument("--ids", nargs="+", help="specific ids, e.g. BUG-012 or a Jira key PROJ-123")
    parser.add_argument("--path", help="JSON file for --source file (default data/bug_reports.json)")
    parser.add_argument("--paraphrase", action="store_true", help="reword descriptions deterministically")
    parser.add_argument("--judge", action="store_true", help="also score each explanation with judge_v2")
    parser.add_argument("--judge-provider", choices=["mock", "anthropic", "openrouter"], default=None)
    args = parser.parse_args()
    try:
        run_id, items, results, verdicts, judge_errors = execute(
            args.prompt, args.provider, args.paraphrase, source=args.source, ids=args.ids,
            path=args.path, judge_provider=args.judge_provider, use_judge=args.judge)
    except (SourceError, LLMError) as e:
        raise SystemExit(f"error: {e}")
    print(f"run {run_id}: {args.prompt}{' +paraphrase' if args.paraphrase else ''} on {len(items)} report(s) from {args.source}")
    if len(items) <= 10:
        _print_tickets(items, results, verdicts, judge_errors)
    elif judge_errors:
        print(f"judge_v2 failed on {len(judge_errors)} item(s): {judge_errors}")
    m, g = evaluate_run(items, results), groundedness_report(items, results)
    print(f"\nhallucinated_citation_rate={g['hallucinated_citation_rate']:.2f} "
          f"parse/provider errors={sum(r.output is None for r in results)}")
    if m is None:
        print("No gold labels on these reports, so accuracy is not computed "
              "(add labels like severity:major, component:cdn, root_cause:network_timeout to score them).")
        return
    for field, fm in m["fields"].items():
        print(f"  {field:<20} accuracy={fm['accuracy']:.2f} macro_f1={fm['macro_f1']:.2f}")
    print(f"  exact_match={m['exact_match']:.2f} parse_error_rate={m['parse_error_rate']:.2f} "
          f"severity_misses={len(m['severity_misses'])}")


if __name__ == "__main__":
    main()
