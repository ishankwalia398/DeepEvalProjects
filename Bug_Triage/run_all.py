"""Run the whole benchmark (offline with the mock by default) and write reports/report.md."""
import argparse
import json
import time

from src import calibration, store, triage
from src.compare import compare_runs
from src.prompts import JUDGE_VERSIONS, TRIAGE_VARIANTS
from src.report import build_report
from src.schemas import REPORTS_DIR, load_bug_reports

DEEPEVAL_SUMMARY = REPORTS_DIR / "deepeval_summary.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run every benchmark step and build the report.")
    parser.add_argument("--provider", choices=["mock", "anthropic", "openrouter"], default="mock",
                        help="triage model provider")
    parser.add_argument("--judge-provider", choices=["mock", "anthropic", "openrouter"], default=None,
                        help="judge provider (default: mock when --provider is mock, else JUDGE_PROVIDER from .env)")
    args = parser.parse_args()
    judge_provider = args.judge_provider or ("mock" if args.provider == "mock" else None)
    start = time.perf_counter()
    store.DB_PATH.unlink(missing_ok=True)
    DEEPEVAL_SUMMARY.unlink(missing_ok=True)

    run_ids = {}
    for paraphrased in (False, True):
        for variant in TRIAGE_VARIANTS:
            run_id, *_ = triage.execute(variant, args.provider, paraphrased)
            run_ids[(variant, paraphrased)] = run_id
            print(f"triage  run {run_id}: {variant}{' +paraphrase' if paraphrased else ''}")
    for version in JUDGE_VERSIONS:
        run_id, m = calibration.execute(version, judge_provider)
        print(f"judge   run {run_id}: judge_{version} weighted kappa={m['weighted_kappa_score']:.2f}")

    with store.connect() as conn:
        c = compare_runs(conn, run_ids[("v1_basic", False)], run_ids[("v3_cot_light", False)], load_bug_reports())
    print(f"compare v1_basic vs v3_cot_light: {c['recommendation']}")

    try:
        from src import deepeval_bridge
    except ImportError:
        print("deepeval not installed: skipping the DeepEval cross-check")
    else:
        try:
            summary = deepeval_bridge.run(run_ids[("v3_cot_light", False)])
        except Exception as e:  # optional layer: report the failure, keep the core benchmark
            print(f"WARNING deepeval cross-check failed and was skipped: {type(e).__name__}: {e}")
        else:
            DEEPEVAL_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(f"deepeval cross-check on v3_cot_light: {({k: f'{v['passed']}/{v['total']}' for k, v in summary.items()})}")

    print(f"report  {build_report()}  ({time.perf_counter() - start:.1f}s)")


if __name__ == "__main__":
    main()
