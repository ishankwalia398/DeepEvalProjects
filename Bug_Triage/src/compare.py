"""Compare two triage runs: metric deltas, regressions, bootstrap CI and a plain recommendation."""
import argparse
import sqlite3
from pathlib import Path

import numpy as np

from src import store
from src.groundedness import groundedness_report
from src.metrics import evaluate_run, exact_match
from src.schemas import LABEL_FIELDS, BugReport, TriageResult, load_bug_reports

N_BOOT = 2000
SEED = 42


def bootstrap_diff_ci(a: np.ndarray, b: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> tuple[float, float]:
    """Paired bootstrap 95% CI of mean(b) - mean(a), resampling items."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    diffs = b[idx].mean(axis=1) - a[idx].mean(axis=1)
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def _changed_fields(item: BugReport, ra: TriageResult, rb: TriageResult) -> list[str]:
    def ok(r: TriageResult, f: str) -> bool:
        return r.output is not None and getattr(r.output, f) == getattr(item.gold, f)
    return [f for f in LABEL_FIELDS if ok(ra, f) != ok(rb, f)]


def recommend(ci: tuple[float, float], new_misses: list[str], n: int) -> str:
    """ship B only if the CI excludes zero in B's favour and B adds no severity misses."""
    low, high = ci
    if low > 0 and not new_misses:
        return "ship B"
    if high < 0:
        return "keep A"
    if low > 0:
        return f"inconclusive: B is more accurate but adds severity misses {new_misses}"
    return f"inconclusive: sample too small (n={n}), the 95% CI includes 0"


def compare_runs(conn: sqlite3.Connection, run_a: int, run_b: int, items: list[BugReport]) -> dict:
    """Everything needed to decide between run A and run B."""
    meta_a, res_a = store.load_triage_run(conn, run_a)
    meta_b, res_b = store.load_triage_run(conn, run_b)
    ma, mb = evaluate_run(items, res_a), evaluate_run(items, res_b)
    ga, gb = groundedness_report(items, res_a), groundedness_report(items, res_b)
    by_a, by_b = {r.item_id: r for r in res_a}, {r.item_id: r for r in res_b}
    em_a = np.array([exact_match(i, by_a[i.id]) for i in items], dtype=float)
    em_b = np.array([exact_match(i, by_b[i.id]) for i in items], dtype=float)

    deltas = {f"{f}_accuracy": (ma["fields"][f]["accuracy"], mb["fields"][f]["accuracy"]) for f in LABEL_FIELDS}
    deltas |= {
        "exact_match": (ma["exact_match"], mb["exact_match"]),
        "parse_error_rate": (ma["parse_error_rate"], mb["parse_error_rate"]),
        "severity_misses": (len(ma["severity_misses"]), len(mb["severity_misses"])),
        "hallucinated_citation_rate": (ga["hallucinated_citation_rate"], gb["hallucinated_citation_rate"]),
    }
    misses_a = {m["item_id"] for m in ma["severity_misses"]}
    new_misses = sorted(m["item_id"] for m in mb["severity_misses"] if m["item_id"] not in misses_a)
    regressed = [{"item_id": i.id, "fields": _changed_fields(i, by_a[i.id], by_b[i.id])}
                 for k, i in enumerate(items) if em_a[k] and not em_b[k]]
    improved = [{"item_id": i.id, "fields": _changed_fields(i, by_a[i.id], by_b[i.id])}
                for k, i in enumerate(items) if em_b[k] and not em_a[k]]
    ci = bootstrap_diff_ci(em_a, em_b)
    return {
        "a": meta_a, "b": meta_b, "deltas": deltas, "regressed": regressed, "improved": improved,
        "new_severity_misses": new_misses, "exact_match_diff": float(em_b.mean() - em_a.mean()),
        "ci95": ci, "recommendation": recommend(ci, new_misses, len(items)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two triage runs.")
    parser.add_argument("--a", type=int, required=True, help="baseline run id")
    parser.add_argument("--b", type=int, required=True, help="candidate run id")
    parser.add_argument("--db", type=Path, default=store.DB_PATH)
    args = parser.parse_args()
    with store.connect(args.db) as conn:
        c = compare_runs(conn, args.a, args.b, load_bug_reports())
    print(f"A = run {args.a} ({c['a']['prompt_variant']}, {c['a']['model']})")
    print(f"B = run {args.b} ({c['b']['prompt_variant']}, {c['b']['model']})")
    for name, (va, vb) in c["deltas"].items():
        print(f"  {name:<28} A={va:<8.3g} B={vb:<8.3g} delta={vb - va:+.3g}")
    print(f"  regressed items: {[r['item_id'] for r in c['regressed']] or 'none'}")
    print(f"  improved items:  {[r['item_id'] for r in c['improved']] or 'none'}")
    print(f"  new severity misses in B: {c['new_severity_misses'] or 'none'}")
    low, high = c["ci95"]
    print(f"  exact-match diff (B-A) = {c['exact_match_diff']:+.3f}, 95% bootstrap CI [{low:+.3f}, {high:+.3f}] "
          f"({N_BOOT} resamples, seed {SEED})")
    print(f"Recommendation: {c['recommendation']}")


if __name__ == "__main__":
    main()
