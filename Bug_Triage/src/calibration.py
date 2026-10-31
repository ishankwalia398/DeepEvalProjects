"""Calibrate the LLM judge against human labels: kappa, confusion, position bias, length bias."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from src import store
from src.judge import judge_explanation, judge_pairwise
from src.llm import LLM, get_judge_llm
from src.prompts import JUDGE_VERSIONS
from src.schemas import HumanJudgement, JudgeScore, load_bug_reports, load_human_judgements

SCORES = [1, 2, 3, 4, 5]
TRUST_THRESHOLD = 0.6
PAIRWISE_VERSIONS = ("v1", "v2")


def kappa_binary(human: list[bool], judge: list[bool]) -> float:
    """Cohen's kappa for the binary grounded flag."""
    return float(cohen_kappa_score(human, judge, labels=[False, True]))


def kappa_weighted(human: list[int], judge: list[int]) -> float:
    """Quadratic-weighted Cohen's kappa for 1-5 scores."""
    return float(cohen_kappa_score(human, judge, labels=SCORES, weights="quadratic"))


def score_confusion(human: list[int], judge: list[int]) -> pd.DataFrame:
    """Rows = human score, columns = judge score."""
    cm = pd.crosstab(pd.Series(human, name="human"), pd.Series(judge, name="judge"))
    return cm.reindex(index=SCORES, columns=SCORES, fill_value=0)


def partial_spearman(x: list[float], y: list[float], control: list[float]) -> float:
    """Spearman correlation of x and y after removing the linear effect of control (all on ranks)."""
    rx, ry, rz = (pd.Series(v).rank().to_numpy() for v in (x, y, control))

    def residual(v: np.ndarray) -> np.ndarray:
        slope, intercept = np.polyfit(rz, v, 1)
        return v - (slope * rz + intercept)

    ex, ey = residual(rx), residual(ry)
    if np.allclose(ex, 0) or np.allclose(ey, 0):
        return 0.0
    return float(np.corrcoef(ex, ey)[0, 1])


def position_outcome(first: str | None, second: str | None) -> dict[str, int]:
    """Compare a pair judged as (reference, alternative) and then as (alternative, reference).

    flip = the preferred explanation changed with order; agree = both orders preferred the reference,
    which humans scored higher.
    """
    w1 = {"A": "ref", "B": "alt"}.get(first or "", "tie")
    w2 = {"A": "alt", "B": "ref"}.get(second or "", "tie")
    return {"pairwise_flip": int(w1 != w2 or first is None or second is None),
            "pairwise_agree": int(w1 == w2 == "ref")}


def run_judge(llm: LLM, version: str, humans: list[HumanJudgement]) -> tuple[dict[str, JudgeScore | None], dict]:
    """Score every reference explanation and, for v1/v2, every pair in both orders."""
    logs = {i.id: i.log_snippet for i in load_bug_reports()}
    scores = {h.item_id: judge_explanation(llm, version, logs[h.item_id], h.explanation) for h in humans}
    pairwise = {}
    if version in PAIRWISE_VERSIONS:
        for h in (h for h in humans if h.alt_explanation):
            log = logs[h.item_id]
            pairwise[h.item_id] = position_outcome(
                judge_pairwise(llm, version, log, h.explanation, h.alt_explanation),
                judge_pairwise(llm, version, log, h.alt_explanation, h.explanation),
            )
    return scores, pairwise


def calibration_metrics(humans: list[HumanJudgement], scores: dict[str, tuple[int, bool]],
                        pairwise: dict[str, dict[str, int]]) -> dict:
    """Agreement and bias statistics of one judge version against the human labels."""
    rated = [h for h in humans if h.item_id in scores]
    hs, js = [h.human_score for h in rated], [scores[h.item_id][0] for h in rated]
    hg, jg = [h.grounded for h in rated], [scores[h.item_id][1] for h in rated]
    lengths = [len(h.explanation) for h in rated]
    weighted = kappa_weighted(hs, js)
    return {
        "n": len(rated),
        "unparsed": len(humans) - len(rated),
        "kappa_grounded": kappa_binary(hg, jg),
        "weighted_kappa_score": weighted,
        "exact_score_agreement": sum(a == b for a, b in zip(hs, js)) / len(rated),
        "confusion": score_confusion(hs, js),
        "length_spearman": float(pd.Series(lengths).corr(pd.Series(js), method="spearman")),
        "length_partial_spearman": partial_spearman(lengths, js, hs),
        "position_flip_rate": (sum(p["pairwise_flip"] for p in pairwise.values()) / len(pairwise)) if pairwise else None,
        "pairwise_agreement": (sum(p["pairwise_agree"] for p in pairwise.values()) / len(pairwise)) if pairwise else None,
        "pairs": len(pairwise),
        "trusted": weighted >= TRUST_THRESHOLD,
        "disagreements": [
            {"item_id": h.item_id, "human": h.human_score, "judge": scores[h.item_id][0],
             "human_grounded": h.grounded, "judge_grounded": scores[h.item_id][1], "note": h.quality_note}
            for h in rated if abs(h.human_score - scores[h.item_id][0]) >= 2 or h.grounded != scores[h.item_id][1]
        ],
    }


def execute(version: str, provider: str | None = None, db_path: Path = store.DB_PATH) -> tuple[int, dict]:
    """Run one judge version with the judge model (JUDGE_PROVIDER / JUDGE_MODEL unless `provider` is given),
    store it, and return (run_id, metrics)."""
    humans = load_human_judgements()
    llm = get_judge_llm(provider)
    raw_scores, pairwise = run_judge(llm, version, humans)
    with store.connect(db_path) as conn:
        run_id = store.save_judge_run(conn, version, llm.name, raw_scores, pairwise)
    scores = {k: (v.score, v.grounded) for k, v in raw_scores.items() if v}
    return run_id, calibration_metrics(humans, scores, pairwise)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate a judge version against human labels.")
    parser.add_argument("--judge", choices=JUDGE_VERSIONS, required=True)
    parser.add_argument("--provider", choices=["mock", "anthropic", "openrouter"], default=None)
    args = parser.parse_args()
    run_id, m = execute(args.judge, args.provider)
    with store.connect() as conn:
        model = next(r["model"] for r in store.list_runs(conn) if r["run_id"] == run_id)
    print(f"run {run_id}: judge_{args.judge} with {model} on {m['n']} items ({m['unparsed']} unparsed)")
    print(f"  kappa(grounded)={m['kappa_grounded']:.2f} weighted_kappa(score)={m['weighted_kappa_score']:.2f} "
          f"exact_agreement={m['exact_score_agreement']:.2f}")
    print(f"  length spearman={m['length_spearman']:.2f} partial (controlling human score)={m['length_partial_spearman']:.2f}")
    if m["pairs"]:
        print(f"  position flip rate={m['position_flip_rate']:.2f} pairwise agreement={m['pairwise_agreement']:.2f} ({m['pairs']} pairs)")
    print(f"  trusted for decisions (weighted kappa >= {TRUST_THRESHOLD}): {m['trusted']}")


if __name__ == "__main__":
    main()
