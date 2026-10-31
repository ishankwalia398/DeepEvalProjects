"""Build reports/report.md (and PNG charts when matplotlib is available) from the SQLite runs."""
import json
import re
import sqlite3
from pathlib import Path

import pandas as pd

from src import store
from src.calibration import TRUST_THRESHOLD, calibration_metrics
from src.compare import N_BOOT, SEED, compare_runs
from src.groundedness import groundedness_report
from src.metrics import evaluate_run, flipped_items
from src.prompts import JUDGE_VERSIONS, TRIAGE_VARIANTS
from src.schemas import LABEL_FIELDS, REPORTS_DIR, ROOT, SEVERITIES, load_bug_reports, load_human_judgements

QUERIES_PATH = ROOT / "notebooks_or_queries" / "queries.sql"
REPORT_PATH = REPORTS_DIR / "report.md"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"


def md_table(df: pd.DataFrame, index: bool = True) -> str:
    """Render a DataFrame as a GitHub markdown table without extra dependencies."""
    frame = df.rename_axis(df.index.name or "").reset_index() if index else df
    cols = [str(c) for c in frame.columns]

    def cell(v: object) -> str:
        return f"{v:.2f}" if isinstance(v, float) else str(v)

    rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    rows += ["| " + " | ".join(cell(v) for v in r) + " |" for r in frame.itertuples(index=False)]
    return "\n".join(rows)


def latest_runs(conn: sqlite3.Connection) -> dict[str, int]:
    """Most recent run id for each prompt_variant label."""
    return {r["prompt_variant"]: r["run_id"] for r in store.list_runs(conn)}


def load_queries(path: Path = QUERIES_PATH) -> dict[str, str]:
    """Split queries.sql into {name: sql} on '-- name:' markers."""
    blocks = re.split(r"^-- name: (\w+)\s*$", path.read_text(encoding="utf-8"), flags=re.MULTILINE)
    return {blocks[i]: blocks[i + 1].strip() for i in range(1, len(blocks), 2)}


def _charts(variant_acc: pd.DataFrame, calib: pd.DataFrame, confusion: pd.DataFrame, title: str) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    plt.rcParams.update({"font.family": "sans-serif", "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK_2,
                         "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK})
    files = []

    def finish(fig, ax, name: str) -> None:
        ax.set_facecolor(SURFACE)
        fig.patch.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        fig.tight_layout()
        fig.savefig(REPORTS_DIR / name, dpi=150)
        plt.close(fig)
        files.append(name)

    def grouped(ax, frame: pd.DataFrame) -> None:
        width = 0.8 / len(frame.columns)
        for k, col in enumerate(frame.columns):
            xs = [i + (k - (len(frame.columns) - 1) / 2) * width for i in range(len(frame.index))]
            bars = ax.bar(xs, frame[col], width * 0.92, color=SERIES[k], label=col)
            ax.bar_label(bars, fmt="%.2f", fontsize=8, color=INK_2, padding=2)
        ax.set_xticks(range(len(frame.index)), frame.index)
        ax.legend(frameon=False, ncols=len(frame.columns), loc="upper left", bbox_to_anchor=(0, 1.12))

    fig, ax = plt.subplots(figsize=(7, 3.6))
    grouped(ax, variant_acc.T)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Accuracy")
    finish(fig, ax, "accuracy_by_variant.png")

    fig, ax = plt.subplots(figsize=(7, 3.6))
    grouped(ax, calib)
    ax.axhline(TRUST_THRESHOLD, color=MUTED, linestyle="--", linewidth=1)
    ax.text(-0.45, TRUST_THRESHOLD + 0.02, f"trust threshold {TRUST_THRESHOLD}", ha="left", fontsize=8, color=INK_2)
    ax.set_ylim(min(0, calib.min().min()) - 0.05, 1.1)
    ax.set_ylabel("Cohen's kappa vs human")
    finish(fig, ax, "judge_kappa.png")

    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    im = ax.imshow(confusion.to_numpy(), cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
        "blue", ["#cde2fb", "#5598e7", "#0d366b"]))
    ax.set_xticks(range(len(confusion.columns)), [str(c).strip("_").replace("_", " ") for c in confusion.columns],
                  rotation=30, ha="right")
    ax.set_yticks(range(len(confusion.index)), confusion.index)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Gold")
    ax.set_title(title, fontsize=10, color=INK)
    vmax = confusion.to_numpy().max()
    for (r, c), v in pd.DataFrame(confusion.to_numpy()).stack().items():
        ax.text(c, r, int(v), ha="center", va="center", fontsize=9, color="#ffffff" if v > vmax / 2 else INK)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "severity_confusion.png", dpi=150)
    plt.close(fig)
    files.append("severity_confusion.png")
    return files


def build_report(db_path: Path = store.DB_PATH, out: Path = REPORT_PATH) -> Path:
    """Write the markdown report from the latest run of each variant and judge version."""
    items, humans = load_bug_reports(), load_human_judgements()
    with store.connect(db_path) as conn:
        runs = latest_runs(conn)
        variants = [v for v in TRIAGE_VARIANTS if v in runs]
        if not variants:
            raise SystemExit("No triage runs found. Run `python run_all.py` first.")
        results = {v: store.load_triage_run(conn, runs[v]) for v in variants}
        para = {v: store.load_triage_run(conn, runs[f"{v}+paraphrase"])[1]
                for v in variants if f"{v}+paraphrase" in runs}
        judges = {j: store.load_judge_run(conn, runs[f"judge_{j}"]) for j in JUDGE_VERSIONS if f"judge_{j}" in runs}
        comparisons = [compare_runs(conn, runs[a], runs[b], items)
                       for a, b in (("v1_basic", "v3_cot_light"), ("v2_structured", "v3_cot_light"))
                       if a in runs and b in runs]
        params = {"run_a": runs.get("v2_structured"), "run_b": runs.get("v3_cot_light")}
        sql = {name: pd.read_sql_query(q, conn, params=params) for name, q in load_queries().items()}

    metrics = {v: evaluate_run(items, results[v][1]) for v in variants}
    ground = {v: groundedness_report(items, results[v][1]) for v in variants}
    L = ["# Bug Triage Benchmark – Report", ""]

    models = sorted({results[v][0]["model"] for v in variants})
    L += ["## Overview", "",
          f"- Items: {len(items)} synthetic bug reports ({sum(i.tricky for i in items)} tricky); "
          f"{len(humans)} human-judged explanations ({sum(1 for h in humans if h.alt_explanation)} pairwise pairs).",
          f"- Triage runs: {', '.join(f'{v} = run {runs[v]}' for v in variants)}. Model(s): {', '.join(models)}.",
          "- Mock numbers come from a scripted, deterministic simulator. They demonstrate the harness, "
          "not the quality of any real model.", ""]

    acc = pd.DataFrame({v: {f: metrics[v]["fields"][f]["accuracy"] for f in LABEL_FIELDS} for v in variants}).T
    table = pd.DataFrame({v: {
        **{f"{f} acc": metrics[v]["fields"][f]["accuracy"] for f in LABEL_FIELDS},
        **{f"{f} macro-F1": metrics[v]["fields"][f]["macro_f1"] for f in LABEL_FIELDS},
        "exact match": metrics[v]["exact_match"],
        "parse error rate": metrics[v]["parse_error_rate"],
        "severity misses": len(metrics[v]["severity_misses"]),
        "hallucinated citation rate": ground[v]["hallucinated_citation_rate"],
    } for v in variants}, dtype=object)
    tricky = pd.DataFrame({v: {f"{f} ({k})": metrics[v]["tricky_split"][f][k]
                               for f in LABEL_FIELDS for k in ("tricky", "non_tricky")} for v in variants})
    L += ["## Prompt-variant comparison", "", md_table(table), "", "Accuracy on tricky vs non-tricky items:", "",
          md_table(tricky), ""]

    flips = flipped_items({v: results[v][1] for v in variants})
    spread = (acc.max() - acc.min()).rename("max - min accuracy")
    L += ["## Prompt sensitivity", "", md_table(spread.to_frame()), "",
          f"{len(flips)} item/field predictions differ between variants "
          f"({len({f['item_id'] for f in flips})} distinct items). First 15:", "",
          md_table(pd.DataFrame(flips[:15]), index=False) if flips else "None.", ""]
    if para:
        changes = {v: len(flipped_items({"original": results[v][1], "paraphrased": para[v]})) for v in para}
        L += ["Paraphrase test (sentence rotation, lowercase, doubled spaces): predictions that changed "
              f"out of {len(items) * len(LABEL_FIELDS)} item/field labels:", "",
              md_table(pd.Series(changes, name="changed labels").to_frame()), ""]

    L += ["## Severity confusion matrices", ""]
    for v in variants:
        L += [f"**{v}** (rows = gold, columns = predicted)", "", md_table(metrics[v]["fields"]["severity"]["confusion"]), ""]

    L += ["## Severity misses (critical/major predicted as minor/trivial)", ""]
    misses = [{"variant": v, **m} for v in variants for m in metrics[v]["severity_misses"]]
    L += [md_table(pd.DataFrame(misses), index=False) if misses else "None.", ""]

    L += ["## Groundedness (deterministic citation check)", "",
          md_table(pd.DataFrame({v: {"citations": ground[v]["citations_total"],
                                     "invalid": ground[v]["citations_invalid"],
                                     "hallucinated rate": ground[v]["hallucinated_citation_rate"],
                                     "zero valid citations": ground[v]["zero_valid_fraction"]}
                                 for v in variants}, dtype=object).T), "", "Worst offenders:", ""]
    offenders = [{"variant": v, "item_id": w["item_id"], "invalid citation": " / ".join(w["invalid"])}
                 for v in variants for w in ground[v]["worst_offenders"]]
    L += [md_table(pd.DataFrame(offenders), index=False) if offenders else "None.", ""]

    calib = {j: calibration_metrics(humans, *judges[j]) for j in judges}
    if calib:
        ctab = pd.DataFrame({f"judge_{j}": {
            "kappa (grounded)": c["kappa_grounded"], "weighted kappa (score)": c["weighted_kappa_score"],
            "exact score agreement": c["exact_score_agreement"], "length Spearman": c["length_spearman"],
            "length partial Spearman": c["length_partial_spearman"],
            "position flip rate": "n/a" if c["position_flip_rate"] is None else c["position_flip_rate"],
            "pairwise agreement": "n/a" if c["pairwise_agreement"] is None else c["pairwise_agreement"],
            "trusted": c["trusted"]} for j, c in calib.items()})
        L += ["## Judge calibration against human labels", "",
              f"**Rule:** the judge is trusted for decisions only if weighted kappa >= {TRUST_THRESHOLD} on this set; "
              "otherwise use deterministic checks and human review.", "", md_table(ctab), ""]
        if "v1" in calib and "v2" in calib:
            a, b = calib["v1"], calib["v2"]
            L += [f"**Lesson:** weighted kappa moved from {a['weighted_kappa_score']:.2f} (judge_v1) to "
                  f"{b['weighted_kappa_score']:.2f} (judge_v2). The partial length correlation (length vs judge score, "
                  f"controlling for human score) moved from {a['length_partial_spearman']:.2f} to "
                  f"{b['length_partial_spearman']:.2f}, and the position flip rate from {a['position_flip_rate']:.2f} "
                  f"to {b['position_flip_rate']:.2f}. judge_v1 trusted: {a['trusted']}; judge_v2 trusted: {b['trusted']}.", ""]
        if "v2" in calib and "v2_reworded" in calib:
            L += [f"Judge prompt sensitivity: rewording the judge_v2 rubric changes weighted kappa from "
                  f"{calib['v2']['weighted_kappa_score']:.2f} to {calib['v2_reworded']['weighted_kappa_score']:.2f}.", ""]
        for j in ("v1", "v2"):
            if j in calib:
                L += [f"judge_{j} vs human score (rows = human, columns = judge):", "", md_table(calib[j]["confusion"]), ""]
        if "v2" in calib:
            dis = calib["v2"]["disagreements"]
            L += ["judge_v2 disagreements (score off by >= 2 or grounded flag differs):", "",
                  md_table(pd.DataFrame(dis), index=False) if dis else "None.", ""]

    if comparisons:
        L += ["## Version comparison", ""]
        for c in comparisons:
            low, high = c["ci95"]
            L += [f"**A = {c['a']['prompt_variant']} (run {c['a']['run_id']}) vs B = {c['b']['prompt_variant']} "
                  f"(run {c['b']['run_id']})**", "",
                  md_table(pd.DataFrame({k: {"A": va, "B": vb, "delta": vb - va} for k, (va, vb) in c["deltas"].items()},
                                        dtype=object).T),
                  "", f"- Regressed (exact match lost): {[r['item_id'] for r in c['regressed']] or 'none'}",
                  f"- Improved: {[r['item_id'] for r in c['improved']] or 'none'}",
                  f"- New severity misses in B: {c['new_severity_misses'] or 'none'}",
                  f"- Exact-match difference {c['exact_match_diff']:+.3f}, 95% paired bootstrap CI "
                  f"[{low:+.3f}, {high:+.3f}] ({N_BOOT} resamples, seed {SEED})",
                  f"- **Recommendation: {c['recommendation']}**", ""]

    titles = {"error_rate_by_severity": "1. Error rate by gold severity and prompt variant",
              "regressed_items": f"2. Labels correct in run {params['run_a']} (v2_structured) and wrong in run {params['run_b']} (v3_cot_light)",
              "hallucination_by_platform": "3. Hallucinated-citation rate by platform"}
    L += ["## SQL analysis (`notebooks_or_queries/queries.sql`)", ""]
    for name, df in sql.items():
        L += [f"**{titles.get(name, name)}**", "", md_table(df, index=False) if len(df) else "No rows.", ""]

    deepeval_file = REPORTS_DIR / "deepeval_summary.json"
    if deepeval_file.exists():
        summary = json.loads(deepeval_file.read_text(encoding="utf-8"))
        L += ["## DeepEval cross-check (v3_cot_light)", "",
              md_table(pd.DataFrame({k: {"passed": v["passed"], "total": v["total"], "failed items": ", ".join(v["failed"])}
                                     for k, v in summary.items()}, dtype=object).T), "",
              "The same deterministic citation check, run through DeepEval's `evaluate` as a custom metric. "
              "The GEval judge is added only with `--geval` and a real provider.", ""]

    best = max(variants, key=lambda v: metrics[v]["exact_match"])
    charts = _charts(acc, pd.DataFrame({f"judge_{j}": {"grounded kappa": c["kappa_grounded"],
                                                       "weighted kappa": c["weighted_kappa_score"]}
                                        for j, c in calib.items()}).T if calib else pd.DataFrame(),
                     metrics[best]["fields"]["severity"]["confusion"], f"Severity confusion – {best}")
    if charts:
        L += ["## Charts", ""] + [f"![{c}]({c})" for c in charts] + [""]

    L += ["## What I would do next", "",
          "- Replace the mock with real models (Anthropic, OpenRouter) and repeat each run several times to measure run-to-run variance.",
          "- Have a second human label the 30 explanations and report inter-annotator kappa as the ceiling for the judge.",
          "- Grow the set beyond 50 items so the bootstrap intervals can separate close variants.",
          "- Add a regression gate in CI: fail when severity misses increase or judge kappa drops below the threshold.",
          "", "## Limitations", "",
          "- 50 synthetic items and 30 human scores drafted for review: small samples, wide intervals.",
          "- The mock model and mock judge are scripted simulators; the judge_v2 heuristic agrees with humans partly by construction.",
          "- One annotator; no inter-annotator agreement yet.", ""]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L), encoding="utf-8")
    return out


def main() -> None:
    print(f"report written to {build_report()}")


if __name__ == "__main__":
    main()
