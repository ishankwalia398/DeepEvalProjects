"""SQLite schema and helpers for runs, items, predictions, scores and judge scores."""
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from src.groundedness import check_citations
from src.schemas import REPORTS_DIR, BugReport, JudgeScore, TriageOutput, TriageResult

DB_PATH = REPORTS_DIR / "benchmark.db"
LABEL_COLUMNS = {"severity": "severity", "component": "component", "root_cause_category": "root_cause"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs(
  run_id INTEGER PRIMARY KEY AUTOINCREMENT, prompt_variant TEXT NOT NULL,
  model TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS items(
  item_id TEXT PRIMARY KEY, platform TEXT, gold_severity TEXT,
  gold_component TEXT, gold_root_cause TEXT, tricky INTEGER);
CREATE TABLE IF NOT EXISTS predictions(
  run_id INTEGER, item_id TEXT, severity TEXT, component TEXT, root_cause TEXT,
  explanation TEXT, parse_error TEXT, cited_log_lines TEXT, PRIMARY KEY(run_id, item_id));
CREATE TABLE IF NOT EXISTS scores(
  run_id INTEGER, item_id TEXT, metric TEXT, value REAL, PRIMARY KEY(run_id, item_id, metric));
CREATE TABLE IF NOT EXISTS judge_scores(
  run_id INTEGER, item_id TEXT, judge_version TEXT, score INTEGER, grounded INTEGER,
  PRIMARY KEY(run_id, item_id));
"""


@contextmanager
def connect(path: Path = DB_PATH) -> Iterator[sqlite3.Connection]:
    """Open the database, create tables if needed, commit and close on exit."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def sync_items(conn: sqlite3.Connection, items: list[BugReport]) -> None:
    """Upsert gold labels and platform for every item (needed by the SQL analysis queries)."""
    conn.executemany(
        "INSERT OR REPLACE INTO items VALUES (?,?,?,?,?,?)",
        [(i.id, i.platform, i.gold and i.gold.severity, i.gold and i.gold.component,
          i.gold and i.gold.root_cause_category, int(i.tricky)) for i in items],
    )


def _new_run(conn: sqlite3.Connection, variant: str, model: str) -> int:
    cur = conn.execute(
        "INSERT INTO runs(prompt_variant, model, created_at) VALUES (?,?,?)",
        (variant, model, datetime.now(timezone.utc).isoformat(timespec="seconds")),
    )
    return int(cur.lastrowid)


def save_triage_run(conn: sqlite3.Connection, variant: str, model: str, items: list[BugReport],
                    results: list[TriageResult], judge: dict[str, JudgeScore | None] | None = None) -> int:
    """Store predictions, citation counts, correctness (labelled items only) and optional judge
    scores; return the run id."""
    run_id = _new_run(conn, variant, model)
    by_id = {i.id: i for i in items}
    for r in results:
        o, item = r.output, by_id[r.item_id]
        conn.execute(
            "INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?)",
            (run_id, r.item_id, o and o.severity, o and o.component, o and o.root_cause_category,
             o and o.explanation, r.parse_error, json.dumps(o.cited_log_lines) if o else None),
        )
        rows = [("parse_error", float(o is None))]
        if item.gold is not None:
            for field in LABEL_COLUMNS:
                correct = o is not None and getattr(o, field) == getattr(item.gold, field)
                rows.append((f"{LABEL_COLUMNS[field]}_correct", float(correct)))
        verdict = (judge or {}).get(r.item_id)
        if verdict is not None:
            rows += [("judge_score", float(verdict.score)), ("judge_grounded", float(verdict.grounded))]
        if o is not None:
            check = check_citations(r.item_id, item.log_snippet, o.cited_log_lines)
            rows += [("citations_total", float(len(o.cited_log_lines))), ("citations_invalid", float(len(check.invalid)))]
        conn.executemany("INSERT INTO scores VALUES (?,?,?,?)", [(run_id, r.item_id, m, v) for m, v in rows])
    return run_id


def load_triage_run(conn: sqlite3.Connection, run_id: int) -> tuple[dict, list[TriageResult]]:
    """Return run metadata and reconstructed triage results."""
    meta = dict(conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone() or {})
    if not meta:
        raise ValueError(f"run {run_id} not found")
    results = []
    for row in conn.execute("SELECT * FROM predictions WHERE run_id=? ORDER BY item_id", (run_id,)):
        output = None
        if row["parse_error"] is None:
            output = TriageOutput(
                severity=row["severity"], component=row["component"], root_cause_category=row["root_cause"],
                explanation=row["explanation"], cited_log_lines=json.loads(row["cited_log_lines"]),
            )
        results.append(TriageResult(item_id=row["item_id"], output=output, parse_error=row["parse_error"]))
    return meta, results


PAIRWISE_METRICS = ("pairwise_flip", "pairwise_agree")


def save_judge_run(conn: sqlite3.Connection, version: str, model: str,
                   scores: dict[str, JudgeScore | None], pairwise: dict[str, dict[str, int]]) -> int:
    """Store pointwise judge scores and pairwise flip/agree flags; return the run id."""
    run_id = _new_run(conn, f"judge_{version}", model)
    conn.executemany(
        "INSERT INTO judge_scores VALUES (?,?,?,?,?)",
        [(run_id, item_id, version, s.score, int(s.grounded)) for item_id, s in scores.items() if s],
    )
    conn.executemany(
        "INSERT INTO scores VALUES (?,?,?,?)",
        [(run_id, item_id, m, float(flags[m])) for item_id, flags in pairwise.items() for m in PAIRWISE_METRICS],
    )
    return run_id


def load_judge_run(conn: sqlite3.Connection, run_id: int) -> tuple[dict[str, tuple[int, bool]], dict[str, dict[str, int]]]:
    """Return {item_id: (score, grounded)} and {item_id: {pairwise_flip, pairwise_agree}} for a judge run."""
    scores = {r["item_id"]: (r["score"], bool(r["grounded"]))
              for r in conn.execute("SELECT * FROM judge_scores WHERE run_id=?", (run_id,))}
    pairwise: dict[str, dict[str, int]] = {}
    for r in conn.execute("SELECT * FROM scores WHERE run_id=? AND metric LIKE 'pairwise_%'", (run_id,)):
        pairwise.setdefault(r["item_id"], {})[r["metric"]] = int(r["value"])
    return scores, pairwise


def list_runs(conn: sqlite3.Connection) -> list[dict]:
    """All runs in creation order."""
    return [dict(r) for r in conn.execute("SELECT * FROM runs ORDER BY run_id")]
