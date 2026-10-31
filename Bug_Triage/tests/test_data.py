"""Validate the synthetic dataset and the human judgements."""
import re
from collections import Counter

from src.schemas import load_bug_reports, load_human_judgements

QUOTE = re.compile(r'"([^"]+)"')


def _norm(text: str) -> str:
    return " ".join(text.split())


def test_bug_reports_schema_and_size():
    items = load_bug_reports()
    assert len(items) == 50
    assert len({i.id for i in items}) == 50
    assert all(i.gold is not None and i.rationale and i.platform != "unknown" for i in items)


def test_label_balance_and_tricky_coverage():
    items = load_bug_reports()
    for field in ("severity", "component"):
        counts = Counter(getattr(i.gold, field) for i in items)
        assert max(counts.values()) - min(counts.values()) <= 4, counts
    tricky = [i for i in items if i.tricky]
    assert len(tricky) >= 12
    assert all(i.tricky_reason for i in tricky)
    assert sum("ignore previous instructions" in i.log_snippet for i in items) == 2
    assert any(i.log_snippet == "" and i.gold.root_cause_category == "unknown" for i in items)


def test_evidence_bearing_logs_for_known_causes():
    for item in load_bug_reports():
        has_signal = any(lvl in item.log_snippet for lvl in (" ERROR ", " WARN "))
        if item.gold.root_cause_category == "unknown":
            assert not has_signal, item.id
        else:
            assert has_signal, item.id


def test_human_judgements_schema_and_bad_examples():
    humans = load_human_judgements()
    ids = {i.id for i in load_bug_reports()}
    assert len(humans) == 30
    assert all(h.item_id in ids for h in humans)
    assert sum(h.quality_note.startswith("bad") for h in humans) == 8
    pairs = [h for h in humans if h.alt_explanation]
    assert len(pairs) == 15 and all(h.alt_human_score is not None for h in pairs)


def test_grounded_explanations_quote_only_real_log_text():
    logs = {i.id: _norm(i.log_snippet) for i in load_bug_reports()}
    for h in load_human_judgements():
        quotes = QUOTE.findall(h.explanation)
        if not h.grounded:
            continue
        if logs[h.item_id] == "":
            assert not quotes and "Insufficient information to determine" in h.explanation
        else:
            assert quotes, h.item_id
            assert all(_norm(q) in logs[h.item_id] for q in quotes), h.item_id
