"""Classification metrics for triage runs: accuracy, per-class P/R/F1, confusion matrix, severity misses."""
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from src.schemas import LABEL_FIELDS, BugReport, TriageResult

PARSE_ERROR = "__parse_error__"
COSTLY_GOLD = {"critical", "major"}
COSTLY_PRED = {"minor", "trivial"}


def field_metrics(gold: list[str], pred: list[str | None], labels: tuple[str, ...]) -> dict:
    """Accuracy, per-class precision/recall/F1, macro-F1 (over classes present in gold) and confusion matrix.

    A missing prediction (parse error) counts as wrong.
    """
    y_pred = [p if p is not None else PARSE_ERROR for p in pred]
    present = [lab for lab in labels if lab in set(gold)]
    p, r, f, s = precision_recall_fscore_support(gold, y_pred, labels=present, zero_division=0)
    cm_labels = list(labels) + [PARSE_ERROR]
    cm = confusion_matrix(gold, y_pred, labels=cm_labels)
    return {
        "accuracy": sum(g == q for g, q in zip(gold, y_pred)) / len(gold),
        "macro_f1": float(f.mean()) if len(f) else 0.0,
        "per_class": {lab: {"precision": float(p[k]), "recall": float(r[k]), "f1": float(f[k]), "support": int(s[k])}
                      for k, lab in enumerate(present)},
        "confusion": pd.DataFrame(cm, index=cm_labels, columns=cm_labels).loc[list(labels)],
    }


def severity_misses(items: list[BugReport], results: list[TriageResult]) -> list[dict]:
    """Critical/major items predicted as minor/trivial: the costly error."""
    by_id = {r.item_id: r for r in results}
    misses = []
    for item in items:
        o = by_id[item.id].output
        if o and item.gold and item.gold.severity in COSTLY_GOLD and o.severity in COSTLY_PRED:
            misses.append({"item_id": item.id, "gold": item.gold.severity, "predicted": o.severity,
                           "tricky": item.tricky, "title": item.title})
    return misses


def _accuracy(items: list[BugReport], by_id: dict[str, TriageResult], field: str) -> float:
    if not items:
        return 0.0
    hits = sum(by_id[i.id].output is not None and getattr(by_id[i.id].output, field) == getattr(i.gold, field)
               for i in items)
    return hits / len(items)


def exact_match(item: BugReport, result: TriageResult) -> bool:
    """True when all three labels are correct."""
    o = result.output
    return o is not None and all(getattr(o, f) == getattr(item.gold, f) for f in LABEL_FIELDS)


def evaluate_run(items: list[BugReport], results: list[TriageResult]) -> dict | None:
    """Every per-run metric the report needs, over the items that have gold labels (None if there are none)."""
    items = [i for i in items if i.gold is not None]
    if not items:
        return None
    by_id = {r.item_id: r for r in results}
    results = [by_id[i.id] for i in items]
    fields = {}
    for field, labels in LABEL_FIELDS.items():
        gold = [getattr(i.gold, field) for i in items]
        pred = [getattr(by_id[i.id].output, field) if by_id[i.id].output else None for i in items]
        fields[field] = field_metrics(gold, pred, labels)
    tricky = [i for i in items if i.tricky]
    plain = [i for i in items if not i.tricky]
    return {
        "fields": fields,
        "exact_match": sum(exact_match(i, by_id[i.id]) for i in items) / len(items),
        "parse_error_rate": sum(r.output is None for r in results) / len(results),
        "severity_misses": severity_misses(items, results),
        "tricky_split": {
            field: {"tricky": _accuracy(tricky, by_id, field), "non_tricky": _accuracy(plain, by_id, field)}
            for field in LABEL_FIELDS
        },
    }


def _label(result: TriageResult, field: str) -> str:
    return getattr(result.output, field) if result.output else PARSE_ERROR


def flipped_items(runs: dict[str, list[TriageResult]]) -> list[dict]:
    """Items whose predicted label differs between any of the given runs, per field."""
    names = list(runs)
    by_run = {n: {r.item_id: r for r in runs[n]} for n in names}
    flips = []
    for item_id in sorted(by_run[names[0]]):
        for field in LABEL_FIELDS:
            values = {n: _label(by_run[n][item_id], field) for n in names}
            if len(set(values.values())) > 1:
                flips.append({"item_id": item_id, "field": field, **values})
    return flips
