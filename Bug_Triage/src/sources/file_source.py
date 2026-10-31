"""Bug reports from a JSON file: the labelled benchmark set, or an export from any tracker."""
from pathlib import Path

from src.schemas import BUG_REPORTS_PATH, BugReport, load_bug_reports
from src.sources.base import SourceError


class FileSource:
    """Reads a JSON list of BugReport objects (see data/bug_reports.json for the format)."""

    name = "file"

    def __init__(self, path: Path = BUG_REPORTS_PATH):
        self.path = Path(path)

    def fetch(self, ids: list[str] | None = None) -> list[BugReport]:
        items = load_bug_reports(self.path)
        if not ids:
            return items
        by_id = {i.id: i for i in items}
        missing = [i for i in ids if i not in by_id]
        if missing:
            raise SourceError(f"ids not found in {self.path.name}: {missing}")
        return [by_id[i] for i in ids]
