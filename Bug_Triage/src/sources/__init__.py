"""Registry of bug sources. To add a tracker, write a class with `name` and
`fetch(ids) -> list[BugReport]`, then register a factory here."""
from collections.abc import Callable
from pathlib import Path

from src.sources.base import BugSource, SourceError
from src.sources.file_source import FileSource
from src.sources.jira_source import JiraSource

SOURCES: dict[str, Callable[..., BugSource]] = {
    "file": lambda path=None: FileSource(Path(path)) if path else FileSource(),
    "jira": lambda path=None: JiraSource.from_env(),
}


def get_source(name: str, path: str | None = None) -> BugSource:
    """Build the named source; `path` is used by the file source only."""
    if name not in SOURCES:
        raise SourceError(f"unknown source '{name}'; available: {sorted(SOURCES)}")
    return SOURCES[name](path=path)
