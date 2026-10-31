"""Bug sources: label convention, Jira rich-text flattening and issue mapping (no network)."""
import pytest

from src.sources import SourceError, get_source
from src.sources.base import parse_triage_labels
from src.sources.jira_source import JiraSource, doc_to_text, issue_to_bug_report

ADF = {
    "type": "doc", "version": 1,
    "content": [
        {"type": "paragraph", "content": [{"type": "text", "text": "Playback stops after 60 min."}]},
        {"type": "codeBlock", "content": [{"type": "text", "text": "ERROR [drm] license denied status=401"}]},
        {"type": "paragraph", "content": [{"type": "text", "text": "Every title."}]},
    ],
}


def test_parse_triage_labels_requires_all_three_for_gold():
    gold, platform = parse_triage_labels(["severity:critical", "component-drm", "root_cause=token-expiry", "platform:ios"])
    assert (gold.severity, gold.component, gold.root_cause_category, platform) == ("critical", "drm", "token_expiry", "ios")
    assert parse_triage_labels(["severity:critical", "component:drm"]) == (None, "unknown")
    assert parse_triage_labels(["severity:blocker", "frontend"]) == (None, "unknown")


def test_doc_to_text_adf_and_wiki_markup():
    assert doc_to_text(ADF) == ("Playback stops after 60 min.\nEvery title.", ["ERROR [drm] license denied status=401"])
    text, code = doc_to_text("Steps below\n{code}ERROR x{code}\nend")
    assert code == ["ERROR x"] and "ERROR" not in text
    assert doc_to_text(None) == ("", [])


def test_issue_to_bug_report_maps_fields_and_logs():
    issue = {
        "key": "PROJ-123",
        "fields": {
            "summary": "License error", "description": ADF,
            "labels": ["severity:critical", "component:drm", "root_cause:token_expiry"],
            "comment": {"comments": [{"body": {"type": "doc", "content": [
                {"type": "codeBlock", "content": [{"type": "text", "text": "WARN [cdn] retry"}]}]}}]},
        },
    }
    report = issue_to_bug_report(issue, "https://example.atlassian.net", ["INFO attachment line"])
    assert report.id == "PROJ-123" and report.source == "jira"
    assert report.url == "https://example.atlassian.net/browse/PROJ-123"
    assert report.log_snippet.splitlines() == ["ERROR [drm] license denied status=401", "WARN [cdn] retry", "INFO attachment line"]
    assert report.gold.severity == "critical" and report.platform == "unknown"


def test_unlabelled_issue_has_no_gold():
    report = issue_to_bug_report({"key": "OPS-7", "fields": {"summary": "s"}}, "https://x.atlassian.net")
    assert report.gold is None and report.description == "(no description)" and report.log_snippet == ""


def test_jira_auth_header_modes():
    assert JiraSource("https://x", "tok", "me@x.com")._auth.startswith("Basic ")
    assert JiraSource("https://x", "tok")._auth == "Bearer tok"


def test_file_source_filters_ids_and_reports_missing():
    assert [r.id for r in get_source("file").fetch(["BUG-012", "BUG-001"])] == ["BUG-012", "BUG-001"]
    with pytest.raises(SourceError, match="BUG-999"):
        get_source("file").fetch(["BUG-999"])


def test_unknown_source_and_jira_without_ids():
    with pytest.raises(SourceError, match="unknown source"):
        get_source("bugzilla")
    with pytest.raises(SourceError, match="at least one"):
        JiraSource("https://x", "tok", "me@x.com").fetch([])
