"""Jira Cloud issues via the REST API (GET /rest/api/3/issue/{key}), mapped to BugReport."""
import base64
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request

from dotenv import load_dotenv

from src.schemas import ROOT, BugReport
from src.sources.base import SourceError, parse_triage_labels

ISSUE_FIELDS = "summary,description,labels,comment,attachment"
_WIKI_CODE = re.compile(r"\{(?:code|noformat)[^}]*\}(.*?)\{(?:code|noformat)\}", re.DOTALL)
_TEXT_EXTENSIONS = (".log", ".txt")
_BLOCK_TYPES = {"paragraph", "heading", "listItem", "blockquote", "panel"}


def doc_to_text(node: object) -> tuple[str, list[str]]:
    """Flatten a Jira rich-text field into (plain text, code blocks).

    Handles Atlassian Document Format (REST v3, a dict) and wiki markup (a string).
    """
    if node is None:
        return "", []
    if isinstance(node, str):
        return _WIKI_CODE.sub("", node).strip(), [c.strip("\n") for c in _WIKI_CODE.findall(node)]
    text: list[str] = []
    code: list[str] = []

    def plain(n: dict) -> str:
        return n.get("text", "") + "".join(plain(c) for c in n.get("content", []) or [])

    def walk(n: dict) -> None:
        kind = n.get("type")
        if kind == "codeBlock":
            code.append(plain(n))
            return
        if kind == "text":
            text.append(n.get("text", ""))
        elif kind == "hardBreak":
            text.append("\n")
        for child in n.get("content", []) or []:
            walk(child)
        if kind in _BLOCK_TYPES:
            text.append("\n")

    if isinstance(node, dict):
        walk(node)
    return "".join(text).strip(), code


def issue_to_bug_report(issue: dict, base_url: str, attachment_texts: list[str] | None = None) -> BugReport:
    """Map a Jira issue JSON object to a BugReport.

    The log is built from code blocks in the description and comments plus text attachments.
    Gold labels come only from triage labels set by a human (see parse_triage_labels).
    """
    fields = issue.get("fields", {})
    description, logs = doc_to_text(fields.get("description"))
    for comment in (fields.get("comment") or {}).get("comments", []):
        logs += doc_to_text(comment.get("body"))[1]
    logs += attachment_texts or []
    gold, platform = parse_triage_labels(fields.get("labels") or [])
    key = issue["key"]
    return BugReport(
        id=key, title=fields.get("summary") or key, description=description or "(no description)",
        log_snippet="\n".join(block.strip("\n") for block in logs if block.strip()),
        platform=platform, gold=gold, source="jira", url=f"{base_url}/browse/{key}",
    )


class JiraSource:
    """Fetch specific Jira issues by key. Credentials come from .env (JIRA_BASE_URL, JIRA_EMAIL, JIRA_API_TOKEN)."""

    name = "jira"

    def __init__(self, base_url: str, token: str, email: str | None = None,
                 timeout: int = 30, max_attachment_bytes: int = 20_000):
        self.base_url = base_url.rstrip("/")
        if email:
            self._auth = "Basic " + base64.b64encode(f"{email}:{token}".encode()).decode()
        else:
            self._auth = f"Bearer {token}"
        self._timeout = timeout
        self._max_attachment = max_attachment_bytes

    @classmethod
    def from_env(cls) -> "JiraSource":
        load_dotenv(ROOT / ".env")
        base, token = os.getenv("JIRA_BASE_URL"), os.getenv("JIRA_API_TOKEN")
        if not base or not token:
            raise SourceError("Set JIRA_BASE_URL and JIRA_API_TOKEN (and JIRA_EMAIL for Jira Cloud) in .env")
        return cls(base, token, os.getenv("JIRA_EMAIL") or None)

    def _get(self, url: str, limit: int | None = None) -> bytes:
        req = urllib.request.Request(url, headers={"Authorization": self._auth, "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                return resp.read(limit) if limit else resp.read()
        except urllib.error.HTTPError as e:
            hint = {401: "check JIRA_EMAIL / JIRA_API_TOKEN", 403: "no permission for this issue",
                    404: "issue not found or not visible to this account"}.get(e.code, "")
            raise SourceError(f"Jira returned {e.code} for {url} {hint}".strip()) from e
        except (urllib.error.URLError, TimeoutError) as e:
            raise SourceError(f"cannot reach Jira at {self.base_url}: {e}") from e

    def _attachment_texts(self, issue: dict) -> list[str]:
        texts = []
        for att in (issue.get("fields") or {}).get("attachment") or []:
            name, mime = att.get("filename", ""), att.get("mimeType", "")
            if att.get("content") and (name.lower().endswith(_TEXT_EXTENSIONS) or mime.startswith("text/")):
                texts.append(self._get(att["content"], self._max_attachment).decode("utf-8", errors="replace"))
        return texts

    def fetch(self, ids: list[str] | None = None) -> list[BugReport]:
        if not ids:
            raise SourceError("Pass at least one Jira issue key, e.g. --ids PROJ-123")
        reports = []
        for key in ids:
            url = f"{self.base_url}/rest/api/3/issue/{urllib.parse.quote(key)}?fields={ISSUE_FIELDS}"
            issue = json.loads(self._get(url))
            reports.append(issue_to_bug_report(issue, self.base_url, self._attachment_texts(issue)))
        return reports
