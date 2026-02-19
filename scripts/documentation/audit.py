#!/usr/bin/env python3
# Owner-Skill: .claude/skills/scripts-maintenance/SKILL.md
"""Audit documentation for broken links and forbidden terms."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.documentation.shared import (
    Scope,
    build_scopes,
    iter_markdown_files,
    write_json,
    write_markdown,
)

LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
INLINE_CODE_RE = re.compile(r"`[^`]*`")


def load_audit_budgets(root: Path) -> tuple[int | None, dict[str, int]]:
    """Load audit issue budgets from architecture config when available."""
    config_path: Path | None = None
    for candidate in [root, *root.parents]:
        path = candidate / "docs/architecture/architecture_config.json"
        if path.exists():
            config_path = path
            break
    if config_path is None:
        return None, {}

    payload = json.loads(config_path.read_text(encoding="utf-8", errors="ignore"))
    docs_validation = payload.get("docs_validation", {})
    audit_gate = docs_validation.get("audit_gate", {})
    default_budget = audit_gate.get("max_issues_default")
    by_scope_raw = audit_gate.get("max_issues_by_scope", {})
    by_scope = {
        str(name): int(value)
        for name, value in by_scope_raw.items()
        if isinstance(value, int | float)
    }
    if isinstance(default_budget, int | float):
        return int(default_budget), by_scope
    return None, by_scope


@dataclass(frozen=True)
class Issue:
    """Single documentation audit finding."""

    file: str
    issue_type: str
    severity: str
    message: str


def normalize_link(target: str) -> str:
    """Strip fragment and query-string from a markdown link target."""
    value = target.strip()
    if value.startswith("<") and value.endswith(">"):
        value = value[1:-1].strip()
    return value.split("#", maxsplit=1)[0].split("?", maxsplit=1)[0]


def should_skip_target(raw: str, target: str) -> bool:
    if target.startswith("http"):
        return False
    if "," in raw and ".md" not in raw and "/" not in raw:
        return True
    if " " in raw and ".md" not in raw and "/" not in raw:
        return True
    return False


def is_external(target: str) -> bool:
    """Return True when *target* points outside the repository."""
    lower = target.strip().lower().lstrip("<")
    return lower.startswith(("http://", "https://", "mailto:", "tel:", "data:"))


def broken_link_issues(scope: Scope) -> list[Issue]:
    """Collect broken internal-link issues for all markdown files in *scope*."""
    issues: list[Issue] = []
    for md_file in iter_markdown_files(scope.path):
        content = md_file.read_text(encoding="utf-8", errors="ignore")
        rel = md_file.relative_to(scope.path).as_posix()
        in_fenced_code = False
        for number, line in enumerate(content.splitlines(), start=1):
            stripped = line.lstrip()
            if stripped.startswith("```"):
                in_fenced_code = not in_fenced_code
                continue
            if in_fenced_code:
                continue
            clean_line = INLINE_CODE_RE.sub("", line)
            for raw in LINK_RE.findall(clean_line):
                target = normalize_link(raw)
                if not target or target.startswith("#") or is_external(target):
                    continue
                if should_skip_target(raw, target):
                    continue
                path = (md_file.parent / target).resolve()
                if not path.exists():
                    issues.append(
                        Issue(
                            file=rel,
                            issue_type="broken_link",
                            severity="high",
                            message=f"line {number}: target not found -> {raw}",
                        ),
                    )
    return issues


def forbidden_term_issues(scope: Scope) -> list[Issue]:
    """Collect forbidden-term issues for markdown files in *scope*."""
    issues: list[Issue] = []
    terms: tuple[str, ...] = ()
    for md_file in iter_markdown_files(scope.path):
        rel = md_file.relative_to(scope.path).as_posix()
        rel_lower = rel.lower()
        if scope.name == "root":
            if not rel_lower.startswith("docs/"):
                continue
        elif not scope.name.startswith("flext-"):
            continue
        content = md_file.read_text(encoding="utf-8", errors="ignore").lower()
        issues.extend(
            Issue(
                file=rel,
                issue_type="forbidden_term",
                severity="medium",
                message=f"contains forbidden term '{term}'",
            )
            for term in terms
            if term in content
        )
    return issues


def to_markdown(scope: Scope, issues: list[Issue]) -> list[str]:
    """Format audit issues as a markdown report."""
    return [
        "# Docs Audit Report",
        "",
        f"Scope: {scope.name}",
        f"Files scanned: {len(iter_markdown_files(scope.path))}",
        f"Issues: {len(issues)}",
        "",
        "| file | type | severity | message |",
        "|---|---|---|---|",
        *[
            f"| {issue.file} | {issue.issue_type} | {issue.severity} | {issue.message} |"
            for issue in issues
        ],
    ]


def run_scope(
    scope: Scope,
    *,
    strict: bool,
    check: str,
    max_issues_default: int | None,
    max_issues_by_scope: dict[str, int],
) -> tuple[int, int]:
    """Run configured audit checks on *scope* and write reports."""
    checks = {part.strip() for part in check.split(",") if part.strip()}
    if not checks or "all" in checks:
        checks = {"links", "forbidden-terms"}

    issues: list[Issue] = []
    if "links" in checks:
        issues.extend(broken_link_issues(scope))
    if "forbidden-terms" in checks:
        issues.extend(forbidden_term_issues(scope))

    summary = {
        "scope": scope.name,
        "issues": len(issues),
        "checks": sorted(checks),
        "strict": strict,
        "report_dir": scope.report_dir.as_posix(),
    }
    write_json(
        scope.report_dir / "audit-summary.json",
        {"summary": summary, "issues": [asdict(i) for i in issues]},
    )
    write_markdown(scope.report_dir / "audit-report.md", to_markdown(scope, issues))

    max_issues = max_issues_by_scope.get(scope.name, max_issues_default)
    if strict:
        limit = 0 if max_issues is None else max_issues
        has_error = 1 if len(issues) > limit else 0
        reason = f"issues:{len(issues)}/max:{limit}"
    else:
        has_error = 0
        reason = f"issues:{len(issues)}"
    result = "FAIL" if has_error else "OK"
    print(f"PROJECT={scope.name} PHASE=audit RESULT={result} REASON={reason}")
    return len(issues), has_error


def main() -> int:
    """CLI entry point for the documentation audit."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--project")
    parser.add_argument("--projects")
    parser.add_argument("--output-dir", default=".reports/docs")
    parser.add_argument("--check", default="all")
    parser.add_argument("--strict", type=int, default=1)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    scopes = build_scopes(
        root=root,
        project=args.project,
        projects=args.projects,
        output_dir=args.output_dir,
    )
    default_budget, by_scope_budget = load_audit_budgets(root)
    failures = 0
    for scope in scopes:
        _, failed = run_scope(
            scope,
            strict=bool(args.strict),
            check=args.check,
            max_issues_default=default_budget,
            max_issues_by_scope=by_scope_budget,
        )
        failures += failed
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
