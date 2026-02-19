from __future__ import annotations

import argparse
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from _shared import (
    Scope,
    build_scopes,
    iter_markdown_files,
    write_json,
    write_markdown,
)

LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


@dataclass(frozen=True)
class Issue:
    file: str
    issue_type: str
    severity: str
    message: str


def normalize_link(target: str) -> str:
    value = target.strip()
    value = value.split("#", maxsplit=1)[0]
    value = value.split("?", maxsplit=1)[0]
    return value


def is_external(target: str) -> bool:
    lower = target.lower()
    return lower.startswith(("http://", "https://", "mailto:", "tel:", "data:"))


def broken_link_issues(scope: Scope) -> list[Issue]:
    issues: list[Issue] = []
    for md_file in iter_markdown_files(scope.path):
        content = md_file.read_text(encoding="utf-8", errors="ignore")
        rel = md_file.relative_to(scope.path).as_posix()
        for number, line in enumerate(content.splitlines(), start=1):
            for raw in LINK_RE.findall(line):
                target = normalize_link(raw)
                if not target or target.startswith("#") or is_external(target):
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
    issues: list[Issue] = []
    terms = ("client-a", "client-b")
    for md_file in iter_markdown_files(scope.path):
        rel = md_file.relative_to(scope.path).as_posix()
        rel_lower = rel.lower()
        if scope.name == "root":
            if not rel_lower.startswith("docs/"):
                continue
        elif not scope.name.startswith("flext-"):
            continue
        content = md_file.read_text(encoding="utf-8", errors="ignore").lower()
        for term in terms:
            if term in content:
                issues.append(
                    Issue(
                        file=rel,
                        issue_type="forbidden_term",
                        severity="medium",
                        message=f"contains forbidden term '{term}'",
                    ),
                )
    return issues


def to_markdown(scope: Scope, issues: list[Issue]) -> list[str]:
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


def run_scope(scope: Scope, strict: bool, check: str) -> tuple[int, int]:
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

    has_error = 1 if strict and issues else 0
    result = "FAIL" if has_error else ("WARN" if issues else "OK")
    print(
        f"PROJECT={scope.name} PHASE=audit RESULT={result} REASON=issues:{len(issues)}"
    )
    return len(issues), has_error


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--project")
    parser.add_argument("--projects")
    parser.add_argument("--output-dir", default=".reports/docs")
    parser.add_argument("--check", default="all")
    parser.add_argument("--strict", type=int, default=0)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    scopes = build_scopes(
        root=root,
        project=args.project,
        projects=args.projects,
        output_dir=args.output_dir,
    )
    failures = 0
    for scope in scopes:
        _, failed = run_scope(scope, strict=bool(args.strict), check=args.check)
        failures += failed
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
