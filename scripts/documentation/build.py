#!/usr/bin/env python3
# Owner-Skill: .claude/skills/scripts-maintenance/SKILL.md
"""Build MkDocs sites for workspace projects."""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from scripts.documentation.shared import (
    Scope,
    build_scopes,
    write_json,
    write_markdown,
)
from scripts.libs.config import STATUS_FAIL, STATUS_OK


@dataclass(frozen=True)
class BuildResult:
    """Outcome of a single MkDocs build attempt."""

    scope: str
    result: str
    reason: str
    site_dir: str


def run_mkdocs(scope: Scope) -> BuildResult:
    """Run ``mkdocs build --strict`` for *scope* and return the result."""
    config = scope.path / "mkdocs.yml"
    if not config.exists():
        return BuildResult(
            scope=scope.name, result="SKIP", reason="mkdocs.yml not found", site_dir=""
        )

    site_dir = (scope.path / ".reports/docs/site").resolve()
    site_dir.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["mkdocs", "build", "--strict", "-f", str(config), "-d", str(site_dir)]
    completed = subprocess.run(
        cmd, cwd=scope.path, check=False, capture_output=True, text=True
    )
    if completed.returncode == 0:
        return BuildResult(
            scope=scope.name,
            result=STATUS_OK,
            reason="build succeeded",
            site_dir=site_dir.as_posix(),
        )
    reason = (completed.stderr or completed.stdout).strip().splitlines()
    tail = reason[-1] if reason else f"mkdocs exited {completed.returncode}"
    return BuildResult(
        scope=scope.name, result=STATUS_FAIL, reason=tail, site_dir=site_dir.as_posix()
    )


def write_reports(scope: Scope, result: BuildResult) -> None:
    """Persist build JSON summary and markdown report for *scope*."""
    write_json(scope.report_dir / "build-summary.json", {"summary": asdict(result)})
    write_markdown(
        scope.report_dir / "build-report.md",
        [
            "# Docs Build Report",
            "",
            f"Scope: {result.scope}",
            f"Result: {result.result}",
            f"Reason: {result.reason}",
            f"Site dir: {result.site_dir}",
        ],
    )


def main() -> int:
    """CLI entry point for the documentation build."""
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--root", default=".")
    _ = parser.add_argument("--project")
    _ = parser.add_argument("--projects")
    _ = parser.add_argument("--output-dir", default=".reports/docs")
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
        result = run_mkdocs(scope)
        write_reports(scope, result)
        print(
            f"PROJECT={scope.name} PHASE=build RESULT={result.result} REASON={result.reason}"
        )
        if result.result == STATUS_FAIL:
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
