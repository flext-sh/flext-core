#!/usr/bin/env python3
# Owner-Skill: .claude/skills/scripts-maintenance/SKILL.md
"""Run docs pipeline project-by-project with compact status output."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def parse_projects(raw: str) -> list[str]:
    """Parse whitespace-separated project names."""
    return [part for part in raw.split() if part]


def sync_base_mk(root: Path, projects: list[str]) -> None:
    """Sync root base.mk into selected project directories."""
    source = root / "base.mk"
    content = source.read_text(encoding="utf-8")
    for name in projects:
        target = root / name / "base.mk"
        target.write_text(content, encoding="utf-8")


def read_summary(report_dir: Path, file_name: str) -> dict[str, object]:
    """Read summary payload from one docs report file."""
    path = report_dir / file_name
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    summary = payload.get("summary", {})
    return summary if isinstance(summary, dict) else {}


def summarize(report_dir: Path, phase: str) -> str:
    """Build compact phase summary from docs reports."""
    parts: list[str] = []
    if phase in {"all", "generate"}:
        parts.append(
            f"gen={read_summary(report_dir, 'generate-summary.json').get('generated', 0)}"
        )
    if phase in {"all", "fix"}:
        parts.append(
            f"fix={read_summary(report_dir, 'fix-summary.json').get('changed_files', 0)}"
        )
    if phase in {"all", "audit"}:
        parts.append(
            f"audit={read_summary(report_dir, 'audit-summary.json').get('issues', 0)}"
        )
    if phase in {"all", "build"}:
        parts.append(
            f"build={read_summary(report_dir, 'build-summary.json').get('result', 'OK')}"
        )
    if phase in {"all", "validate"}:
        parts.append(
            f"validate={read_summary(report_dir, 'validate-summary.json').get('result', 'OK')}"
        )
    return " ".join(parts)


def extract_reason(log_text: str) -> str:
    """Extract one-line failure reason from command logs."""
    lines = [line.strip() for line in log_text.splitlines() if line.strip()]
    for line in lines:
        if line.startswith("Error:"):
            return line.replace(" ", "_")
    if lines:
        return lines[-1].replace(" ", "_")
    return "docs-run-failed"


def main() -> int:
    """Run docs phase across selected projects."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--projects", required=True)
    parser.add_argument("--phase", default="all")
    parser.add_argument("--fix", default="")
    parser.add_argument("--fail-fast", default="0")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    projects = parse_projects(args.projects)
    reports_dir = root / ".reports/docs"
    reports_dir.mkdir(parents=True, exist_ok=True)

    sync_base_mk(root, projects)

    failed = 0
    total = 0
    for project in projects:
        total += 1
        log_file = reports_dir / f"docs-{project}.log"
        cmd = ["make", "-C", project, "docs", "-s", f"DOCS_PHASE={args.phase}"]
        if args.fix == "1":
            cmd.append("FIX=1")
        completed = subprocess.run(
            cmd,
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        log_text = (completed.stdout or "") + (completed.stderr or "")
        log_file.write_text(log_text, encoding="utf-8")

        if completed.returncode == 0:
            summary = summarize(root / project / ".reports/docs", args.phase)
            print(f"PROJECT={project} RESULT=OK {summary}".rstrip())
            continue

        failed += 1
        print(f"PROJECT={project} RESULT=FAIL REASON={extract_reason(log_text)}")
        if args.fail_fast == "1":
            print(f"DOCS RESULT=FAIL failed={failed}/{total}")
            return 1

    if failed:
        print(f"DOCS RESULT=FAIL failed={failed}/{total}")
        return 1
    print(f"DOCS RESULT=OK projects={total} phase={args.phase}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
