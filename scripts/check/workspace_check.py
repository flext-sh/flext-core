#!/usr/bin/env python3
# Owner-Skill: .claude/skills/scripts-validation/SKILL.md
"""Run lint gates across FLEXT workspace projects and generate reports.

Gates: ruff lint/format, pyrefly, mypy, pyright, bandit, markdownlint, go vet.
Outputs: ``.reports/check/check-report.md`` and ``.reports/check/check-report.sarif``.

Usage::

    python scripts/check/workspace_check.py [options] project1 project2 ...
    python scripts/check/workspace_check.py --gates lint,pyrefly,mypy flext-core flext-api
    python scripts/check/workspace_check.py --gates lint,type --reports-dir .reports/check flext-core
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / ".reports" / "check"
DEFAULT_GATES = "lint,format,pyrefly,mypy,pyright,security,markdown,go"
DEFAULT_SRC_DIR = "src"
_CHECK_DIRS = ("src", "tests", "examples", "scripts")


def _existing_check_dirs(project_dir: Path) -> list[str]:
    """Return subset of _CHECK_DIRS that exist in *project_dir*."""
    return [d for d in _CHECK_DIRS if (project_dir / d).is_dir()]


@dataclass
class CheckError:
    """Represent one lint or type-check issue."""

    file: str
    line: int
    column: int
    code: str
    message: str
    severity: str = "error"


@dataclass
class GateResult:
    """Store the result of running one gate for one project."""

    gate: str
    project: str
    passed: bool
    errors: list[CheckError] = field(default_factory=list)
    raw_output: str = ""

    @property
    def error_count(self) -> int:
        """Return total issues found by this gate."""
        return len(self.errors)


@dataclass
class ProjectResult:
    """Aggregate all gate results for a project."""

    project: str
    gates: dict[str, GateResult] = field(default_factory=dict)

    @property
    def total_errors(self) -> int:
        """Return total issues across all gates."""
        return sum(g.error_count for g in self.gates.values())

    @property
    def passed(self) -> bool:
        """Return True when every gate passed."""
        return all(g.passed for g in self.gates.values())


def _run(
    cmd: list[str], cwd: Path, timeout: int = 300
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=cwd,
        timeout=timeout,
        check=False,
    )


def _run_ruff_lint(project_dir: Path, _src_dir: str) -> GateResult:
    check_dirs = _existing_check_dirs(project_dir)
    targets = check_dirs or ["."]
    result = _run(
        ["ruff", "check", *targets, "--output-format", "json", "--quiet"],
        project_dir,
    )
    errors: list[CheckError] = []
    try:
        for e in json.loads(result.stdout or "[]"):
            loc = e.get("location", {})
            errors.append(
                CheckError(
                    file=e.get("filename", "?"),
                    line=loc.get("row", 0),
                    column=loc.get("column", 0),
                    code=e.get("code", ""),
                    message=e.get("message", ""),
                )
            )
    except (json.JSONDecodeError, KeyError):
        pass
    return GateResult(
        gate="lint",
        project=project_dir.name,
        passed=result.returncode == 0,
        errors=errors,
        raw_output=result.stderr,
    )


def _run_ruff_format(project_dir: Path, _src_dir: str) -> GateResult:
    check_dirs = _existing_check_dirs(project_dir)
    targets = check_dirs or ["."]
    result = _run(["ruff", "format", "--check", *targets, "--quiet"], project_dir)
    errors: list[CheckError] = []
    if result.returncode != 0 and result.stdout.strip():
        for line in result.stdout.strip().splitlines():
            path = line.strip()
            if path and not path.startswith("Would"):
                errors.append(
                    CheckError(
                        file=path,
                        line=0,
                        column=0,
                        code="format",
                        message="Would be reformatted",
                    )
                )
    return GateResult(
        gate="format",
        project=project_dir.name,
        passed=result.returncode == 0,
        errors=errors,
        raw_output=result.stderr,
    )


def _run_pyrefly(project_dir: Path, src_dir: str, reports_dir: Path) -> GateResult:
    check_dirs = _existing_check_dirs(project_dir)
    targets = check_dirs or [src_dir]
    json_file = reports_dir / f"{project_dir.name}-pyrefly.json"
    cmd = [
        "pyrefly",
        "check",
        *targets,
        "--config",
        "pyproject.toml",
        "--output-format",
        "json",
        "-o",
        str(json_file),
        "--summary=none",
    ]
    result = _run(cmd, project_dir)

    errors: list[CheckError] = []
    if json_file.exists():
        try:
            data = json.loads(json_file.read_text())
            raw_errors = data.get("errors", []) if isinstance(data, dict) else data
            errors.extend(
                CheckError(
                    file=e.get("path", "?"),
                    line=e.get("line", 0),
                    column=e.get("column", 0),
                    code=e.get("name", ""),
                    message=e.get("description", ""),
                    severity=e.get("severity", "error"),
                )
                for e in raw_errors
            )
        except (json.JSONDecodeError, KeyError):
            pass

    if not errors and result.returncode != 0:
        m = re.search(r"(\d+)\s+errors?", result.stderr + result.stdout)
        if m:
            errors = [
                CheckError(
                    file="?",
                    line=0,
                    column=0,
                    code="pyrefly",
                    message=f"Pyrefly reported {m.group(1)} error(s)",
                )
            ] * int(m.group(1))

    return GateResult(
        gate="pyrefly",
        project=project_dir.name,
        passed=result.returncode == 0,
        errors=errors,
        raw_output=result.stderr,
    )


def _run_bandit(project_dir: Path, src_dir: str) -> GateResult:
    src_path = project_dir / src_dir
    if not src_path.exists():
        return GateResult(gate="security", project=project_dir.name, passed=True)
    result = _run(["bandit", "-r", src_dir, "-f", "json", "-q", "-ll"], project_dir)
    errors: list[CheckError] = []
    try:
        data = json.loads(result.stdout or "{}")
        errors.extend(
            CheckError(
                file=e.get("filename", "?"),
                line=e.get("line_number", 0),
                column=0,
                code=e.get("test_id", ""),
                message=e.get("issue_text", ""),
                severity=e.get("issue_severity", "MEDIUM").lower(),
            )
            for e in data.get("results", [])
        )
    except (json.JSONDecodeError, KeyError):
        pass
    return GateResult(
        gate="security",
        project=project_dir.name,
        passed=result.returncode == 0,
        errors=errors,
        raw_output=result.stderr,
    )


def _run_mypy(project_dir: Path, _src_dir: str) -> GateResult:
    check_dirs = _existing_check_dirs(project_dir)
    if not check_dirs:
        return GateResult(gate="mypy", project=project_dir.name, passed=True)

    config_file = str(ROOT / "pyproject.toml")
    result = _run(
        ["mypy", *check_dirs, "--config-file", config_file, "--output", "json"],
        project_dir,
    )

    errors: list[CheckError] = []
    for raw_line in (result.stdout or "").splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            e = json.loads(raw_line)
            if e.get("severity") in {"error", "warning", "note"}:
                errors.append(
                    CheckError(
                        file=e.get("file", "?"),
                        line=e.get("line", 0),
                        column=e.get("column", 0),
                        code=e.get("code", ""),
                        message=e.get("message", ""),
                        severity=e.get("severity", "error"),
                    )
                )
        except json.JSONDecodeError:
            continue
    return GateResult(
        gate="mypy",
        project=project_dir.name,
        passed=result.returncode == 0,
        errors=errors,
        raw_output=result.stderr,
    )


def _run_pyright(project_dir: Path, _src_dir: str) -> GateResult:
    check_dirs = _existing_check_dirs(project_dir)
    if not check_dirs:
        return GateResult(gate="pyright", project=project_dir.name, passed=True)
    result = _run(["pyright", *check_dirs, "--outputjson"], project_dir, timeout=600)
    errors: list[CheckError] = []
    try:
        data = json.loads(result.stdout or "{}")
        for e in data.get("generalDiagnostics", []):
            rng = e.get("range", {}).get("start", {})
            errors.append(
                CheckError(
                    file=e.get("file", "?"),
                    line=rng.get("line", 0) + 1,
                    column=rng.get("character", 0) + 1,
                    code=e.get("rule", ""),
                    message=e.get("message", ""),
                    severity=e.get("severity", "error"),
                )
            )
    except (json.JSONDecodeError, KeyError):
        pass
    return GateResult(
        gate="pyright",
        project=project_dir.name,
        passed=result.returncode == 0,
        errors=errors,
        raw_output=result.stderr,
    )


def _collect_markdown_files(project_dir: Path) -> list[Path]:
    excluded = {
        ".git",
        ".reports",
        ".venv",
        "node_modules",
        ".flext-deps",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "dist",
        "build",
        "reports",
    }
    files: list[Path] = []
    for path in project_dir.rglob("*.md"):
        if any(part in excluded for part in path.parts):
            continue
        files.append(path)
    return files


def _run_markdown(project_dir: Path) -> GateResult:
    md_files = _collect_markdown_files(project_dir)
    if not md_files:
        return GateResult(gate="markdown", project=project_dir.name, passed=True)
    cmd = ["markdownlint"]
    root_config = ROOT / ".markdownlint.json"
    local_config = project_dir / ".markdownlint.json"
    if root_config.exists():
        cmd.extend(["--config", str(root_config)])
    elif local_config.exists():
        cmd.extend(["--config", str(local_config)])
    cmd.extend(str(p.relative_to(project_dir)) for p in md_files)
    result = _run(
        cmd,
        project_dir,
    )
    errors: list[CheckError] = []
    pattern = re.compile(
        r"^(?P<file>.*?):(?P<line>\d+)(?::(?P<col>\d+))?\s+error\s+(?P<code>MD\d+)(?:/[^\s]+)?\s+(?P<msg>.*)$"
    )
    for line in (result.stdout + "\n" + result.stderr).splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        errors.append(
            CheckError(
                file=match.group("file"),
                line=int(match.group("line")),
                column=int(match.group("col") or 1),
                code=match.group("code"),
                message=match.group("msg"),
            )
        )
    if result.returncode != 0 and not errors:
        errors.append(
            CheckError(
                file=".",
                line=1,
                column=1,
                code="markdownlint",
                message=(
                    result.stdout or result.stderr or "markdownlint failed"
                ).strip(),
            )
        )

    return GateResult(
        gate="markdown",
        project=project_dir.name,
        passed=result.returncode == 0,
        errors=errors,
        raw_output=result.stderr,
    )


def _run_go(project_dir: Path) -> GateResult:
    if not (project_dir / "go.mod").exists():
        return GateResult(gate="go", project=project_dir.name, passed=True)
    errors: list[CheckError] = []
    raw_output = ""

    vet_result = _run(["go", "vet", "./..."], project_dir, timeout=900)
    raw_output = "\n".join(
        part for part in (vet_result.stdout, vet_result.stderr) if part
    )
    vet_pattern = re.compile(
        r"^(?P<file>[^:\n]+\.go):(?P<line>\d+)(?::(?P<col>\d+))?:\s*(?P<msg>.*)$"
    )
    for line in (vet_result.stdout + "\n" + vet_result.stderr).splitlines():
        match = vet_pattern.match(line.strip())
        if not match:
            continue
        errors.append(
            CheckError(
                file=match.group("file"),
                line=int(match.group("line")),
                column=int(match.group("col") or 1),
                code="govet",
                message=match.group("msg"),
            )
        )
    if vet_result.returncode != 0 and not errors:
        errors.append(
            CheckError(
                file=".",
                line=1,
                column=1,
                code="govet",
                message=(
                    vet_result.stdout or vet_result.stderr or "go vet failed"
                ).strip(),
            )
        )

    go_files = list(project_dir.rglob("*.go"))
    if go_files:
        fmt_result = _run(
            ["gofmt", "-l", *[str(p.relative_to(project_dir)) for p in go_files]],
            project_dir,
            timeout=900,
        )
        fmt_raw_output = "\n".join(
            part for part in (fmt_result.stdout, fmt_result.stderr) if part
        )
        raw_output = "\n".join(part for part in (raw_output, fmt_raw_output) if part)
        for file_name in fmt_result.stdout.splitlines():
            file_name = file_name.strip()
            if not file_name:
                continue
            errors.append(
                CheckError(
                    file=file_name,
                    line=1,
                    column=1,
                    code="gofmt",
                    message="File is not gofmt-formatted",
                )
            )

    return GateResult(
        gate="go",
        project=project_dir.name,
        passed=len(errors) == 0,
        errors=errors,
        raw_output=raw_output,
    )


def check_project(
    project_dir: Path, gates: list[str], reports_dir: Path
) -> ProjectResult:
    """Run selected gates for one project and collect results."""
    result = ProjectResult(project=project_dir.name)
    src_dir = DEFAULT_SRC_DIR
    runners = {
        "lint": lambda: _run_ruff_lint(project_dir, src_dir),
        "format": lambda: _run_ruff_format(project_dir, src_dir),
        "pyrefly": lambda: _run_pyrefly(project_dir, src_dir, reports_dir),
        "mypy": lambda: _run_mypy(project_dir, src_dir),
        "pyright": lambda: _run_pyright(project_dir, src_dir),
        "security": lambda: _run_bandit(project_dir, src_dir),
        "markdown": lambda: _run_markdown(project_dir),
        "go": lambda: _run_go(project_dir),
    }
    for gate in gates:
        runner = runners.get(gate)
        if runner:
            result.gates[gate] = runner()
    return result


def _generate_md(results: list[ProjectResult], gates: list[str], timestamp: str) -> str:
    lines: list[str] = [
        "# FLEXT Check Report",
        "",
        f"**Generated**: {timestamp}  ",
        f"**Projects**: {len(results)}  ",
        f"**Gates**: {', '.join(gates)}  ",
        "",
        "## Summary",
        "",
    ]

    header = "| Project |"
    sep = "|---------|"
    for g in gates:
        header += f" {g.capitalize()} |"
        sep += "------|"
    header += " Total | Status |"
    sep += "-------|--------|"
    lines.extend([header, sep])

    total_all = 0
    failed_count = 0
    for r in results:
        row = f"| {r.project} |"
        for g in gates:
            gate = r.gates.get(g)
            row += f" {gate.error_count if gate else 0} |"
        status = "PASS" if r.passed else "**FAIL**"
        if not r.passed:
            failed_count += 1
        row += f" {r.total_errors} | {status} |"
        total_all += r.total_errors
        lines.append(row)

    lines.extend([
        "",
        f"**Total errors**: {total_all}  ",
        f"**Failed projects**: {failed_count}/{len(results)}  ",
        "",
    ])

    for r in sorted(results, key=lambda x: x.total_errors, reverse=True):
        if r.total_errors == 0:
            continue
        lines.extend([f"## {r.project}", ""])
        for g in gates:
            gate = r.gates.get(g)
            if not gate or gate.error_count == 0:
                continue
            lines.extend([f"### {g} ({gate.error_count} errors)", "", "```"])
            for err in gate.errors[:50]:
                code_part = f"[{err.code}] " if err.code else ""
                lines.append(
                    f"{err.file}:{err.line}:{err.column} {code_part}{err.message}"
                )
            if gate.error_count > 50:
                lines.append(f"... and {gate.error_count - 50} more errors")
            lines.extend(["```", ""])

    return "\n".join(lines)


def _generate_sarif(
    results: list[ProjectResult], gates: list[str]
) -> dict[str, object]:
    tool_info = {
        "lint": ("Ruff Linter", "https://docs.astral.sh/ruff/"),
        "format": ("Ruff Formatter", "https://docs.astral.sh/ruff/formatter/"),
        "pyrefly": ("Pyrefly", "https://github.com/facebook/pyrefly"),
        "mypy": ("Mypy", "https://mypy.readthedocs.io/"),
        "pyright": ("Pyright", "https://github.com/microsoft/pyright"),
        "security": ("Bandit", "https://bandit.readthedocs.io/"),
        "markdown": ("MarkdownLint", "https://github.com/DavidAnson/markdownlint"),
        "go": ("Go Vet", "https://pkg.go.dev/cmd/vet"),
    }

    runs = []
    for gate in gates:
        tool_name, tool_url = tool_info.get(gate, (gate, ""))
        sarif_results: list[dict[str, object]] = []
        rules_seen: set[str] = set()
        rules: list[dict[str, object]] = []

        for project in results:
            gate_result = project.gates.get(gate)
            if not gate_result:
                continue
            for err in gate_result.errors:
                rule_id = err.code or "unknown"
                if rule_id not in rules_seen:
                    rules_seen.add(rule_id)
                    rules.append({"id": rule_id, "shortDescription": {"text": rule_id}})
                sarif_results.append({
                    "ruleId": rule_id,
                    "level": "error" if err.severity == "error" else "warning",
                    "message": {"text": err.message},
                    "locations": [
                        {
                            "physicalLocation": {
                                "artifactLocation": {
                                    "uri": err.file,
                                    "uriBaseId": "%SRCROOT%",
                                },
                                "region": {
                                    "startLine": max(err.line, 1),
                                    "startColumn": max(err.column, 1),
                                },
                            }
                        }
                    ],
                })

        runs.append({
            "tool": {
                "driver": {
                    "name": tool_name,
                    "informationUri": tool_url,
                    "rules": rules,
                }
            },
            "results": sarif_results,
        })

    return {
        "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/main/Schemata/sarif-schema-2.1.0.json",
        "version": "2.1.0",
        "runs": runs,
    }


def main() -> int:
    """Run workspace checks and write markdown plus SARIF reports."""
    parser = argparse.ArgumentParser(description="FLEXT Workspace Check")
    _ = parser.add_argument("projects", nargs="*")
    _ = parser.add_argument("--gates", default=DEFAULT_GATES)
    _ = parser.add_argument("--reports-dir", default=str(REPORTS_DIR))
    _ = parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    if not args.projects:
        print("ERROR: no projects specified", file=sys.stderr)
        return 1

    requested_gates = [g.strip() for g in args.gates.split(",") if g.strip()]
    gates: list[str] = []
    for gate in requested_gates:
        resolved_gate = "pyrefly" if gate == "type" else gate
        if resolved_gate not in {
            "lint",
            "format",
            "pyrefly",
            "mypy",
            "pyright",
            "security",
            "markdown",
            "go",
        }:
            print(f"ERROR: unknown gate '{gate}'", file=sys.stderr)
            return 2
        if resolved_gate not in gates:
            gates.append(resolved_gate)

    reports_dir = Path(args.reports_dir).expanduser()
    if not reports_dir.is_absolute():
        reports_dir = (Path.cwd() / reports_dir).resolve()

    reports_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[ProjectResult] = []
    failed = 0
    total = len(args.projects)

    for i, proj_name in enumerate(args.projects, 1):
        project_dir = ROOT / proj_name
        if not project_dir.is_dir() or not (project_dir / "pyproject.toml").exists():
            print(f"[{i:2d}/{total:2d}] {proj_name} ... skipped")
            continue

        _ = sys.stdout.write(f"[{i:2d}/{total:2d}] {proj_name} ... ")
        _ = sys.stdout.flush()
        result = check_project(project_dir, gates, reports_dir)
        all_results.append(result)

        if result.passed:
            print("ok")
        else:
            counts = " ".join(
                f"{g}={result.gates[g].error_count}"
                for g in gates
                if g in result.gates and result.gates[g].error_count > 0
            )
            print(f"FAIL ({result.total_errors} errors: {counts})")
            failed += 1
            if args.fail_fast:
                break

    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

    md_path = reports_dir / "check-report.md"
    _ = md_path.write_text(_generate_md(all_results, gates, timestamp))

    sarif_path = reports_dir / "check-report.sarif"
    _ = sarif_path.write_text(json.dumps(_generate_sarif(all_results, gates), indent=2))

    total_errors = sum(r.total_errors for r in all_results)
    print(f"\n{'=' * 60}")
    print(f"Check: {len(all_results)} projects, {total_errors} errors, {failed} failed")
    print(f"Reports: {md_path}")
    print(f"         {sarif_path}")
    print(f"{'=' * 60}")

    if total_errors > 0:
        print("\nErrors by project:")
        for r in sorted(all_results, key=lambda x: x.total_errors, reverse=True):
            if r.total_errors > 0:
                breakdown = ", ".join(
                    f"{g}={r.gates[g].error_count}"
                    for g in gates
                    if g in r.gates and r.gates[g].error_count > 0
                )
                print(f"  {r.project:30s} {r.total_errors:6d}  ({breakdown})")

    return 1 if failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
