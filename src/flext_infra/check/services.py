"""Quality gate execution and workspace checking services."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import override

import structlog
from flext_core.result import FlextResult as r
from flext_core.service import FlextService
from flext_core.typings import t

from flext_infra.constants import ic
from flext_infra.discovery import DiscoveryService
from flext_infra.json_io import JsonService
from flext_infra.models import im
from flext_infra.paths import PathResolver
from flext_infra.reporting import ReportingService
from flext_infra.subprocess import CommandRunner

DEFAULT_GATES = ic.Gates.DEFAULT_CSV
_REQUIRED_EXCLUDES = ['"**/*_pb2*.py"', '"**/*_pb2_grpc*.py"']
_RUFF_FORMAT_FILE_RE = re.compile(r"^\s*-->\s*(.+?):\d+:\d+\s*$")
_MARKDOWN_RE = re.compile(
    r"^(?P<file>.*?):(?P<line>\d+)(?::(?P<col>\d+))?\s+error\s+"
    r"(?P<code>MD\d+)(?:/[^\s]+)?\s+(?P<msg>.*)$",
)
_GO_VET_RE = re.compile(
    r"^(?P<file>[^:\n]+\.go):(?P<line>\d+)(?::(?P<col>\d+))?:\s*(?P<msg>.*)$",
)
logger = structlog.get_logger(__name__)
_MAX_DISPLAY_ISSUES = 50


@dataclass
class _CheckIssue:
    file: str
    line: int
    column: int
    code: str
    message: str
    severity: str = "error"


@dataclass
class _GateExecution:
    result: im.GateResult
    issues: list[_CheckIssue] = field(default_factory=list)
    raw_output: str = ""


@dataclass
class _ProjectResult:
    project: str
    gates: MutableMapping[str, _GateExecution] = field(default_factory=dict)

    @property
    def total_errors(self) -> int:
        return sum(len(v.issues) for v in self.gates.values())

    @property
    def passed(self) -> bool:
        return all(v.result.passed for v in self.gates.values())


@dataclass
class _RunCommandResult:
    stdout: str
    stderr: str
    returncode: int


def _format_issue(issue: _CheckIssue) -> str:
    code_part = f"[{issue.code}] " if issue.code else ""
    return (
        f"{issue.file}:{issue.line}:{issue.column} {code_part}{issue.message}".strip()
    )


class WorkspaceChecker(FlextService[list[_ProjectResult]]):
    """Run quality gates across one or more workspace projects."""

    def __init__(self, workspace_root: Path | None = None) -> None:
        """Initialize checker dependencies and default report directory."""
        super().__init__()
        self._path_resolver = PathResolver()
        self._reporting = ReportingService()
        self._json = JsonService()
        self._runner = CommandRunner()
        self._workspace_root = self._resolve_workspace_root(workspace_root)
        report_dir_result = self._reporting.ensure_report_dir(
            self._workspace_root, "check"
        )
        self._default_reports_dir = (
            report_dir_result.value
            if report_dir_result.is_success
            else self._workspace_root / ".reports" / "check"
        )

    @override
    def execute(self) -> r[list[_ProjectResult]]:
        """Return a failure because this service requires explicit run inputs."""
        return r[list[_ProjectResult]].fail("Use run() or run_projects() directly")

    def run(
        self,
        project: str,
        gates: Sequence[str],
    ) -> r[list[_ProjectResult]]:
        """Run selected gates for a single project."""
        return self.run_projects([project], list(gates)).map(lambda value: value)

    def run_projects(
        self,
        projects: Sequence[str],
        gates: Sequence[str],
        *,
        reports_dir: Path | None = None,
        fail_fast: bool = False,
    ) -> r[list[_ProjectResult]]:
        """Run selected gates for multiple projects and emit reports."""
        resolved_gates_result = self.resolve_gates(gates)
        if resolved_gates_result.is_failure:
            return r[list[_ProjectResult]].fail(
                resolved_gates_result.error or "invalid gates",
            )
        resolved_gates = resolved_gates_result.value

        report_base = reports_dir or self._default_reports_dir
        report_base.mkdir(parents=True, exist_ok=True)

        results: list[_ProjectResult] = []
        total = len(projects)
        failed = 0

        for index, project_name in enumerate(projects, 1):
            project_dir = self._workspace_root / project_name
            pyproject_path = project_dir / ic.Files.PYPROJECT_FILENAME
            if not project_dir.is_dir() or not pyproject_path.exists():
                _ = sys.stdout.write(
                    f"[{index:2d}/{total:2d}] {project_name} ... skipped\n"
                )
                continue

            _ = sys.stdout.write(f"[{index:2d}/{total:2d}] {project_name} ... ")
            _ = sys.stdout.flush()
            project_result = self._check_project(
                project_dir, resolved_gates, report_base
            )
            results.append(project_result)

            if project_result.passed:
                _ = sys.stdout.write("ok\n")
            else:
                counts = " ".join(
                    f"{gate}={len(project_result.gates[gate].issues)}"
                    for gate in resolved_gates
                    if gate in project_result.gates
                    and len(project_result.gates[gate].issues) > 0
                )
                _ = sys.stdout.write(
                    f"FAIL ({project_result.total_errors} errors: {counts})\n"
                )
                failed += 1
                if fail_fast:
                    break

        timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
        md_path = report_base / "check-report.md"
        _ = md_path.write_text(
            self.generate_markdown_report(results, resolved_gates, timestamp),
            encoding=ic.Encoding.DEFAULT,
        )

        sarif_path = report_base / "check-report.sarif"
        sarif_payload = self.generate_sarif_report(results, resolved_gates)
        json_write_result = self._json.write(sarif_path, sarif_payload)
        if json_write_result.is_failure:
            return r[list[_ProjectResult]].fail(
                json_write_result.error or "failed to write sarif report",
            )

        total_errors = sum(project.total_errors for project in results)
        _ = sys.stdout.write(
            f"\n{'=' * 60}\n"
            f"Check: {len(results)} projects, {total_errors} errors, {failed} failed\n"
            f"Reports: {md_path}\n"
            f"         {sarif_path}\n"
            f"{'=' * 60}\n"
        )

        if total_errors > 0:
            _ = sys.stdout.write("\nErrors by project:\n")
            for project in sorted(
                results, key=lambda item: item.total_errors, reverse=True
            ):
                if project.total_errors == 0:
                    continue
                breakdown = ", ".join(
                    f"{gate}={len(project.gates[gate].issues)}"
                    for gate in resolved_gates
                    if gate in project.gates and len(project.gates[gate].issues) > 0
                )
                _ = sys.stdout.write(
                    f"  {project.project:30s} {project.total_errors:6d}  ({breakdown})\n"
                )

        return r[list[_ProjectResult]].ok(results)

    def lint(self, project_dir: Path) -> r[im.GateResult]:
        """Run the Ruff lint gate for a project."""
        return r[im.GateResult].ok(self._run_ruff_lint(project_dir).result)

    def format(self, project_dir: Path) -> r[im.GateResult]:
        """Run the Ruff format check gate for a project."""
        return r[im.GateResult].ok(self._run_ruff_format(project_dir).result)

    def pyrefly(self, project_dir: Path, reports_dir: Path) -> r[im.GateResult]:
        """Run the Pyrefly type-checking gate for a project."""
        return r[im.GateResult].ok(self._run_pyrefly(project_dir, reports_dir).result)

    def mypy(self, project_dir: Path) -> r[im.GateResult]:
        """Run the Mypy gate for a project."""
        return r[im.GateResult].ok(self._run_mypy(project_dir).result)

    def pyright(self, project_dir: Path) -> r[im.GateResult]:
        """Run the Pyright gate for a project."""
        return r[im.GateResult].ok(self._run_pyright(project_dir).result)

    def security(self, project_dir: Path) -> r[im.GateResult]:
        """Run the Bandit security gate for a project."""
        return r[im.GateResult].ok(self._run_bandit(project_dir).result)

    def markdown(self, project_dir: Path) -> r[im.GateResult]:
        """Run the Markdown lint gate for a project."""
        return r[im.GateResult].ok(self._run_markdown(project_dir).result)

    def go(self, project_dir: Path) -> r[im.GateResult]:
        """Run Go vet and gofmt checks for a Go project."""
        return r[im.GateResult].ok(self._run_go(project_dir).result)

    @staticmethod
    def resolve_gates(gates: Sequence[str]) -> r[list[str]]:
        """Validate and normalize user-provided gate names."""
        allowed = {
            ic.Gates.LINT,
            ic.Gates.FORMAT,
            ic.Gates.PYREFLY,
            ic.Gates.MYPY,
            ic.Gates.PYRIGHT,
            ic.Gates.SECURITY,
            ic.Gates.MARKDOWN,
            ic.Gates.GO,
        }
        resolved: list[str] = []
        for gate in gates:
            name = gate.strip()
            if not name:
                continue
            mapped = ic.Gates.PYREFLY if name == ic.Gates.TYPE_ALIAS else name
            if mapped not in allowed:
                return r[list[str]].fail(f"ERROR: unknown gate '{gate}'")
            if mapped not in resolved:
                resolved.append(mapped)
        return r[list[str]].ok(resolved)

    @staticmethod
    def parse_gate_csv(raw: str) -> list[str]:
        """Parse comma-separated gate names into a normalized list."""
        return [gate.strip() for gate in raw.split(",") if gate.strip()]

    def generate_markdown_report(
        self,
        results: list[_ProjectResult],
        gates: list[str],
        timestamp: str,
    ) -> str:
        """Render the workspace check report in Markdown format."""
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
        for gate in gates:
            header += f" {gate.capitalize()} |"
            sep += "------|"
        header += " Total | Status |"
        sep += "-------|--------|"
        lines.extend([header, sep])

        total_all = 0
        failed_count = 0
        for project in results:
            row = f"| {project.project} |"
            for gate in gates:
                gate_result = project.gates.get(gate)
                row += f" {len(gate_result.issues) if gate_result else 0} |"
            status = ic.Status.PASS if project.passed else f"**{ic.Status.FAIL}**"
            if not project.passed:
                failed_count += 1
            row += f" {project.total_errors} | {status} |"
            total_all += project.total_errors
            lines.append(row)

        lines.extend(
            [
                "",
                f"**Total errors**: {total_all}  ",
                f"**Failed projects**: {failed_count}/{len(results)}  ",
                "",
            ],
        )

        for project in sorted(
            results, key=lambda item: item.total_errors, reverse=True
        ):
            if project.total_errors == 0:
                continue
            lines.extend([f"## {project.project}", ""])
            for gate in gates:
                gate_result = project.gates.get(gate)
                if not gate_result or len(gate_result.issues) == 0:
                    continue
                lines.extend([
                    f"### {gate} ({len(gate_result.issues)} errors)",
                    "",
                    "```",
                ])
                lines.extend(_format_issue(issue) for issue in gate_result.issues[:_MAX_DISPLAY_ISSUES])
                if len(gate_result.issues) > _MAX_DISPLAY_ISSUES:
                    lines.append(f"... and {len(gate_result.issues) - _MAX_DISPLAY_ISSUES} more errors")
                lines.extend(["```", ""])

        return "\n".join(lines)

    @staticmethod
    def generate_sarif_report(
        results: list[_ProjectResult],
        gates: list[str],
    ) -> Mapping[str, t.ConfigMapValue]:
        """Render gate results as a SARIF 2.1.0 payload."""
        tool_info = {
            ic.Gates.LINT: ("Ruff Linter", "https://docs.astral.sh/ruff/"),
            ic.Gates.FORMAT: (
                "Ruff Formatter",
                "https://docs.astral.sh/ruff/formatter/",
            ),
            ic.Gates.PYREFLY: ("Pyrefly", "https://github.com/facebook/pyrefly"),
            ic.Gates.MYPY: ("Mypy", "https://mypy.readthedocs.io/"),
            ic.Gates.PYRIGHT: ("Pyright", "https://github.com/microsoft/pyright"),
            ic.Gates.SECURITY: ("Bandit", "https://bandit.readthedocs.io/"),
            ic.Gates.MARKDOWN: (
                "MarkdownLint",
                "https://github.com/DavidAnson/markdownlint",
            ),
            ic.Gates.GO: ("Go Vet", "https://pkg.go.dev/cmd/vet"),
        }

        runs: list[Mapping[str, t.ConfigMapValue]] = []
        for gate in gates:
            tool_name, tool_url = tool_info.get(gate, (gate, ""))
            sarif_results: list[Mapping[str, t.ConfigMapValue]] = []
            rules_seen: set[str] = set()
            rules: list[Mapping[str, t.ConfigMapValue]] = []

            for project in results:
                gate_result = project.gates.get(gate)
                if not gate_result:
                    continue
                for issue in gate_result.issues:
                    rule_id = issue.code or "unknown"
                    if rule_id not in rules_seen:
                        rules_seen.add(rule_id)
                        rules.append(
                            {"id": rule_id, "shortDescription": {"text": rule_id}},
                        )
                    sarif_results.append(
                        {
                            "ruleId": rule_id,
                            "level": (
                                "error" if issue.severity == "error" else "warning"
                            ),
                            "message": {"text": issue.message},
                            "locations": [
                                {
                                    "physicalLocation": {
                                        "artifactLocation": {
                                            "uri": issue.file,
                                            "uriBaseId": "%SRCROOT%",
                                        },
                                        "region": {
                                            "startLine": max(issue.line, 1),
                                            "startColumn": max(issue.column, 1),
                                        },
                                    },
                                },
                            ],
                        },
                    )

            runs.append(
                {
                    "tool": {
                        "driver": {
                            "name": tool_name,
                            "informationUri": tool_url,
                            "rules": rules,
                        },
                    },
                    "results": sarif_results,
                },
            )

        return {
            "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/main/Schemata/sarif-schema-2.1.0.json",
            "version": "2.1.0",
            "runs": runs,
        }

    def _check_project(
        self,
        project_dir: Path,
        gates: list[str],
        reports_dir: Path,
    ) -> _ProjectResult:
        result = _ProjectResult(project=project_dir.name)
        runners: Mapping[str, Callable[[], _GateExecution]] = {
            ic.Gates.LINT: lambda: self._run_ruff_lint(project_dir),
            ic.Gates.FORMAT: lambda: self._run_ruff_format(project_dir),
            ic.Gates.PYREFLY: lambda: self._run_pyrefly(project_dir, reports_dir),
            ic.Gates.MYPY: lambda: self._run_mypy(project_dir),
            ic.Gates.PYRIGHT: lambda: self._run_pyright(project_dir),
            ic.Gates.SECURITY: lambda: self._run_bandit(project_dir),
            ic.Gates.MARKDOWN: lambda: self._run_markdown(project_dir),
            ic.Gates.GO: lambda: self._run_go(project_dir),
        }
        for gate in gates:
            runner = runners.get(gate)
            if runner:
                result.gates[gate] = runner()
        return result

    def _existing_check_dirs(self, project_dir: Path) -> list[str]:
        dirs = (
            ic.Check.DEFAULT_CHECK_DIRS
            if project_dir.resolve() == self._workspace_root.resolve()
            else ic.Check.CHECK_DIRS_SUBPROJECT
        )
        return [directory for directory in dirs if (project_dir / directory).is_dir()]

    @staticmethod
    def _dirs_with_py(project_dir: Path, dirs: list[str]) -> list[str]:
        out: list[str] = []
        for directory in dirs:
            path = project_dir / directory
            if not path.is_dir():
                continue
            if next(path.rglob("*.py"), None) or next(path.rglob("*.pyi"), None):
                out.append(directory)
        return out

    def _run(
        self,
        cmd: list[str],
        cwd: Path,
        timeout: int = 300,
        env: Mapping[str, str] | None = None,
    ) -> _RunCommandResult:
        result = self._runner.run_raw(
            cmd,
            cwd=cwd,
            timeout=timeout,
            env=env,
        )
        if result.is_failure:
            return _RunCommandResult(
                stdout="",
                stderr=result.error or "command execution failed",
                returncode=1,
            )

        output = result.value
        return _RunCommandResult(
            stdout=output.stdout,
            stderr=output.stderr,
            returncode=output.exit_code,
        )

    def _build_gate_result(
        self,
        *,
        gate: str,
        project: str,
        passed: bool,
        issues: list[_CheckIssue],
        duration: float,
        raw_output: str,
    ) -> _GateExecution:
        model = im.GateResult(
            gate=gate,
            project=project,
            passed=passed,
            errors=[_format_issue(issue) for issue in issues],
            duration=round(duration, 3),
        )
        return _GateExecution(result=model, issues=issues, raw_output=raw_output)

    def _run_ruff_lint(self, project_dir: Path) -> _GateExecution:
        started = time.monotonic()
        check_dirs = self._existing_check_dirs(project_dir)
        targets = check_dirs or ["."]
        result = self._run(
            [
                sys.executable,
                "-m",
                "ruff",
                "check",
                *targets,
                "--output-format",
                "json",
                "--quiet",
            ],
            project_dir,
        )
        issues: list[_CheckIssue] = []
        try:
            for error in json.loads(result.stdout or "[]"):
                loc = error.get("location", {})
                issues.append(
                    _CheckIssue(
                        file=error.get("filename", "?"),
                        line=loc.get("row", 0),
                        column=loc.get("column", 0),
                        code=error.get("code", ""),
                        message=error.get("message", ""),
                    ),
                )
        except (json.JSONDecodeError, KeyError):
            pass
        return self._build_gate_result(
            gate="lint",
            project=project_dir.name,
            passed=result.returncode == 0,
            issues=issues,
            duration=time.monotonic() - started,
            raw_output=result.stderr,
        )

    def _run_ruff_format(self, project_dir: Path) -> _GateExecution:
        started = time.monotonic()
        check_dirs = self._existing_check_dirs(project_dir)
        targets = check_dirs or ["."]
        result = self._run(
            [sys.executable, "-m", "ruff", "format", "--check", *targets, "--quiet"],
            project_dir,
        )
        issues: list[_CheckIssue] = []
        if result.returncode != 0 and result.stdout.strip():
            seen: set[str] = set()
            for line in result.stdout.strip().splitlines():
                path = line.strip()
                if not path:
                    continue
                match = _RUFF_FORMAT_FILE_RE.match(path)
                if match:
                    file_path = match.group(1).strip()
                    if file_path in seen:
                        continue
                    seen.add(file_path)
                    issues.append(
                        _CheckIssue(
                            file=file_path,
                            line=0,
                            column=0,
                            code="format",
                            message="Would be reformatted",
                        ),
                    )
                elif path.endswith(".py") and " " not in path and path not in seen:
                    seen.add(path)
                    issues.append(
                        _CheckIssue(
                            file=path,
                            line=0,
                            column=0,
                            code="format",
                            message="Would be reformatted",
                        ),
                    )
        return self._build_gate_result(
            gate="format",
            project=project_dir.name,
            passed=result.returncode == 0,
            issues=issues,
            duration=time.monotonic() - started,
            raw_output=result.stderr,
        )

    def _run_pyrefly(self, project_dir: Path, reports_dir: Path) -> _GateExecution:
        started = time.monotonic()
        check_dirs = self._existing_check_dirs(project_dir)
        targets = check_dirs or [ic.Paths.DEFAULT_SRC_DIR]
        json_file = reports_dir / f"{project_dir.name}-pyrefly.json"
        cmd = [
            sys.executable,
            "-m",
            "pyrefly",
            "check",
            *targets,
            "--config",
            ic.Files.PYPROJECT_FILENAME,
            "--output-format",
            "json",
            "-o",
            str(json_file),
            "--summary=none",
        ]
        result = self._run(cmd, project_dir)
        issues: list[_CheckIssue] = []
        if json_file.exists():
            try:
                data = json.loads(json_file.read_text(encoding=ic.Encoding.DEFAULT))
                raw_errors = data.get("errors", []) if type(data) is dict else data
                issues.extend(
                    _CheckIssue(
                        file=error.get("path", "?"),
                        line=error.get("line", 0),
                        column=error.get("column", 0),
                        code=error.get("name", ""),
                        message=error.get("description", ""),
                        severity=error.get("severity", "error"),
                    )
                    for error in raw_errors
                )
            except (json.JSONDecodeError, KeyError):
                pass

        if not issues and result.returncode != 0:
            match = re.search(r"(\d+)\s+errors?", result.stderr + result.stdout)
            if match:
                count = int(match.group(1))
                issues = [
                    _CheckIssue(
                        file="?",
                        line=0,
                        column=0,
                        code="pyrefly",
                        message=f"Pyrefly reported {count} error(s)",
                    ),
                ] * count

        return self._build_gate_result(
            gate="pyrefly",
            project=project_dir.name,
            passed=result.returncode == 0,
            issues=issues,
            duration=time.monotonic() - started,
            raw_output=result.stderr,
        )

    def _run_mypy(self, project_dir: Path) -> _GateExecution:
        started = time.monotonic()
        check_dirs = self._existing_check_dirs(project_dir)
        mypy_dirs = self._dirs_with_py(project_dir, check_dirs)
        if not mypy_dirs:
            return self._build_gate_result(
                gate="mypy",
                project=project_dir.name,
                passed=True,
                issues=[],
                duration=time.monotonic() - started,
                raw_output="",
            )

        proj_py = project_dir / ic.Files.PYPROJECT_FILENAME
        cfg = (
            proj_py
            if proj_py.exists()
            and "[tool.mypy]" in proj_py.read_text(encoding=ic.Encoding.DEFAULT)
            else self._workspace_root / ic.Files.PYPROJECT_FILENAME
        )
        typings_generated = self._workspace_root / "typings" / "generated"
        mypy_env = os.environ.copy()
        if typings_generated.is_dir():
            existing = mypy_env.get("MYPYPATH", "")
            mypy_env["MYPYPATH"] = str(typings_generated) + (
                f":{existing}" if existing else ""
            )

        result = self._run(
            [
                sys.executable,
                "-m",
                "mypy",
                *mypy_dirs,
                "--config-file",
                str(cfg),
                "--output",
                "json",
            ],
            project_dir,
            env=mypy_env,
        )

        issues: list[_CheckIssue] = []
        for raw_line in (result.stdout or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                error = json.loads(line)
                if error.get("severity") in {"error", "warning", "note"}:
                    issues.append(
                        _CheckIssue(
                            file=error.get("file", "?"),
                            line=error.get("line", 0),
                            column=error.get("column", 0),
                            code=error.get("code", ""),
                            message=error.get("message", ""),
                            severity=error.get("severity", "error"),
                        ),
                    )
            except json.JSONDecodeError:
                continue

        return self._build_gate_result(
            gate="mypy",
            project=project_dir.name,
            passed=result.returncode == 0,
            issues=issues,
            duration=time.monotonic() - started,
            raw_output=result.stderr,
        )

    def _run_pyright(self, project_dir: Path) -> _GateExecution:
        started = time.monotonic()
        check_dirs = self._dirs_with_py(
            project_dir, self._existing_check_dirs(project_dir)
        )
        if not check_dirs:
            return self._build_gate_result(
                gate="pyright",
                project=project_dir.name,
                passed=True,
                issues=[],
                duration=time.monotonic() - started,
                raw_output="",
            )
        result = self._run(
            [sys.executable, "-m", "pyright", *check_dirs, "--outputjson"],
            project_dir,
            timeout=600,
        )
        issues: list[_CheckIssue] = []
        try:
            data = json.loads(result.stdout or "{}")
            for error in data.get("generalDiagnostics", []):
                rng = error.get("range", {}).get("start", {})
                issues.append(
                    _CheckIssue(
                        file=error.get("file", "?"),
                        line=rng.get("line", 0) + 1,
                        column=rng.get("character", 0) + 1,
                        code=error.get("rule", ""),
                        message=error.get("message", ""),
                        severity=error.get("severity", "error"),
                    ),
                )
        except (json.JSONDecodeError, KeyError):
            pass

        return self._build_gate_result(
            gate="pyright",
            project=project_dir.name,
            passed=result.returncode == 0,
            issues=issues,
            duration=time.monotonic() - started,
            raw_output=result.stderr,
        )

    def _run_bandit(self, project_dir: Path) -> _GateExecution:
        started = time.monotonic()
        src_path = project_dir / ic.Paths.DEFAULT_SRC_DIR
        if not src_path.exists():
            return self._build_gate_result(
                gate="security",
                project=project_dir.name,
                passed=True,
                issues=[],
                duration=time.monotonic() - started,
                raw_output="",
            )
        result = self._run(
            [
                "bandit",
                "-r",
                ic.Paths.DEFAULT_SRC_DIR,
                "-f",
                "json",
                "-q",
                "-ll",
            ],
            project_dir,
        )
        issues: list[_CheckIssue] = []
        try:
            data = json.loads(result.stdout or "{}")
            issues.extend(
                _CheckIssue(
                    file=error.get("filename", "?"),
                    line=error.get("line_number", 0),
                    column=0,
                    code=error.get("test_id", ""),
                    message=error.get("issue_text", ""),
                    severity=error.get("issue_severity", "MEDIUM").lower(),
                )
                for error in data.get("results", [])
            )
        except (json.JSONDecodeError, KeyError):
            pass
        return self._build_gate_result(
            gate="security",
            project=project_dir.name,
            passed=result.returncode == 0,
            issues=issues,
            duration=time.monotonic() - started,
            raw_output=result.stderr,
        )

    def _collect_markdown_files(self, project_dir: Path) -> list[Path]:
        files: list[Path] = []
        for path in project_dir.rglob("*.md"):
            if any(part in ic.Excluded.CHECK_EXCLUDED_DIRS for part in path.parts):
                continue
            files.append(path)
        return files

    def _run_markdown(self, project_dir: Path) -> _GateExecution:
        started = time.monotonic()
        md_files = self._collect_markdown_files(project_dir)
        if not md_files:
            return self._build_gate_result(
                gate="markdown",
                project=project_dir.name,
                passed=True,
                issues=[],
                duration=time.monotonic() - started,
                raw_output="",
            )
        cmd = ["markdownlint"]
        root_config = self._workspace_root / ".markdownlint.json"
        local_config = project_dir / ".markdownlint.json"
        if root_config.exists():
            cmd.extend(["--config", str(root_config)])
        elif local_config.exists():
            cmd.extend(["--config", str(local_config)])
        cmd.extend(str(path.relative_to(project_dir)) for path in md_files)
        result = self._run(cmd, project_dir)
        issues: list[_CheckIssue] = []
        for line in (result.stdout + "\n" + result.stderr).splitlines():
            match = _MARKDOWN_RE.match(line.strip())
            if not match:
                continue
            issues.append(
                _CheckIssue(
                    file=match.group("file"),
                    line=int(match.group("line")),
                    column=int(match.group("col") or 1),
                    code=match.group("code"),
                    message=match.group("msg"),
                ),
            )
        if result.returncode != 0 and not issues:
            issues.append(
                _CheckIssue(
                    file=".",
                    line=1,
                    column=1,
                    code="markdownlint",
                    message=(
                        result.stdout or result.stderr or "markdownlint failed"
                    ).strip(),
                ),
            )

        return self._build_gate_result(
            gate="markdown",
            project=project_dir.name,
            passed=result.returncode == 0,
            issues=issues,
            duration=time.monotonic() - started,
            raw_output=result.stderr,
        )

    def _run_go(self, project_dir: Path) -> _GateExecution:
        started = time.monotonic()
        if not (project_dir / "go.mod").exists():
            return self._build_gate_result(
                gate="go",
                project=project_dir.name,
                passed=True,
                issues=[],
                duration=time.monotonic() - started,
                raw_output="",
            )
        issues: list[_CheckIssue] = []
        raw_output = ""

        vet_result = self._run(["go", "vet", "./..."], project_dir, timeout=900)
        raw_output = "\n".join(
            part for part in (vet_result.stdout, vet_result.stderr) if part
        )
        for line in (vet_result.stdout + "\n" + vet_result.stderr).splitlines():
            match = _GO_VET_RE.match(line.strip())
            if not match:
                continue
            issues.append(
                _CheckIssue(
                    file=match.group("file"),
                    line=int(match.group("line")),
                    column=int(match.group("col") or 1),
                    code="govet",
                    message=match.group("msg"),
                ),
            )
        if vet_result.returncode != 0 and not issues:
            issues.append(
                _CheckIssue(
                    file=".",
                    line=1,
                    column=1,
                    code="govet",
                    message=(
                        vet_result.stdout or vet_result.stderr or "go vet failed"
                    ).strip(),
                ),
            )

        go_files = list(project_dir.rglob("*.go"))
        if go_files:
            fmt_result = self._run(
                [
                    "gofmt",
                    "-l",
                    *[str(path.relative_to(project_dir)) for path in go_files],
                ],
                project_dir,
                timeout=900,
            )
            fmt_raw_output = "\n".join(
                part for part in (fmt_result.stdout, fmt_result.stderr) if part
            )
            raw_output = "\n".join(
                part for part in (raw_output, fmt_raw_output) if part
            )
            for file_name in fmt_result.stdout.splitlines():
                cleaned = file_name.strip()
                if not cleaned:
                    continue
                issues.append(
                    _CheckIssue(
                        file=cleaned,
                        line=1,
                        column=1,
                        code="gofmt",
                        message="File is not gofmt-formatted",
                    ),
                )

        return self._build_gate_result(
            gate="go",
            project=project_dir.name,
            passed=len(issues) == 0,
            issues=issues,
            duration=time.monotonic() - started,
            raw_output=raw_output,
        )

    def _resolve_workspace_root(self, workspace_root: Path | None) -> Path:
        if workspace_root is not None:
            return workspace_root.resolve()
        result = self._path_resolver.workspace_root()
        return result.value if result.is_success else Path.cwd().resolve()


class PyreflyConfigFixer(FlextService[list[str]]):
    """Repair workspace and project pyrefly configuration blocks."""

    def __init__(self, workspace_root: Path | None = None) -> None:
        """Initialize fixer dependencies and resolve workspace root."""
        super().__init__()
        self._path_resolver = PathResolver()
        self._discovery = DiscoveryService()
        self._workspace_root = self._resolve_workspace_root(workspace_root)

    @override
    def execute(self) -> r[list[str]]:
        """Return a failure because this service requires explicit run inputs."""
        return r[list[str]].fail("Use run() directly")

    def run(
        self,
        projects: Sequence[str],
        *,
        dry_run: bool = False,
        verbose: bool = False,
    ) -> r[list[str]]:
        """Apply pyrefly config fixes for selected projects."""
        project_paths = [self._resolve_project_path(project) for project in projects]
        files_result = self.find_pyproject_files(project_paths or None)
        if files_result.is_failure:
            return r[list[str]].fail(
                files_result.error or "failed to find pyproject files"
            )

        messages: list[str] = []
        total_fixes = 0
        for path in files_result.value:
            fixes_result = self.process_file(path, dry_run=dry_run)
            if fixes_result.is_failure:
                return r[list[str]].fail(
                    fixes_result.error or f"failed to process {path}",
                )
            fixes = fixes_result.value
            if not fixes:
                continue
            total_fixes += len(fixes)
            if verbose:
                try:
                    rel = path.relative_to(self._workspace_root)
                except ValueError:
                    rel = path
                for fix in fixes:
                    line = f"  {'(dry)' if dry_run else '✓'} {rel}: {fix}"
                    logger.info("pyrefly_config_fix", detail=line)
                    messages.append(line)

        if verbose and total_fixes == 0:
            logger.info("pyrefly_configs_clean")

        return r[list[str]].ok(messages)

    def find_pyproject_files(
        self,
        project_paths: list[Path] | None = None,
    ) -> r[list[Path]]:
        """Find pyproject.toml files in workspace or selected project paths."""
        return self._discovery.find_all_pyproject_files(
            self._workspace_root,
            project_paths=project_paths,
        )

    def process_file(self, path: Path, *, dry_run: bool = False) -> r[list[str]]:
        """Apply all pyrefly block fixes to a single pyproject.toml file."""
        try:
            text = path.read_text(encoding=ic.Encoding.DEFAULT)
        except OSError as exc:
            return r[list[str]].fail(f"failed to read {path}: {exc}")

        if "[tool.pyrefly]" not in text:
            return r[list[str]].ok([])

        original = text
        all_fixes: list[str] = []

        text, fixes = self._fix_search_paths(text, path.parent)
        all_fixes.extend(fixes)

        text, fixes = self._remove_ignore_sub_config(text)
        all_fixes.extend(fixes)

        if (
            any("removed ignore" in item for item in all_fixes)
            or path.parent == self._workspace_root
        ):
            text, fixes = self._ensure_project_excludes(text)
            all_fixes.extend(fixes)

        if text != original and not dry_run:
            try:
                _ = path.write_text(text, encoding=ic.Encoding.DEFAULT)
            except OSError as exc:
                return r[list[str]].fail(f"failed to write {path}: {exc}")

        return r[list[str]].ok(all_fixes)

    def _resolve_project_path(self, raw: str) -> Path:
        path = Path(raw)
        if not path.is_absolute():
            path = self._workspace_root / path
        return path.resolve()

    def _fix_search_paths(self, text: str, project_dir: Path) -> tuple[str, list[str]]:
        fixes: list[str] = []

        if project_dir == self._workspace_root:
            if '"../typings/generated"' in text:
                text = text.replace('"../typings/generated"', '"typings/generated"')
                fixes.append("search-path ../typings/generated -> typings/generated")
            if '"../typings"' in text:
                text = text.replace('"../typings"', '"typings"')
                fixes.append("search-path ../typings -> typings")

        sp_match = re.search(r"(search-path\s*=\s*\[)(.*?)(\])", text, flags=re.DOTALL)
        if sp_match:
            original_entries = sp_match.group(2)
            entries = re.findall(r'"([^"]+)"', original_entries)
            nonexistent = [
                entry for entry in entries if not (project_dir / entry).exists()
            ]
            if nonexistent:
                new_entries = original_entries
                for entry in nonexistent:
                    new_entries = re.sub(
                        rf'[ \t]*"{re.escape(entry)}"\s*,?\s*\n',
                        "",
                        new_entries,
                    )
                if new_entries != original_entries:
                    text = (
                        text[: sp_match.start(2)]
                        + new_entries
                        + text[sp_match.end(2) :]
                    )
                    fixes.append(
                        f"removed nonexistent search-path: {', '.join(nonexistent)}",
                    )

        return text, fixes

    @staticmethod
    def _remove_ignore_sub_config(text: str) -> tuple[str, list[str]]:
        fixes: list[str] = []
        pattern = re.compile(
            r"\s*\[\[tool\.pyrefly\.sub-config\]\]\s*\n"
            r"\s*matches\s*=\s*\"([^\"]+)\"\s*\n"
            r"\s*ignore\s*=\s*true\s*\n?",
            flags=re.MULTILINE,
        )
        match = pattern.search(text)
        if match:
            text = text[: match.start()] + text[match.end() :]
            text = re.sub(r"\n{3,}", "\n\n", text)
            fixes.append(f"removed ignore=true sub-config for '{match.group(1)}'")
        return text, fixes

    @staticmethod
    def _ensure_project_excludes(text: str) -> tuple[str, list[str]]:
        fixes: list[str] = []
        pe_match = re.search(
            r"(project-excludes\s*=\s*\[)(.*?)(\])",
            text,
            flags=re.DOTALL,
        )
        if pe_match:
            existing = pe_match.group(2)
            to_add = [glob for glob in _REQUIRED_EXCLUDES if glob not in existing]
            if to_add:
                new_content = existing.rstrip()
                sep = ", " if new_content.strip() else ""
                new_content += sep + ", ".join(to_add) + " "
                text = text[: pe_match.start(2)] + new_content + text[pe_match.end(2) :]
                fixes.append(f"added {', '.join(to_add)} to project-excludes")
        else:
            sp_end = re.search(
                r"([ \t]*)search-path\s*=\s*\[.*?\]\s*\n",
                text,
                flags=re.DOTALL,
            )
            if sp_end:
                indent = sp_end.group(1)
                line = f"{indent}project-excludes = [{', '.join(_REQUIRED_EXCLUDES)}]\n"
                text = text[: sp_end.end()] + line + text[sp_end.end() :]
                fixes.append("added project-excludes for pb2 files")
        return text, fixes

    def _resolve_workspace_root(self, workspace_root: Path | None) -> Path:
        if workspace_root is not None:
            return workspace_root.resolve()
        result = self._path_resolver.workspace_root()
        return result.value if result.is_success else Path.cwd().resolve()


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for check and pyrefly-fix commands."""
    parser = argparse.ArgumentParser(description="FLEXT check utilities")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run quality gates")
    _ = run_parser.add_argument("--gates", default=DEFAULT_GATES)
    _ = run_parser.add_argument("--project", action="append", required=True)
    _ = run_parser.add_argument("--reports-dir", default=".reports/check")
    _ = run_parser.add_argument("--fail-fast", action="store_true")

    fix_parser = subparsers.add_parser(
        "fix-pyrefly-config",
        help="Repair [tool.pyrefly] blocks",
    )
    _ = fix_parser.add_argument("projects", nargs="*")
    _ = fix_parser.add_argument("--dry-run", action="store_true")
    _ = fix_parser.add_argument("--verbose", action="store_true")

    return parser


def run_cli(argv: list[str] | None = None) -> int:
    """Execute check service CLI commands and return process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        checker = WorkspaceChecker()
        gates = WorkspaceChecker.parse_gate_csv(args.gates)
        reports_dir = Path(args.reports_dir).expanduser()
        if not reports_dir.is_absolute():
            reports_dir = (Path.cwd() / reports_dir).resolve()
        run_result = checker.run_projects(
            projects=args.project,
            gates=gates,
            reports_dir=reports_dir,
            fail_fast=args.fail_fast,
        )
        if run_result.is_failure:
            _ = sys.stderr.write(f"{run_result.error or 'check failed'}\n")
            return 2
        failed_projects = [
            project for project in run_result.value if not project.passed
        ]
        return 1 if failed_projects else 0

    if args.command == "fix-pyrefly-config":
        fixer = PyreflyConfigFixer()
        fix_result = fixer.run(
            projects=args.projects,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
        if fix_result.is_failure:
            _ = sys.stderr.write(f"{fix_result.error or 'pyrefly config fix failed'}\n")
            return 1
        return 0

    parser.print_help()
    return 1


__all__ = [
    "DEFAULT_GATES",
    "PyreflyConfigFixer",
    "WorkspaceChecker",
    "run_cli",
]
