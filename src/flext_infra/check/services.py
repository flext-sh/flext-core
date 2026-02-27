"""Quality gate execution and workspace checking services."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import override

import tomlkit
from flext_core.loggings import FlextLogger
from flext_core.result import r
from flext_core.service import FlextService
from flext_core.typings import t
from pydantic import BaseModel, ConfigDict, Field, ValidationError, computed_field

from flext_infra.constants import c
from flext_infra.discovery import DiscoveryService
from flext_infra.json_io import JsonService
from flext_infra.models import m
from flext_infra.output import output
from flext_infra.paths import PathResolver
from flext_infra.reporting import REPORTS_DIR_NAME, ReportingService
from flext_infra.subprocess import CommandRunner

DEFAULT_GATES = c.Gates.DEFAULT_CSV
_REQUIRED_EXCLUDES = ["**/*_pb2*.py", "**/*_pb2_grpc*.py"]
_RUFF_FORMAT_FILE_RE = re.compile(r"^\s*-->\s*(.+?):\d+:\d+\s*$")
_MARKDOWN_RE = re.compile(
    r"^(?P<file>.*?):(?P<line>\d+)(?::(?P<col>\d+))?\s+error\s+"
    r"(?P<code>MD\d+)(?:/[^\s]+)?\s+(?P<msg>.*)$",
)
_GO_VET_RE = re.compile(
    r"^(?P<file>[^:\n]+\.go):(?P<line>\d+)(?::(?P<col>\d+))?:\s*(?P<msg>.*)$",
)
_logger = FlextLogger.create_module_logger(__name__)
_MAX_DISPLAY_ISSUES = 50


class _CheckIssue(BaseModel):
    """Single issue reported by a quality gate tool."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    file: str = Field(description="Source file path")
    line: int = Field(description="Line number")
    column: int = Field(description="Column number")
    code: str = Field(description="Rule or error code")
    message: str = Field(description="Human-readable issue description")
    severity: str = Field(default="error", description="Issue severity level")

    @computed_field    @property
    def formatted(self) -> str:
        """Format issue as ``file:line:col [code] message``."""
        code_part = f"[{self.code}] " if self.code else ""
        return (
            f"{self.file}:{self.line}:{self.column} {code_part}{self.message}".strip()
        )


class _GateExecution(BaseModel):
    """Execution result for a single quality gate."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    result: m.GateResult = Field(description="Gate result model")
    issues: list[_CheckIssue] = Field(
        default_factory=list, description="Detected issues"
    )
    raw_output: str = Field(default="", description="Raw tool output")


class _ProjectResult(BaseModel):
    """Aggregated gate results for a single project."""

    model_config = ConfigDict(extra="forbid")

    project: str = Field(description="Project name")
    gates: dict[str, _GateExecution] = Field(
        default_factory=dict, description="Gate name to execution mapping"
    )

    @computed_field    @property
    def total_errors(self) -> int:
        """Total issue count across all gates."""
        return sum(len(v.issues) for v in self.gates.values())

    @computed_field    @property
    def passed(self) -> bool:
        """Whether every gate passed."""
        return all(v.result.passed for v in self.gates.values())


class _RunCommandResult(BaseModel):
    """Subprocess execution result."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    stdout: str = Field(description="Captured standard output")
    stderr: str = Field(description="Captured standard error")
    returncode: int = Field(description="Process exit code")


# -- Tool-specific JSON parsing models --


class _RuffLintLocation(BaseModel):
    """Location block inside a Ruff lint JSON entry."""

    model_config = ConfigDict(extra="ignore")

    row: int = Field(default=0, description="Line number")
    column: int = Field(default=0, description="Column number")


class _RuffLintError(BaseModel):
    """Single Ruff lint error from JSON output."""

    model_config = ConfigDict(extra="ignore")

    filename: str = Field(default="?", description="Source file path")
    location: _RuffLintLocation = Field(
        default_factory=_RuffLintLocation, description="Error location"
    )
    code: str = Field(default="", description="Ruff rule code")
    message: str = Field(default="", description="Error description")


class _PyreflyError(BaseModel):
    """Single Pyrefly error entry."""

    model_config = ConfigDict(extra="ignore")

    path: str = Field(default="?", description="Source file path")
    line: int = Field(default=0, description="Line number")
    column: int = Field(default=0, description="Column number")
    name: str = Field(default="", description="Error name/code")
    description: str = Field(default="", description="Error description")
    severity: str = Field(default="error", description="Severity level")


class _PyreflyOutput(BaseModel):
    """Pyrefly JSON output wrapper."""

    model_config = ConfigDict(extra="ignore")

    errors: list[_PyreflyError] = Field(
        default_factory=list, description="Pyrefly errors"
    )


class _MypyJsonError(BaseModel):
    """Single mypy JSON error entry."""

    model_config = ConfigDict(extra="ignore")

    file: str = Field(default="?", description="Source file path")
    line: int = Field(default=0, description="Line number")
    column: int = Field(default=0, description="Column number")
    code: str = Field(default="", description="Mypy error code")
    message: str = Field(default="", description="Error description")
    severity: str = Field(default="error", description="Severity level")


class _PyrightPosition(BaseModel):
    """Pyright position with zero-based line/character."""

    model_config = ConfigDict(extra="ignore")

    line: int = Field(default=0, description="Zero-based line number")
    character: int = Field(default=0, description="Zero-based character offset")


class _PyrightRange(BaseModel):
    """Pyright range with start position."""

    model_config = ConfigDict(extra="ignore")

    start: _PyrightPosition = Field(
        default_factory=_PyrightPosition, description="Range start"
    )


class _PyrightDiagnostic(BaseModel):
    """Single Pyright diagnostic entry."""

    model_config = ConfigDict(extra="ignore")

    file: str = Field(default="?", description="Source file path")
    range: _PyrightRange = Field(
        default_factory=_PyrightRange, description="Diagnostic range"
    )
    rule: str = Field(default="", description="Pyright rule name")
    message: str = Field(default="", description="Diagnostic message")
    severity: str = Field(default="error", description="Severity level")


class _PyrightOutput(BaseModel):
    """Pyright JSON output wrapper."""

    model_config = ConfigDict(extra="ignore")

    generalDiagnostics: list[_PyrightDiagnostic] = Field(        default_factory=list, description="General diagnostics list"
    )


class _BanditIssue(BaseModel):
    """Single Bandit security finding."""

    model_config = ConfigDict(extra="ignore")

    filename: str = Field(default="?", description="Source file path")
    line_number: int = Field(default=0, description="Line number")
    test_id: str = Field(default="", description="Bandit test ID")
    issue_text: str = Field(default="", description="Issue description")
    issue_severity: str = Field(default="MEDIUM", description="Severity level")


class _BanditOutput(BaseModel):
    """Bandit JSON output wrapper."""

    model_config = ConfigDict(extra="ignore")

    results: list[_BanditIssue] = Field(
        default_factory=list, description="Bandit findings"
    )


# -- SARIF 2.1.0 report models --


class _SarifMessageText(BaseModel):
    """SARIF message with text content."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(description="Message text")


class _SarifRuleDescriptor(BaseModel):
    """SARIF rule descriptor with short description."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Rule identifier")
    shortDescription: _SarifMessageText = Field(description="Rule short description")

class _SarifArtifactLocation(BaseModel):
    """SARIF artifact location with URI."""

    model_config = ConfigDict(extra="forbid")

    uri: str = Field(description="Artifact URI")
    uriBaseId: str = Field(default="%SRCROOT%", description="URI base identifier")

class _SarifRegion(BaseModel):
    """SARIF region with start line/column."""

    model_config = ConfigDict(extra="forbid")

    startLine: int = Field(description="Start line (1-based)")    startColumn: int = Field(description="Start column (1-based)")

class _SarifPhysicalLocation(BaseModel):
    """SARIF physical location combining artifact and region."""

    model_config = ConfigDict(extra="forbid")

    artifactLocation: _SarifArtifactLocation = Field(description="Artifact location")    region: _SarifRegion = Field(description="Source region")


class _SarifLocation(BaseModel):
    """SARIF location wrapper."""

    model_config = ConfigDict(extra="forbid")

    physicalLocation: _SarifPhysicalLocation = Field(description="Physical location")

class _SarifResult(BaseModel):
    """SARIF result entry."""

    model_config = ConfigDict(extra="forbid")

    ruleId: str = Field(description="Rule identifier")    level: str = Field(description="Result level (error/warning)")
    message: _SarifMessageText = Field(description="Result message")
    locations: list[_SarifLocation] = Field(description="Result locations")


class _SarifToolDriver(BaseModel):
    """SARIF tool driver metadata."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Tool name")
    informationUri: str = Field(default="", description="Tool documentation URL")    rules: list[_SarifRuleDescriptor] = Field(
        default_factory=list, description="Rule descriptors"
    )


class _SarifTool(BaseModel):
    """SARIF tool wrapper."""

    model_config = ConfigDict(extra="forbid")

    driver: _SarifToolDriver = Field(description="Tool driver")


class _SarifRun(BaseModel):
    """SARIF run entry."""

    model_config = ConfigDict(extra="forbid")

    tool: _SarifTool = Field(description="Tool information")
    results: list[_SarifResult] = Field(default_factory=list, description="Run results")


class _SarifReport(BaseModel):
    """Complete SARIF 2.1.0 report."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_uri: str = Field(
        default="https://raw.githubusercontent.com/oasis-tcs/sarif-spec/main/Schemata/sarif-schema-2.1.0.json",
        alias="$schema",
        description="SARIF schema URI",
    )
    version: str = Field(default="2.1.0", description="SARIF version")
    runs: list[_SarifRun] = Field(default_factory=list, description="SARIF runs")


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
            else self._workspace_root / REPORTS_DIR_NAME / "check"
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
        skipped = 0
        loop_start = time.monotonic()

        for index, project_name in enumerate(projects, 1):
            project_dir = self._workspace_root / project_name
            pyproject_path = project_dir / c.Files.PYPROJECT_FILENAME
            if not project_dir.is_dir() or not pyproject_path.exists():
                output.progress(index, total, project_name, "skip")
                skipped += 1
                continue

            output.progress(index, total, project_name, "check")
            start = time.monotonic()
            project_result = self._check_project(
                project_dir, resolved_gates, report_base
            )
            elapsed = time.monotonic() - start
            results.append(project_result)

            if project_result.passed:
                output.status("check", project_name, True, elapsed)
            else:
                output.status("check", project_name, False, elapsed)
                failed += 1
                if fail_fast:
                    break

        total_elapsed = time.monotonic() - loop_start

        timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
        md_path = report_base / "check-report.md"
        _ = md_path.write_text(
            self.generate_markdown_report(results, resolved_gates, timestamp),
            encoding=c.Encoding.DEFAULT,
        )

        sarif_path = report_base / "check-report.sarif"
        sarif_payload = self.generate_sarif_report(results, resolved_gates)
        json_write_result = self._json.write(sarif_path, sarif_payload)
        if json_write_result.is_failure:
            return r[list[_ProjectResult]].fail(
                json_write_result.error or "failed to write sarif report",
            )

        total_errors = sum(project.total_errors for project in results)
        success = len(results) - failed
        output.summary("check", len(results), success, failed, skipped, total_elapsed)
        output.info(f"Reports: {md_path}")
        output.info(f"         {sarif_path}")

        if total_errors > 0:
            output.info("Errors by project:")
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
                output.error(
                    f"{project.project:30s} {project.total_errors:6d}  ({breakdown})"
                )

        return r[list[_ProjectResult]].ok(results)

    def lint(self, project_dir: Path) -> r[m.GateResult]:
        """Run the Ruff lint gate for a project."""
        return r[m.GateResult].ok(self._run_ruff_lint(project_dir).result)

    def format(self, project_dir: Path) -> r[m.GateResult]:
        """Run the Ruff format check gate for a project."""
        return r[m.GateResult].ok(self._run_ruff_format(project_dir).result)

    @staticmethod
    def resolve_gates(gates: Sequence[str]) -> r[list[str]]:
        """Validate and normalize user-provided gate names."""
        allowed = {
            c.Gates.LINT,
            c.Gates.FORMAT,
            c.Gates.PYREFLY,
            c.Gates.MYPY,
            c.Gates.PYRIGHT,
            c.Gates.SECURITY,
            c.Gates.MARKDOWN,
            c.Gates.GO,
        }
        resolved: list[str] = []
        for gate in gates:
            name = gate.strip()
            if not name:
                continue
            mapped = c.Gates.PYREFLY if name == c.Gates.TYPE_ALIAS else name
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
            status = c.Status.PASS if project.passed else f"**{c.Status.FAIL}**"
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
                lines.extend(
                    issue.formatted
                    for issue in gate_result.issues[:_MAX_DISPLAY_ISSUES]
                )
                if len(gate_result.issues) > _MAX_DISPLAY_ISSUES:
                    lines.append(
                        f"... and {len(gate_result.issues) - _MAX_DISPLAY_ISSUES} more errors"
                    )
                lines.extend(["```", ""])

        return "\n".join(lines)

    @staticmethod
    def generate_sarif_report(
        results: list[_ProjectResult],
        gates: list[str],
    ) -> Mapping[str, t.ConfigMapValue]:
        """Render gate results as a SARIF 2.1.0 payload."""
        tool_info = {
            c.Gates.LINT: ("Ruff Linter", "https://docs.astral.sh/ruff/"),
            c.Gates.FORMAT: (
                "Ruff Formatter",
                "https://docs.astral.sh/ruff/formatter/",
            ),
            c.Gates.PYREFLY: ("Pyrefly", "https://github.com/facebook/pyrefly"),
            c.Gates.MYPY: ("Mypy", "https://mypy.readthedocs.io/"),
            c.Gates.PYRIGHT: ("Pyright", "https://github.com/microsoft/pyright"),
            c.Gates.SECURITY: ("Bandit", "https://bandit.readthedocs.io/"),
            c.Gates.MARKDOWN: (
                "MarkdownLint",
                "https://github.com/DavidAnson/markdownlint",
            ),
            c.Gates.GO: ("Go Vet", "https://pkg.go.dev/cmd/vet"),
        }

        sarif_runs: list[_SarifRun] = []
        for gate in gates:
            tool_name, tool_url = tool_info.get(gate, (gate, ""))
            sarif_results: list[_SarifResult] = []
            rules_seen: set[str] = set()
            rules: list[_SarifRuleDescriptor] = []

            for project in results:
                gate_result = project.gates.get(gate)
                if not gate_result:
                    continue
                for issue in gate_result.issues:
                    rule_id = issue.code or "unknown"
                    if rule_id not in rules_seen:
                        rules_seen.add(rule_id)
                        rules.append(
                            _SarifRuleDescriptor(
                                id=rule_id,
                                shortDescription=_SarifMessageText(text=rule_id),
                            ),
                        )
                    sarif_results.append(
                        _SarifResult(
                            ruleId=rule_id,
                            level=("error" if issue.severity == "error" else "warning"),
                            message=_SarifMessageText(text=issue.message),
                            locations=[
                                _SarifLocation(
                                    physicalLocation=_SarifPhysicalLocation(
                                        artifactLocation=_SarifArtifactLocation(
                                            uri=issue.file,
                                        ),
                                        region=_SarifRegion(
                                            startLine=max(issue.line, 1),
                                            startColumn=max(issue.column, 1),
                                        ),
                                    ),
                                ),
                            ],
                        ),
                    )

            sarif_runs.append(
                _SarifRun(
                    tool=_SarifTool(
                        driver=_SarifToolDriver(
                            name=tool_name,
                            informationUri=tool_url,
                            rules=rules,
                        ),
                    ),
                    results=sarif_results,
                ),
            )

        return _SarifReport(runs=sarif_runs).model_dump(by_alias=True)

    def _check_project(
        self,
        project_dir: Path,
        gates: list[str],
        reports_dir: Path,
    ) -> _ProjectResult:
        result = _ProjectResult(project=project_dir.name)
        runners: Mapping[str, Callable[[], _GateExecution]] = {
            c.Gates.LINT: lambda: self._run_ruff_lint(project_dir),
            c.Gates.FORMAT: lambda: self._run_ruff_format(project_dir),
            c.Gates.PYREFLY: lambda: self._run_pyrefly(project_dir, reports_dir),
            c.Gates.MYPY: lambda: self._run_mypy(project_dir),
            c.Gates.PYRIGHT: lambda: self._run_pyright(project_dir),
            c.Gates.SECURITY: lambda: self._run_bandit(project_dir),
            c.Gates.MARKDOWN: lambda: self._run_markdown(project_dir),
            c.Gates.GO: lambda: self._run_go(project_dir),
        }
        for gate in gates:
            runner = runners.get(gate)
            if runner:
                result.gates[gate] = runner()
        return result

    def _existing_check_dirs(self, project_dir: Path) -> list[str]:
        dirs = (
            c.Check.DEFAULT_CHECK_DIRS
            if project_dir.resolve() == self._workspace_root.resolve()
            else c.Check.CHECK_DIRS_SUBPROJECT
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
        model = m.GateResult(
            gate=gate,
            project=project,
            passed=passed,
            errors=[issue.formatted for issue in issues],
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
            for entry in json.loads(result.stdout or "[]"):
                parsed = _RuffLintError.model_validate(entry)
                issues.append(
                    _CheckIssue(
                        file=parsed.filename,
                        line=parsed.location.row,
                        column=parsed.location.column,
                        code=parsed.code,
                        message=parsed.message,
                    ),
                )
        except (json.JSONDecodeError, ValidationError):
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
        targets = check_dirs or [c.Paths.DEFAULT_SRC_DIR]
        json_file = reports_dir / f"{project_dir.name}-pyrefly.json"
        cmd = [
            sys.executable,
            "-m",
            "pyrefly",
            "check",
            *targets,
            "--config",
            c.Files.PYPROJECT_FILENAME,
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
                raw = json.loads(json_file.read_text(encoding=c.Encoding.DEFAULT))
                if isinstance(raw, dict):
                    output = _PyreflyOutput.model_validate(raw)
                    pyrefly_errors = output.errors
                else:
                    pyrefly_errors = [
                        _PyreflyError.model_validate(item) for item in raw
                    ]
                issues.extend(
                    _CheckIssue(
                        file=err.path,
                        line=err.line,
                        column=err.column,
                        code=err.name,
                        message=err.description,
                        severity=err.severity,
                    )
                    for err in pyrefly_errors
                )
            except (json.JSONDecodeError, ValidationError):
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

        proj_py = project_dir / c.Files.PYPROJECT_FILENAME
        cfg = (
            proj_py
            if proj_py.exists()
            and "[tool.mypy]" in proj_py.read_text(encoding=c.Encoding.DEFAULT)
            else self._workspace_root / c.Files.PYPROJECT_FILENAME
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
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                parsed = _MypyJsonError.model_validate(json.loads(stripped))
                if parsed.severity in {"error", "warning", "note"}:
                    issues.append(
                        _CheckIssue(
                            file=parsed.file,
                            line=parsed.line,
                            column=parsed.column,
                            code=parsed.code,
                            message=parsed.message,
                            severity=parsed.severity,
                        ),
                    )
            except (json.JSONDecodeError, ValidationError):
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
            output = _PyrightOutput.model_validate(json.loads(result.stdout or "{}"))
            issues.extend(
                _CheckIssue(
                    file=diag.file,
                    line=diag.range.start.line + 1,
                    column=diag.range.start.character + 1,
                    code=diag.rule,
                    message=diag.message,
                    severity=diag.severity,
                )
                for diag in output.generalDiagnostics
            )
        except (json.JSONDecodeError, ValidationError):
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
        src_path = project_dir / c.Paths.DEFAULT_SRC_DIR
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
                sys.executable,
                "-m",
                "bandit",
                "-r",
                c.Paths.DEFAULT_SRC_DIR,
                "-f",
                "json",
                "-q",
                "-ll",
            ],
            project_dir,
        )
        issues: list[_CheckIssue] = []
        try:
            output = _BanditOutput.model_validate(json.loads(result.stdout or "{}"))
            issues.extend(
                _CheckIssue(
                    file=finding.filename,
                    line=finding.line_number,
                    column=0,
                    code=finding.test_id,
                    message=finding.issue_text,
                    severity=finding.issue_severity.lower(),
                )
                for finding in output.results
            )
        except (json.JSONDecodeError, ValidationError):
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
            if any(part in c.Excluded.CHECK_EXCLUDED_DIRS for part in path.parts):
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
                    _logger.info("pyrefly_config_fix", detail=line)
                    messages.append(line)

        if verbose and total_fixes == 0:
            _logger.info("pyrefly_configs_clean")

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
            text = path.read_text(encoding=c.Encoding.DEFAULT)
            doc = tomlkit.parse(text)
        except OSError as exc:
            return r[list[str]].fail(f"failed to read {path}: {exc}")
        except Exception as exc:
            return r[list[str]].fail(f"failed to parse {path}: {exc}")

        tool = doc.get("tool")
        if not isinstance(tool, Mapping) or "pyrefly" not in tool:
            return r[list[str]].ok([])

        pyrefly = tool["pyrefly"]
        if not isinstance(pyrefly, MutableMapping):
            return r[list[str]].ok([])

        all_fixes: list[str] = []

        # 1. Fix search paths
        fixes = self._fix_search_paths_tk(pyrefly, path.parent)
        all_fixes.extend(fixes)

        # 2. Remove ignore=true sub-configs
        fixes = self._remove_ignore_sub_config_tk(pyrefly)
        all_fixes.extend(fixes)

        # 3. Ensure project excludes
        if (
            any("removed ignore" in item for item in all_fixes)
            or path.parent == self._workspace_root
        ):
            fixes = self._ensure_project_excludes_tk(pyrefly)
            all_fixes.extend(fixes)

        if all_fixes and not dry_run:
            try:
                new_text = tomlkit.dumps(doc)
                _ = path.write_text(new_text, encoding=c.Encoding.DEFAULT)
            except OSError as exc:
                return r[list[str]].fail(f"failed to write {path}: {exc}")

        return r[list[str]].ok(all_fixes)

    def _fix_search_paths_tk(
        self, pyrefly: MutableMapping[str, t.ConfigMapValue], project_dir: Path
    ) -> list[str]:
        fixes: list[str] = []
        search_path = pyrefly.get("search-path")

        if not isinstance(search_path, list):
            return []

        # Normalize paths for root
        if project_dir == self._workspace_root:
            new_paths = []
            for p in search_path:
                if p == "../typings/generated":
                    new_paths.append("typings/generated")
                    fixes.append(
                        "search-path ../typings/generated -> typings/generated"
                    )
                elif p == "../typings":
                    new_paths.append("typings")
                    fixes.append("search-path ../typings -> typings")
                else:
                    new_paths.append(p)

            if fixes:
                pyrefly["search-path"] = self._to_array(new_paths)

        # Remove nonexistent paths
        current_paths = list(pyrefly.get("search-path", []))
        nonexistent = [
            p
            for p in current_paths
            if isinstance(p, str) and not (project_dir / p).exists()
        ]
        if nonexistent:
            remaining = [p for p in current_paths if p not in nonexistent]
            pyrefly["search-path"] = self._to_array(remaining)
            fixes.append(f"removed nonexistent search-path: {', '.join(nonexistent)}")

        return fixes

    def _remove_ignore_sub_config_tk(
        self, pyrefly: MutableMapping[str, t.ConfigMapValue]
    ) -> list[str]:
        fixes: list[str] = []
        sub_configs = pyrefly.get("sub-config")
        if not isinstance(sub_configs, list):
            return []

        new_configs = []
        for conf in sub_configs:
            if isinstance(conf, Mapping) and conf.get("ignore") is True:
                matches = conf.get("matches", "unknown")
                fixes.append(f"removed ignore=true sub-config for '{matches}'")
                continue
            new_configs.append(conf)

        if len(new_configs) != len(sub_configs):
            pyrefly["sub-config"] = new_configs

        return fixes

    def _ensure_project_excludes_tk(
        self, pyrefly: MutableMapping[str, t.ConfigMapValue]
    ) -> list[str]:
        fixes: list[str] = []
        excludes = pyrefly.get("project-excludes")

        current = []
        if isinstance(excludes, list):
            current = [str(x) for x in excludes]

        # Check without quotes too just in case
        stripped_to_add = []
        for glob in _REQUIRED_EXCLUDES:
            clean_glob = glob.strip('"').strip("'")
            if clean_glob not in current and glob not in current:
                stripped_to_add.append(clean_glob)

        if stripped_to_add:
            updated = sorted(set(current) | set(stripped_to_add))
            pyrefly["project-excludes"] = self._to_array(updated)
            fixes.append(f"added {', '.join(stripped_to_add)} to project-excludes")

        return fixes

    @staticmethod
    def _to_array(items: list[str]) -> tomlkit.items.Array:
        arr = tomlkit.array()
        for item in items:
            arr.append(item)
        if len(items) > 1:
            arr.multiline(True)
        return arr

    def _resolve_project_path(self, raw: str) -> Path:
        path = Path(raw)
        if not path.is_absolute():
            path = self._workspace_root / path
        return path.resolve()

    # Legacy regex-based methods (kept for reference or if still called, but preferred tk versions above)
    def _fix_search_paths(self, text: str) -> tuple[str, list[str]]:
        return text, []  # Should not be called anymore

    def _remove_ignore_sub_config(self, text: str) -> tuple[str, list[str]]:
        return text, []

    def _ensure_project_excludes(self, text: str) -> tuple[str, list[str]]:
        return text, []

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
    _ = run_parser.add_argument("--reports-dir", default=f"{REPORTS_DIR_NAME}/check")
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
            output.error(run_result.error or "check failed")
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
            output.error(fix_result.error or "pyrefly config fix failed")
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
