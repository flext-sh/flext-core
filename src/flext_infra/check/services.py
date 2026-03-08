"""Quality gate execution and workspace checking services."""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import override

import tomlkit
from pydantic import TypeAdapter, ValidationError
from tomlkit import items

from flext_core import FlextLogger, r, s
from flext_infra import (
    FlextInfraCommandRunner,
    FlextInfraDiscoveryService,
    FlextInfraJsonService,
    FlextInfraPathResolver,
    FlextInfraReportingService,
    c,
    m,
    output,
    p,
    t,
)

_logger = FlextLogger.create_module_logger(__name__)


class FlextInfraWorkspaceChecker(s[list[m.Infra.Check.ProjectResult]]):
    """Run quality gates across one or more workspace projects."""

    def __init__(self, workspace_root: Path | None = None) -> None:
        """Initialize checker dependencies and default report directory."""
        super().__init__()
        self._path_resolver = FlextInfraPathResolver()
        self._reporting = FlextInfraReportingService()
        self._json = FlextInfraJsonService()
        self._runner: p.Infra.CommandRunner = FlextInfraCommandRunner()
        self._workspace_root = self._resolve_workspace_root(workspace_root)
        report_dir = self._reporting.get_report_dir(
            self._workspace_root, c.Infra.Toml.PROJECT, c.Infra.Verbs.CHECK
        )
        try:
            report_dir.mkdir(parents=True, exist_ok=True)
            self._default_reports_dir = report_dir
        except OSError:
            self._default_reports_dir = (
                self._workspace_root
                / c.Infra.Reporting.REPORTS_DIR_NAME
                / c.Infra.Verbs.CHECK
            )

    @staticmethod
    def _dirs_with_py(project_dir: Path, dirs: list[str]) -> list[str]:
        out: list[str] = []
        for directory in dirs:
            path = project_dir / directory
            if not path.is_dir():
                continue
            if next(path.rglob(c.Infra.Extensions.PYTHON_GLOB), None) or next(
                path.rglob("*.pyi"), None
            ):
                out.append(directory)
        return out

    @staticmethod
    def generate_sarif_report(
        results: list[m.Infra.Check.ProjectResult], gates: list[str]
    ) -> Mapping[str, t.ContainerValue]:
        """Render gate results as a SARIF 2.1.0 payload."""
        tool_info = {
            c.Infra.Gates.LINT: ("Ruff Linter", "https://docs.astral.sh/ruff/"),
            c.Infra.Gates.FORMAT: (
                "Ruff Formatter",
                "https://docs.astral.sh/ruff/formatter/",
            ),
            c.Infra.Gates.PYREFLY: ("Pyrefly", "https://github.com/facebook/pyrefly"),
            c.Infra.Gates.MYPY: ("Mypy", "https://mypy.readthedocs.io/"),
            c.Infra.Gates.PYRIGHT: ("Pyright", "https://github.com/microsoft/pyright"),
            c.Infra.Gates.SECURITY: ("Bandit", "https://bandit.readthedocs.io/"),
            c.Infra.Gates.MARKDOWN: (
                "MarkdownLint",
                "https://github.com/DavidAnson/markdownlint",
            ),
            c.Infra.Gates.GO: ("Go Vet", "https://pkg.go.dev/cmd/vet"),
        }
        sarif_runs: list[m.Infra.Check.Sarif.Run] = []
        for gate in gates:
            tool_name, tool_url = tool_info.get(gate, (gate, ""))
            sarif_results: list[m.Infra.Check.Sarif.Result] = []
            rules_seen: set[str] = set()
            rules: list[m.Infra.Check.Sarif.Rule] = []
            for project in results:
                gate_result = project.gates.get(gate)
                if not gate_result:
                    continue
                for issue in gate_result.issues:
                    rule_id = issue.code or c.Infra.Defaults.UNKNOWN
                    if rule_id not in rules_seen:
                        rules_seen.add(rule_id)
                        rules.append(
                            m.Infra.Check.Sarif.Rule(
                                id=rule_id, short_description=rule_id
                            )
                        )
                    sarif_results.append(
                        m.Infra.Check.Sarif.Result(
                            rule_id=rule_id,
                            level=c.Infra.Toml.ERROR
                            if issue.severity == "error"
                            else c.Infra.Severity.WARNING,
                            message=issue.message,
                            locations=[
                                m.Infra.Check.Sarif.Location(
                                    uri=issue.file,
                                    start_line=max(issue.line, 1),
                                    start_column=max(issue.column, 1),
                                )
                            ],
                        )
                    )
            sarif_runs.append(
                m.Infra.Check.Sarif.Run(
                    tool_name=tool_name,
                    information_uri=tool_url,
                    rules=rules,
                    results=sarif_results,
                )
            )
        return m.Infra.Check.Sarif.Report(runs=sarif_runs).model_dump(by_alias=True)

    @staticmethod
    def parse_gate_csv(raw: str) -> list[str]:
        """Parse comma-separated gate names into a normalized list."""
        return [gate.strip() for gate in raw.split(",") if gate.strip()]

    @staticmethod
    def resolve_gates(gates: Sequence[str]) -> r[list[str]]:
        """Validate and normalize user-provided gate names."""
        allowed = {
            c.Infra.Gates.LINT,
            c.Infra.Gates.FORMAT,
            c.Infra.Gates.PYREFLY,
            c.Infra.Gates.MYPY,
            c.Infra.Gates.PYRIGHT,
            c.Infra.Gates.SECURITY,
            c.Infra.Gates.MARKDOWN,
            c.Infra.Gates.GO,
        }
        resolved: list[str] = []
        for gate in gates:
            name = gate.strip()
            if not name:
                continue
            mapped = c.Infra.Gates.PYREFLY if name == c.Infra.Gates.TYPE_ALIAS else name
            if mapped not in allowed:
                return r[list[str]].fail(f"ERROR: unknown gate '{gate}'")
            if mapped not in resolved:
                resolved.append(mapped)
        return r[list[str]].ok(resolved)

    @override
    def execute(self) -> r[list[m.Infra.Check.ProjectResult]]:
        """Return a failure because this service requires explicit run inputs."""
        return r[list[m.Infra.Check.ProjectResult]].fail(
            "Use run() or run_projects() directly"
        )

    def format(self, project_dir: Path) -> r[m.Infra.Check.GateResult]:
        """Run the Ruff format check gate for a project."""
        return r[m.Infra.Check.GateResult].ok(self._run_ruff_format(project_dir).result)

    def generate_markdown_report(
        self,
        results: list[m.Infra.Check.ProjectResult],
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
                row += f" {(len(gate_result.issues) if gate_result else 0)} |"
            status = (
                c.Infra.Status.PASS if project.passed else f"**{c.Infra.Status.FAIL}**"
            )
            if not project.passed:
                failed_count += 1
            row += f" {project.total_errors} | {status} |"
            total_all += project.total_errors
            lines.append(row)
        lines.extend([
            "",
            f"**Total errors**: {total_all}  ",
            f"**Failed projects**: {failed_count}/{len(results)}  ",
            "",
        ])
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
                    for issue in gate_result.issues[: c.Infra.Check.MAX_DISPLAY_ISSUES]
                )
                if len(gate_result.issues) > c.Infra.Check.MAX_DISPLAY_ISSUES:
                    lines.append(
                        f"... and {len(gate_result.issues) - c.Infra.Check.MAX_DISPLAY_ISSUES} more errors"
                    )
                lines.extend(["```", ""])
        return "\n".join(lines)

    def lint(self, project_dir: Path) -> r[m.Infra.Check.GateResult]:
        """Run the Ruff lint gate for a project."""
        return r[m.Infra.Check.GateResult].ok(self._run_ruff_lint(project_dir).result)

    def run(
        self, project: str, gates: Sequence[str]
    ) -> r[list[m.Infra.Check.ProjectResult]]:
        """Run selected gates for a single project."""
        return self.run_projects([project], list(gates)).map(lambda value: value)

    def run_projects(
        self,
        projects: Sequence[str],
        gates: Sequence[str],
        *,
        reports_dir: Path | None = None,
        fail_fast: bool = False,
    ) -> r[list[m.Infra.Check.ProjectResult]]:
        """Run selected gates for multiple projects and emit reports."""
        resolved_gates_result = self.resolve_gates(gates)
        if resolved_gates_result.is_failure:
            return r[list[m.Infra.Check.ProjectResult]].fail(
                resolved_gates_result.error or "invalid gates"
            )
        resolved_gates: list[str] = resolved_gates_result.value
        report_base = reports_dir or self._default_reports_dir
        report_base.mkdir(parents=True, exist_ok=True)
        results: list[m.Infra.Check.ProjectResult] = []
        total = len(projects)
        failed = 0
        skipped = 0
        loop_start = time.monotonic()
        for index, project_name in enumerate(projects, 1):
            project_dir = self._workspace_root / project_name
            pyproject_path = project_dir / c.Infra.Files.PYPROJECT_FILENAME
            if not project_dir.is_dir() or not pyproject_path.exists():
                output.progress(index, total, project_name, c.Infra.Severity.SKIP)
                skipped += 1
                continue
            output.progress(index, total, project_name, c.Infra.Verbs.CHECK)
            start = time.monotonic()
            project_result = self._check_project(
                project_dir, resolved_gates, report_base
            )
            elapsed = time.monotonic() - start
            results.append(project_result)
            if project_result.passed:
                output.status(c.Infra.Verbs.CHECK, project_name, True, elapsed)
            else:
                output.status(c.Infra.Verbs.CHECK, project_name, False, elapsed)
                failed += 1
                if fail_fast:
                    break
        total_elapsed = time.monotonic() - loop_start
        timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
        md_path = report_base / "check-report.md"
        _ = md_path.write_text(
            self.generate_markdown_report(results, resolved_gates, timestamp),
            encoding=c.Infra.Encoding.DEFAULT,
        )
        sarif_path = report_base / "check-report.sarif"
        sarif_payload = self.generate_sarif_report(results, resolved_gates)
        json_write_result = self._json.write(sarif_path, sarif_payload)
        if json_write_result.is_failure:
            return r[list[m.Infra.Check.ProjectResult]].fail(
                json_write_result.error or "failed to write sarif report"
            )
        total_errors = sum(project.total_errors for project in results)
        success = len(results) - failed
        output.summary(
            c.Infra.Verbs.CHECK, len(results), success, failed, skipped, total_elapsed
        )
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
        return r[list[m.Infra.Check.ProjectResult]].ok(results)

    def _build_gate_result(
        self,
        *,
        gate: str,
        project: str,
        passed: bool,
        issues: list[m.Infra.Check.Issue],
        duration: float,
        raw_output: str,
    ) -> m.Infra.Check.GateExecution:
        model = m.Infra.Check.GateResult(
            gate=gate,
            project=project,
            passed=passed,
            errors=[issue.formatted for issue in issues],
            duration=round(duration, 3),
        )
        return m.Infra.Check.GateExecution(
            result=model, issues=issues, raw_output=raw_output
        )

    def _check_project(
        self, project_dir: Path, gates: list[str], reports_dir: Path
    ) -> m.Infra.Check.ProjectResult:
        result = m.Infra.Check.ProjectResult(project=project_dir.name)
        runners: Mapping[str, Callable[[], m.Infra.Check.GateExecution]] = {
            c.Infra.Gates.LINT: lambda: self._run_ruff_lint(project_dir),
            c.Infra.Gates.FORMAT: lambda: self._run_ruff_format(project_dir),
            c.Infra.Gates.PYREFLY: lambda: self._run_pyrefly(project_dir, reports_dir),
            c.Infra.Gates.MYPY: lambda: self._run_mypy(project_dir),
            c.Infra.Gates.PYRIGHT: lambda: self._run_pyright(project_dir),
            c.Infra.Gates.SECURITY: lambda: self._run_bandit(project_dir),
            c.Infra.Gates.MARKDOWN: lambda: self._run_markdown(project_dir),
            c.Infra.Gates.GO: lambda: self._run_go(project_dir),
        }
        for gate in gates:
            runner = runners.get(gate)
            if runner:
                result.gates[gate] = runner()
        return result

    def _collect_markdown_files(self, project_dir: Path) -> list[Path]:
        files: list[Path] = []
        for path in project_dir.rglob("*.md"):
            if any(part in c.Infra.Excluded.CHECK_EXCLUDED_DIRS for part in path.parts):
                continue
            files.append(path)
        return files

    def _existing_check_dirs(self, project_dir: Path) -> list[str]:
        dirs = (
            c.Infra.Check.DEFAULT_CHECK_DIRS
            if project_dir.resolve() == self._workspace_root.resolve()
            else c.Infra.Check.CHECK_DIRS_SUBPROJECT
        )
        return [directory for directory in dirs if (project_dir / directory).is_dir()]

    def _resolve_workspace_root(self, workspace_root: Path | None) -> Path:
        if workspace_root is not None:
            return workspace_root.resolve()
        result = self._path_resolver.workspace_root()
        return result.value if result.is_success else Path.cwd().resolve()

    @staticmethod
    def _to_mapping(value: t.ContainerValue) -> dict[str, t.ContainerValue]:
        if not isinstance(value, Mapping):
            return {}
        return TypeAdapter(dict[str, t.ContainerValue]).validate_python(value)

    @classmethod
    def _to_mapping_list(
        cls, value: t.ContainerValue
    ) -> list[dict[str, t.ContainerValue]]:
        if not isinstance(value, list):
            return []
        return [cls._to_mapping(item) for item in value if isinstance(item, Mapping)]

    @staticmethod
    def _as_int(value: t.ContainerValue, default: int = 0) -> int:
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                return default
        return default

    @staticmethod
    def _as_str(value: t.ContainerValue, default: str = "") -> str:
        return value if isinstance(value, str) else default

    @staticmethod
    def _nested_mapping(
        data: dict[str, t.ContainerValue], *keys: str
    ) -> dict[str, t.ContainerValue]:
        """Walk a chain of string keys, returning the nested Mapping or {}."""
        current: t.ContainerValue = data
        for key in keys:
            if not isinstance(current, Mapping):
                return {}
            child: t.ContainerValue = current.get(key)
            if child is None:
                return {}
            current = child
        if not isinstance(current, Mapping):
            return {}
        return dict(current)

    @classmethod
    def _nested_int(
        cls, data: dict[str, t.ContainerValue], *keys: str, default: int = 0
    ) -> int:
        """Walk keys into nested mappings, returning an int leaf or *default*."""
        target = cls._nested_mapping(data, *keys[:-1])
        raw: t.ContainerValue = target.get(keys[-1])
        if raw is None:
            return default
        return cls._as_int(raw, default)

    @classmethod
    def _result_exit_code(cls, result: p.Infra.CommandOutput) -> int:
        """Extract exit code from command output payloads and test doubles."""
        try:
            payload = TypeAdapter(dict[str, t.ContainerValue]).validate_python(
                vars(result)
            )
        except (TypeError, ValidationError, AttributeError):
            return 1
        code = payload.get("exit_code")
        if code is None:
            code = payload.get("returncode")
        if code is None:
            return 1
        return cls._as_int(code, 1)

    def _run(
        self,
        cmd: list[str],
        cwd: Path,
        timeout: int = c.Infra.Timeouts.DEFAULT,
        env: Mapping[str, str] | None = None,
    ) -> m.Infra.Core.CommandOutput:
        result = self._runner.run_raw(cmd, cwd=cwd, timeout=timeout, env=env)
        if result.is_failure:
            return m.Infra.Core.CommandOutput(
                stdout="",
                stderr=result.error or "command execution failed",
                exit_code=1,
            )
        cmd_output: m.Infra.Core.CommandOutput = result.value
        return m.Infra.Core.CommandOutput(
            stdout=cmd_output.stdout,
            stderr=cmd_output.stderr,
            exit_code=cmd_output.exit_code,
            duration=cmd_output.duration,
        )

    def _run_bandit(self, project_dir: Path) -> m.Infra.Check.GateExecution:
        started = time.monotonic()
        src_path = project_dir / c.Infra.Paths.DEFAULT_SRC_DIR
        if not src_path.exists():
            return self._build_gate_result(
                gate=c.Infra.Gates.SECURITY,
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
                c.Infra.Cli.BANDIT,
                "-r",
                c.Infra.Paths.DEFAULT_SRC_DIR,
                "-f",
                c.Infra.Cli.OUTPUT_JSON,
                "-q",
                "-ll",
            ],
            project_dir,
        )
        issues: list[m.Infra.Check.Issue] = []
        bandit_data: dict[str, t.ContainerValue] = {}
        try:
            parsed = self._json.parse(result.stdout or "{}")
            if parsed.is_success and isinstance(parsed.value, Mapping):
                bandit_data = self._to_mapping(parsed.value)
            raw_results: list[dict[str, t.ContainerValue]] = self._to_mapping_list(
                bandit_data.get("results", [])
            )
            issues.extend(
                m.Infra.Check.Issue(
                    file=self._as_str(raw_item.get("filename", "?"), "?"),
                    line=self._as_int(raw_item.get("line_number", 0)),
                    column=0,
                    code=self._as_str(raw_item.get("test_id", "")),
                    message=self._as_str(raw_item.get("issue_text", "")),
                    severity=self._as_str(
                        raw_item.get("issue_severity", "MEDIUM"), "MEDIUM"
                    ).lower(),
                )
                for raw_item in raw_results
            )
        except (TypeError, ValidationError):
            pass
        return self._build_gate_result(
            gate=c.Infra.Gates.SECURITY,
            project=project_dir.name,
            passed=self._result_exit_code(result) == 0,
            issues=issues,
            duration=time.monotonic() - started,
            raw_output=result.stderr,
        )

    def _run_go(self, project_dir: Path) -> m.Infra.Check.GateExecution:
        started = time.monotonic()
        if not (project_dir / c.Infra.Files.GO_MOD).exists():
            return self._build_gate_result(
                gate=c.Infra.Gates.GO,
                project=project_dir.name,
                passed=True,
                issues=[],
                duration=time.monotonic() - started,
                raw_output="",
            )
        issues: list[m.Infra.Check.Issue] = []
        raw_output = ""
        vet_result = self._run(
            [c.Infra.Cli.GOVET, "vet", "./..."],
            project_dir,
            timeout=c.Infra.Timeouts.CI,
        )
        raw_output = "\n".join(
            part for part in (vet_result.stdout, vet_result.stderr) if part
        )
        for line in (vet_result.stdout + "\n" + vet_result.stderr).splitlines():
            match = c.Infra.Check.GO_VET_RE.match(line.strip())
            if not match:
                continue
            issues.append(
                m.Infra.Check.Issue(
                    file=match.group("file"),
                    line=int(match.group("line")),
                    column=int(match.group("col") or 1),
                    code=c.Infra.Gates.GOVET,
                    message=match.group("msg"),
                )
            )
        if self._result_exit_code(vet_result) != 0 and (not issues):
            issues.append(
                m.Infra.Check.Issue(
                    file=".",
                    line=1,
                    column=1,
                    code=c.Infra.Gates.GOVET,
                    message=(
                        vet_result.stdout or vet_result.stderr or "go vet failed"
                    ).strip(),
                )
            )
        go_files = list(project_dir.rglob("*.go"))
        if go_files:
            fmt_result = self._run(
                [
                    c.Infra.Cli.GOFMT,
                    "-l",
                    *[str(path.relative_to(project_dir)) for path in go_files],
                ],
                project_dir,
                timeout=c.Infra.Timeouts.CI,
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
                    m.Infra.Check.Issue(
                        file=cleaned,
                        line=1,
                        column=1,
                        code=c.Infra.Gates.GOFMT,
                        message="File is not gofmt-formatted",
                    )
                )
        return self._build_gate_result(
            gate=c.Infra.Gates.GO,
            project=project_dir.name,
            passed=len(issues) == 0,
            issues=issues,
            duration=time.monotonic() - started,
            raw_output=raw_output,
        )

    def _run_markdown(self, project_dir: Path) -> m.Infra.Check.GateExecution:
        started = time.monotonic()
        md_files = self._collect_markdown_files(project_dir)
        if not md_files:
            return self._build_gate_result(
                gate=c.Infra.Gates.MARKDOWN,
                project=project_dir.name,
                passed=True,
                issues=[],
                duration=time.monotonic() - started,
                raw_output="",
            )
        cmd = [c.Infra.Cli.MARKDOWNLINT]
        root_config = self._workspace_root / ".markdownlint.json"
        local_config = project_dir / ".markdownlint.json"
        if root_config.exists():
            cmd.extend(["--config", str(root_config)])
        elif local_config.exists():
            cmd.extend(["--config", str(local_config)])
        cmd.extend(str(path.relative_to(project_dir)) for path in md_files)
        result = self._run(cmd, project_dir)
        issues: list[m.Infra.Check.Issue] = []
        for line in (result.stdout + "\n" + result.stderr).splitlines():
            match = c.Infra.Check.MARKDOWN_RE.match(line.strip())
            if not match:
                continue
            issues.append(
                m.Infra.Check.Issue(
                    file=match.group("file"),
                    line=int(match.group("line")),
                    column=int(match.group("col") or 1),
                    code=match.group("code"),
                    message=match.group("msg"),
                )
            )
        if self._result_exit_code(result) != 0 and (not issues):
            issues.append(
                m.Infra.Check.Issue(
                    file=".",
                    line=1,
                    column=1,
                    code=c.Infra.Gates.MARKDOWNLINT,
                    message=(
                        result.stdout or result.stderr or "markdownlint failed"
                    ).strip(),
                )
            )
        return self._build_gate_result(
            gate=c.Infra.Gates.MARKDOWN,
            project=project_dir.name,
            passed=self._result_exit_code(result) == 0,
            issues=issues,
            duration=time.monotonic() - started,
            raw_output=result.stderr,
        )

    def _run_mypy(self, project_dir: Path) -> m.Infra.Check.GateExecution:
        started = time.monotonic()
        check_dirs = self._existing_check_dirs(project_dir)
        mypy_dirs = self._dirs_with_py(project_dir, check_dirs)
        if not mypy_dirs:
            return self._build_gate_result(
                gate=c.Infra.Gates.MYPY,
                project=project_dir.name,
                passed=True,
                issues=[],
                duration=time.monotonic() - started,
                raw_output="",
            )
        proj_py = project_dir / c.Infra.Files.PYPROJECT_FILENAME
        cfg = (
            proj_py
            if proj_py.exists()
            and "[tool.mypy]" in proj_py.read_text(encoding=c.Infra.Encoding.DEFAULT)
            else self._workspace_root / c.Infra.Files.PYPROJECT_FILENAME
        )
        typings_generated = (
            self._workspace_root / c.Infra.Directories.TYPINGS / "generated"
        )
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
                c.Infra.Cli.MYPY,
                *mypy_dirs,
                "--config-file",
                str(cfg),
                "--output",
                c.Infra.Cli.OUTPUT_JSON,
            ],
            project_dir,
            env=mypy_env,
        )
        issues: list[m.Infra.Check.Issue] = []
        for raw_line in (result.stdout or "").splitlines():
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                line_data = TypeAdapter(dict[str, t.ContainerValue]).validate_json(
                    stripped
                )
            except ValidationError:
                continue
            try:
                severity = self._as_str(
                    line_data.get("severity", c.Infra.Toml.ERROR), c.Infra.Toml.ERROR
                )
                if severity in {"error", "warning", "note"}:
                    issues.append(
                        m.Infra.Check.Issue(
                            file=self._as_str(line_data.get("file", "?"), "?"),
                            line=self._as_int(line_data.get("line", 0)),
                            column=self._as_int(line_data.get("column", 0)),
                            code=self._as_str(line_data.get("code", "")),
                            message=self._as_str(line_data.get("message", "")),
                            severity=severity,
                        )
                    )
            except ValidationError:
                continue
        return self._build_gate_result(
            gate=c.Infra.Gates.MYPY,
            project=project_dir.name,
            passed=self._result_exit_code(result) == 0,
            issues=issues,
            duration=time.monotonic() - started,
            raw_output=result.stderr,
        )

    def _run_pyrefly(
        self, project_dir: Path, reports_dir: Path
    ) -> m.Infra.Check.GateExecution:
        started = time.monotonic()
        check_dirs = self._existing_check_dirs(project_dir)
        targets = check_dirs or [c.Infra.Paths.DEFAULT_SRC_DIR]
        json_file = reports_dir / f"{project_dir.name}-pyrefly.json"
        cmd = [
            sys.executable,
            "-m",
            c.Infra.Cli.PYREFLY,
            c.Infra.Cli.RuffCmd.CHECK,
            *targets,
            "--config",
            c.Infra.Files.PYPROJECT_FILENAME,
            "--output-format",
            c.Infra.Cli.OUTPUT_JSON,
            "-o",
            str(json_file),
            "--summary=none",
        ]
        result = self._run(cmd, project_dir)
        issues: list[m.Infra.Check.Issue] = []
        if json_file.exists():
            try:
                raw_text = json_file.read_text(encoding=c.Infra.Encoding.DEFAULT)
                parsed = self._json.parse(raw_text)
                if parsed.is_success and isinstance(parsed.value, Mapping):
                    parsed_map = self._to_mapping(parsed.value)
                    error_items: list[dict[str, t.ContainerValue]] = (
                        self._to_mapping_list(parsed_map.get("errors", []))
                    )
                elif parsed.is_success and isinstance(parsed.value, list):
                    error_items = self._to_mapping_list(parsed.value)
                else:
                    error_items = []
                issues.extend(
                    m.Infra.Check.Issue(
                        file=self._as_str(d.get("path"), "?"),
                        line=self._as_int(d.get("line"), 0),
                        column=self._as_int(d.get("column"), 0),
                        code=self._as_str(d.get("name"), ""),
                        message=self._as_str(d.get("description"), ""),
                        severity=self._as_str(d.get("severity"), c.Infra.Toml.ERROR),
                    )
                    for err in error_items
                    for d in [dict(err)]
                )
            except (TypeError, ValidationError):
                pass
        if not issues and self._result_exit_code(result) != 0:
            match = re.search("(\\d+)\\s+errors?", result.stderr + result.stdout)
            if match:
                count = int(match.group(1))
                issues = [
                    m.Infra.Check.Issue(
                        file="?",
                        line=0,
                        column=0,
                        code=c.Infra.Gates.PYREFLY,
                        message=f"Pyrefly reported {count} error(s)",
                    )
                ] * count
        return self._build_gate_result(
            gate=c.Infra.Gates.PYREFLY,
            project=project_dir.name,
            passed=self._result_exit_code(result) == 0,
            issues=issues,
            duration=time.monotonic() - started,
            raw_output=result.stderr,
        )

    def _run_pyright(self, project_dir: Path) -> m.Infra.Check.GateExecution:
        started = time.monotonic()
        check_dirs = self._dirs_with_py(
            project_dir, self._existing_check_dirs(project_dir)
        )
        if not check_dirs:
            return self._build_gate_result(
                gate=c.Infra.Gates.PYRIGHT,
                project=project_dir.name,
                passed=True,
                issues=[],
                duration=time.monotonic() - started,
                raw_output="",
            )
        result = self._run(
            [sys.executable, "-m", c.Infra.Cli.PYRIGHT, *check_dirs, "--outputjson"],
            project_dir,
            timeout=c.Infra.Timeouts.LONG,
        )
        issues: list[m.Infra.Check.Issue] = []
        pyright_parse_result = self._json.parse(result.stdout or "{}")
        pyright_data: dict[str, t.ContainerValue] = self._to_mapping(
            pyright_parse_result.value if pyright_parse_result.is_success else {}
        )
        try:
            raw_diagnostics: list[dict[str, t.ContainerValue]] = self._to_mapping_list(
                pyright_data.get("generalDiagnostics", [])
            )
            issues.extend(
                m.Infra.Check.Issue(
                    file=str(diag.get("file", "?")),
                    line=self._nested_int(diag, "range", "start", "line") + 1,
                    column=self._nested_int(diag, "range", "start", "character") + 1,
                    code=str(diag.get("rule", "")),
                    message=str(diag.get("message", "")),
                    severity=str(diag.get("severity", c.Infra.Toml.ERROR)),
                )
                for diag in raw_diagnostics
            )
        except (TypeError, ValidationError):
            pass
        return self._build_gate_result(
            gate=c.Infra.Gates.PYRIGHT,
            project=project_dir.name,
            passed=self._result_exit_code(result) == 0,
            issues=issues,
            duration=time.monotonic() - started,
            raw_output=result.stderr,
        )

    def _run_ruff_format(self, project_dir: Path) -> m.Infra.Check.GateExecution:
        started = time.monotonic()
        check_dirs = self._existing_check_dirs(project_dir)
        targets = check_dirs or ["."]
        result = self._run(
            [
                sys.executable,
                "-m",
                c.Infra.Cli.RUFF,
                c.Infra.Cli.RuffCmd.FORMAT,
                "--check",
                *targets,
                "--quiet",
            ],
            project_dir,
        )
        issues: list[m.Infra.Check.Issue] = []
        if self._result_exit_code(result) != 0 and result.stdout.strip():
            seen: set[str] = set()
            for line in result.stdout.strip().splitlines():
                path = line.strip()
                if not path:
                    continue
                match = c.Infra.Check.RUFF_FORMAT_FILE_RE.match(path)
                if match:
                    file_path = match.group(1).strip()
                    if file_path in seen:
                        continue
                    seen.add(file_path)
                    issues.append(
                        m.Infra.Check.Issue(
                            file=file_path,
                            line=0,
                            column=0,
                            code=c.Infra.Gates.FORMAT,
                            message="Would be reformatted",
                        )
                    )
                elif (
                    path.endswith(c.Infra.Extensions.PYTHON)
                    and " " not in path
                    and (path not in seen)
                ):
                    seen.add(path)
                    issues.append(
                        m.Infra.Check.Issue(
                            file=path,
                            line=0,
                            column=0,
                            code=c.Infra.Gates.FORMAT,
                            message="Would be reformatted",
                        )
                    )
        return self._build_gate_result(
            gate=c.Infra.Gates.FORMAT,
            project=project_dir.name,
            passed=self._result_exit_code(result) == 0,
            issues=issues,
            duration=time.monotonic() - started,
            raw_output=result.stderr,
        )

    def _run_ruff_lint(self, project_dir: Path) -> m.Infra.Check.GateExecution:
        started = time.monotonic()
        check_dirs = self._existing_check_dirs(project_dir)
        targets = check_dirs or ["."]
        result = self._run(
            [
                sys.executable,
                "-m",
                c.Infra.Toml.RUFF,
                c.Infra.Verbs.CHECK,
                *targets,
                "--output-format",
                c.Infra.Cli.OUTPUT_JSON,
                "--quiet",
            ],
            project_dir,
        )
        issues: list[m.Infra.Check.Issue] = []
        ruff_parse_result = self._json.parse(result.stdout or "[]")
        ruff_data: t.ContainerValue = (
            ruff_parse_result.value if ruff_parse_result.is_success else []
        )
        try:
            if isinstance(ruff_data, list):
                issues.extend(
                    m.Infra.Check.Issue(
                        file=str(entry.get("filename", "?")),
                        line=self._nested_int(dict(entry), "location", "row"),
                        column=self._nested_int(dict(entry), "location", "column"),
                        code=str(entry.get("code", "")),
                        message=str(entry.get("message", "")),
                    )
                    for entry in ruff_data
                    if isinstance(entry, Mapping)
                )
        except (TypeError, ValidationError):
            pass
        return self._build_gate_result(
            gate=c.Infra.Gates.LINT,
            project=project_dir.name,
            passed=self._result_exit_code(result) == 0,
            issues=issues,
            duration=time.monotonic() - started,
            raw_output=result.stderr,
        )


class FlextInfraConfigFixer(s[list[str]]):
    """Repair workspace and project pyrefly configuration blocks."""

    def __init__(self, workspace_root: Path | None = None) -> None:
        """Initialize fixer dependencies and resolve workspace root."""
        super().__init__()
        self._path_resolver = FlextInfraPathResolver()
        self._discovery = FlextInfraDiscoveryService()
        self._workspace_root = self._resolve_workspace_root(workspace_root)

    @staticmethod
    def _to_array(items_list: list[str]) -> items.Array:
        serialized_result = FlextInfraJsonService().serialize(items_list)
        if serialized_result.is_failure:
            return tomlkit.array()
        inline_doc = tomlkit.parse(f"items = {serialized_result.value}\n")
        arr_raw = inline_doc["items"]
        if not isinstance(arr_raw, items.Array):
            return tomlkit.array()
        arr = arr_raw
        if len(items_list) > 1:
            arr.multiline(True)
        return arr

    @override
    def execute(self) -> r[list[str]]:
        """Return a failure because this service requires explicit run inputs."""
        return r[list[str]].fail("Use run() directly")

    def find_pyproject_files(
        self, project_paths: list[Path] | None = None
    ) -> r[list[Path]]:
        """Find pyproject.toml files in workspace or selected project paths."""
        return self._discovery.find_all_pyproject_files(
            self._workspace_root, project_paths=project_paths
        )

    def process_file(self, path: Path, *, dry_run: bool = False) -> r[list[str]]:
        """Apply all pyrefly block fixes to a single pyproject.toml file."""
        try:
            text = path.read_text(encoding=c.Infra.Encoding.DEFAULT)
            doc = tomlkit.parse(text)
            doc_data = doc.unwrap()
        except OSError as exc:
            return r[list[str]].fail(f"failed to read {path}: {exc}")
        except Exception as exc:
            return r[list[str]].fail(f"failed to parse {path}: {exc}")
        tool_data = doc_data.get(c.Infra.Toml.TOOL)
        if not isinstance(tool_data, dict):
            return r[list[str]].ok([])
        typed_tool_data = TypeAdapter(dict[str, t.ContainerValue]).validate_python(
            tool_data
        )
        pyrefly_data = typed_tool_data.get(c.Infra.Toml.PYREFLY)
        if not isinstance(pyrefly_data, Mapping):
            return r[list[str]].ok([])
        pyrefly: MutableMapping[str, t.ContainerValue] = dict(pyrefly_data)
        all_fixes: list[str] = []
        fixes = self._fix_search_paths_tk(pyrefly, path.parent)
        all_fixes.extend(fixes)
        fixes = self._remove_ignore_sub_config_tk(pyrefly)
        all_fixes.extend(fixes)
        if (
            any("removed ignore" in item for item in all_fixes)
            or path.parent == self._workspace_root
        ):
            fixes = self._ensure_project_excludes_tk(pyrefly)
            all_fixes.extend(fixes)
        if all_fixes and (not dry_run):
            try:
                typed_tool_data[c.Infra.Toml.PYREFLY] = dict(pyrefly)
                doc_data[c.Infra.Toml.TOOL] = typed_tool_data
                new_doc = tomlkit.document()
                for key, value in doc_data.items():
                    new_doc[str(key)] = value
                new_text = new_doc.as_string()
                _ = path.write_text(new_text, encoding=c.Infra.Encoding.DEFAULT)
            except OSError as exc:
                return r[list[str]].fail(f"failed to write {path}: {exc}")
        return r[list[str]].ok(all_fixes)

    def run(
        self, projects: Sequence[str], *, dry_run: bool = False, verbose: bool = False
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
        pyproject_files: list[Path] = files_result.value
        for path in pyproject_files:
            fixes_result = self.process_file(path, dry_run=dry_run)
            if fixes_result.is_failure:
                return r[list[str]].fail(
                    fixes_result.error or f"failed to process {path}"
                )
            fixes: list[str] = fixes_result.value
            if not fixes:
                continue
            total_fixes += len(fixes)
            if verbose:
                try:
                    rel = path.relative_to(self._workspace_root)
                except ValueError:
                    rel = path
                for fix in fixes:
                    line = f"  {('(dry)' if dry_run else '✓')} {rel}: {fix}"
                    _logger.info("pyrefly_config_fix", detail=line)
                    messages.append(line)
        if verbose and total_fixes == 0:
            _logger.info("pyrefly_configs_clean")
        return r[list[str]].ok(messages)

    def _ensure_project_excludes_tk(
        self, pyrefly: MutableMapping[str, t.ContainerValue]
    ) -> list[str]:
        fixes: list[str] = []
        excludes = pyrefly.get(c.Infra.Toml.PROJECT_EXCLUDES)
        current: list[str] = []
        if isinstance(excludes, list):
            current = [str(x) for x in excludes]
        stripped_to_add: list[str] = []
        for glob in c.Infra.Check.REQUIRED_EXCLUDES:
            clean_glob = glob.strip('"').strip("'")
            if clean_glob not in current and glob not in current:
                stripped_to_add.append(clean_glob)
        if stripped_to_add:
            updated = sorted(set(current) | set(stripped_to_add))
            pyrefly[c.Infra.Toml.PROJECT_EXCLUDES] = self._to_array(updated)
            fixes.append(f"added {', '.join(stripped_to_add)} to project-excludes")
        return fixes

    def _fix_search_paths_tk(
        self, pyrefly: MutableMapping[str, t.ContainerValue], project_dir: Path
    ) -> list[str]:
        fixes: list[str] = []
        search_path = pyrefly.get(c.Infra.Toml.SEARCH_PATH)
        if not isinstance(search_path, list):
            return []
        if project_dir == self._workspace_root:
            new_paths: list[str] = []
            for p in search_path:
                if not isinstance(p, str):
                    continue
                if p == "../typings/generated":
                    new_paths.append("typings/generated")
                    fixes.append(
                        "search-path ../typings/generated -> typings/generated"
                    )
                elif p == "../typings":
                    new_paths.append(c.Infra.Directories.TYPINGS)
                    fixes.append("search-path ../typings -> typings")
                else:
                    new_paths.append(p)
            if fixes:
                pyrefly[c.Infra.Toml.SEARCH_PATH] = self._to_array(new_paths)
        search_raw = pyrefly.get(c.Infra.Toml.SEARCH_PATH)
        current_paths: list[t.ContainerValue] = (
            list(search_raw) if isinstance(search_raw, list) else []
        )
        nonexistent = [
            p
            for p in current_paths
            if isinstance(p, str) and (not (project_dir / p).exists())
        ]
        if nonexistent:
            remaining: list[str] = [
                str(p)
                for p in current_paths
                if isinstance(p, str) and p not in nonexistent
            ]
            pyrefly[c.Infra.Toml.SEARCH_PATH] = self._to_array(remaining)
            fixes.append(f"removed nonexistent search-path: {', '.join(nonexistent)}")
        return fixes

    def _remove_ignore_sub_config_tk(
        self, pyrefly: MutableMapping[str, t.ContainerValue]
    ) -> list[str]:
        fixes: list[str] = []
        sub_configs = pyrefly.get(c.Infra.Toml.SUB_CONFIG)
        if not isinstance(sub_configs, list):
            return []
        new_configs: list[t.ContainerValue] = []
        for conf in sub_configs:
            if isinstance(conf, Mapping) and conf.get(c.Infra.Toml.IGNORE) is True:
                matches = conf.get("matches", c.Infra.Defaults.UNKNOWN)
                fixes.append(f"removed ignore=true sub-config for '{matches}'")
                continue
            new_configs.append(conf)
        if len(new_configs) != len(sub_configs):
            pyrefly[c.Infra.Toml.SUB_CONFIG] = new_configs
        return fixes

    def _resolve_project_path(self, raw: str) -> Path:
        path = Path(raw)
        if not path.is_absolute():
            path = self._workspace_root / path
        return path.resolve()

    def _resolve_workspace_root(self, workspace_root: Path | None) -> Path:
        if workspace_root is not None:
            return workspace_root.resolve()
        result = self._path_resolver.workspace_root()
        return result.value if result.is_success else Path.cwd().resolve()


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for check and pyrefly-fix commands."""
    parser = argparse.ArgumentParser(description="FLEXT check utilities")
    subparsers = parser.add_subparsers(dest="command")
    run_parser = subparsers.add_parser(c.Infra.Verbs.RUN, help="Run quality gates")
    _ = run_parser.add_argument("--gates", default=c.Infra.Gates.DEFAULT_CSV)
    _ = run_parser.add_argument("--project", action="append", required=True)
    _ = run_parser.add_argument(
        "--reports-dir", default=f"{c.Infra.Reporting.REPORTS_DIR_NAME}/check"
    )
    _ = run_parser.add_argument("--fail-fast", action="store_true")
    fix_parser = subparsers.add_parser(
        "fix-pyrefly-config", help="Repair [tool.pyrefly] blocks"
    )
    _ = fix_parser.add_argument("projects", nargs="*")
    _ = fix_parser.add_argument("--dry-run", action="store_true")
    _ = fix_parser.add_argument("--verbose", action="store_true")
    return parser


def run_cli(argv: list[str] | None = None) -> int:
    """Execute check service CLI commands and return process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == c.Infra.Verbs.RUN:
        checker = FlextInfraWorkspaceChecker()
        gates = FlextInfraWorkspaceChecker.parse_gate_csv(args.gates)
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
        run_results: list[m.Infra.Check.ProjectResult] = run_result.value
        failed_projects = [project for project in run_results if not project.passed]
        return 1 if failed_projects else 0
    if args.command == "fix-pyrefly-config":
        fixer = FlextInfraConfigFixer()
        fix_result = fixer.run(
            projects=args.projects, dry_run=args.dry_run, verbose=args.verbose
        )
        if fix_result.is_failure:
            output.error(fix_result.error or "pyrefly config fix failed")
            return 1
        return 0
    parser.print_help()
    return 1


_CheckIssue = m.Infra.Check.Issue
_GateExecution = m.Infra.Check.GateExecution
_ProjectResult = m.Infra.Check.ProjectResult
__all__ = [
    "FlextInfraConfigFixer",
    "FlextInfraWorkspaceChecker",
    "_CheckIssue",
    "_GateExecution",
    "_ProjectResult",
    "run_cli",
]
