"""FLEXT infrastructure workspace checker."""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import override

from pydantic import BaseModel, JsonValue, TypeAdapter, ValidationError

from flext_core import r, s, t
from flext_infra import (
    c,
    m,
    output,
    p,
    t as t_infra,
    u,
)
from flext_infra.check._constants import FlextInfraCheckConstants
from flext_infra.check.fix_pyrefly_config import FlextInfraConfigFixer


class FlextInfraWorkspaceChecker(s):
    """Run workspace quality gates and generate reports."""

    def __init__(self, workspace_root: Path | None = None) -> None:
        """Initialize workspace checker services and paths."""
        super().__init__(
            config_type=None,
            config_overrides=None,
            initial_context=None,
            subproject=None,
            services=None,
            factories=None,
            resources=None,
            container_overrides=None,
            wire_modules=None,
            wire_packages=None,
            wire_classes=None,
        )
        self._workspace_root = self._resolve_workspace_root(workspace_root)
        report_dir = u.Infra.get_report_dir(
            self._workspace_root,
            c.Infra.Toml.PROJECT,
            c.Infra.Verbs.CHECK,
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
                path.rglob("*.pyi"),
                None,
            ):
                out.append(directory)
        return out

    @staticmethod
    def generate_sarif_report(
        results: list[m.Infra.Check.ProjectResult],
        gates: list[str],
    ) -> JsonValue:
        """Generate a SARIF payload from gate results."""
        sarif_runs: list[m.Infra.Check.Sarif.Run] = []
        for gate in gates:
            tool_name, tool_url = FlextInfraCheckConstants.SARIF_TOOL_INFO.get(
                gate,
                (gate, ""),
            )
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
                                id=rule_id,
                                short_description=rule_id,
                            ),
                        )
                    sarif_results.append(
                        m.Infra.Check.Sarif.Result(
                            rule_id=rule_id,
                            level=c.Infra.Toml.ERROR
                            if issue.severity == c.Infra.Toml.ERROR
                            else c.Infra.Severity.WARNING,
                            message=issue.message,
                            locations=[
                                m.Infra.Check.Sarif.Location(
                                    uri=issue.file,
                                    start_line=max(issue.line, 1),
                                    start_column=max(issue.column, 1),
                                ),
                            ],
                        ),
                    )
            sarif_runs.append(
                m.Infra.Check.Sarif.Run(
                    tool_name=tool_name,
                    information_uri=tool_url,
                    rules=rules,
                    results=sarif_results,
                ),
            )
        sarif_report = m.Infra.Check.Sarif.Report(runs=sarif_runs)
        sarif_dict: dict[str, JsonValue] = sarif_report.model_dump(by_alias=True)
        return sarif_dict

    @staticmethod
    def parse_gate_csv(raw: str) -> list[str]:
        """Parse a comma-separated gate list."""
        return [gate.strip() for gate in raw.split(",") if gate.strip()]

    @staticmethod
    def resolve_gates(gates: Sequence[str]) -> r[list[str]]:
        """Resolve and validate requested gate names."""
        resolved: list[str] = []
        for gate in gates:
            name = gate.strip()
            if not name:
                continue
            mapped = c.Infra.Gates.PYREFLY if name == c.Infra.Gates.TYPE_ALIAS else name
            if mapped not in FlextInfraCheckConstants.ALLOWED_GATES:
                return r[list[str]].fail(f"ERROR: unknown gate '{gate}'")
            if mapped not in resolved:
                resolved.append(mapped)
        return r[list[str]].ok(resolved)

    @override
    def execute(
        self,
    ) -> r[t.NormalizedValue | BaseModel | list[t.NormalizedValue | BaseModel]]:
        return r[
            t.NormalizedValue | BaseModel | list[t.NormalizedValue | BaseModel]
        ].fail(
            "Use run() or run_projects() directly",
        )

    def format(self, project_dir: Path) -> r[m.Infra.Check.GateResult]:
        """Run format checks for one project."""
        return r[m.Infra.Check.GateResult].ok(self._run_ruff_format(project_dir).result)

    def generate_markdown_report(
        self,
        results: list[m.Infra.Check.ProjectResult],
        gates: list[str],
        timestamp: str,
    ) -> str:
        """Generate a markdown summary report for check results."""
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
            results,
            key=lambda item: item.total_errors,
            reverse=True,
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
                        f"... and {len(gate_result.issues) - c.Infra.Check.MAX_DISPLAY_ISSUES} more errors",
                    )
                lines.extend(["```", ""])
        return "\n".join(lines)

    def lint(self, project_dir: Path) -> r[m.Infra.Check.GateResult]:
        """Run lint checks for one project."""
        return r[m.Infra.Check.GateResult].ok(self._run_ruff_lint(project_dir).result)

    def run(
        self,
        project: str,
        gates: Sequence[str],
    ) -> r[list[m.Infra.Check.ProjectResult]]:
        """Run selected gates for one project."""
        return self.run_projects([project], list(gates)).map(lambda value: value)

    def run_projects(
        self,
        projects: Sequence[str],
        gates: Sequence[str],
        *,
        reports_dir: Path | None = None,
        fail_fast: bool = False,
    ) -> r[list[m.Infra.Check.ProjectResult]]:
        """Run selected gates for multiple projects."""
        resolved_gates_result = self.resolve_gates(gates)
        if resolved_gates_result.is_failure:
            return r[list[m.Infra.Check.ProjectResult]].fail(
                resolved_gates_result.error or "invalid gates",
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
                project_dir,
                resolved_gates,
                report_base,
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
        json_write_result = u.Infra.write_json(sarif_path, sarif_payload)
        if json_write_result.is_failure:
            return r[list[m.Infra.Check.ProjectResult]].fail(
                json_write_result.error or "failed to write sarif report",
            )
        total_errors = sum(project.total_errors for project in results)
        success = len(results) - failed
        output.summary(
            c.Infra.Verbs.CHECK,
            len(results),
            success,
            failed,
            skipped,
            total_elapsed,
        )
        output.info(f"Reports: {md_path}")
        output.info(f"         {sarif_path}")
        if total_errors > 0:
            output.info("Errors by project:")
            for project in sorted(
                results,
                key=lambda item: item.total_errors,
                reverse=True,
            ):
                if project.total_errors == 0:
                    continue
                breakdown = ", ".join(
                    f"{gate}={len(project.gates[gate].issues)}"
                    for gate in resolved_gates
                    if gate in project.gates and len(project.gates[gate].issues) > 0
                )
                output.error(
                    f"{project.project:30s} {project.total_errors:6d}  ({breakdown})",
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
            result=model,
            issues=issues,
            raw_output=raw_output,
        )

    def _check_project(
        self,
        project_dir: Path,
        gates: list[str],
        reports_dir: Path,
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
                execution = runner()
                result.gates[gate] = execution
                output.gate_result(
                    gate,
                    len(execution.issues),
                    execution.result.passed,
                    execution.result.duration,
                )
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
        result = u.Infra.workspace_root()
        return result.value if result.is_success else Path.cwd().resolve()

    @staticmethod
    def _to_mapping(
        value: t_infra.Infra.InfraValue,
    ) -> dict[str, t_infra.Infra.InfraValue]:
        if not isinstance(value, Mapping):
            return {}
        return TypeAdapter(dict[str, t_infra.Infra.InfraValue]).validate_python(value)

    @classmethod
    def _to_mapping_list(
        cls,
        value: t_infra.Infra.InfraValue,
    ) -> list[dict[str, t_infra.Infra.InfraValue]]:
        if not isinstance(value, list):
            return []
        typed_items = TypeAdapter(list[t_infra.Infra.InfraValue]).validate_python(value)
        normalized: list[dict[str, t_infra.Infra.InfraValue]] = []
        for raw_item in typed_items:
            try:
                typed_item = TypeAdapter(
                    dict[str, t_infra.Infra.InfraValue]
                ).validate_python(raw_item)
            except ValidationError:
                continue
            normalized.append(typed_item)
        return normalized

    @staticmethod
    def _as_int(value: t_infra.Infra.InfraValue, default: int = 0) -> int:
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
    def _as_str(value: t_infra.Infra.InfraValue, default: str = "") -> str:
        return value if isinstance(value, str) else default

    @staticmethod
    def _nested_mapping(
        data: dict[str, t_infra.Infra.InfraValue],
        *keys: str,
    ) -> dict[str, t_infra.Infra.InfraValue]:
        current: t_infra.Infra.InfraValue = data
        for key in keys:
            if not isinstance(current, Mapping):
                return {}
            typed_current = TypeAdapter(
                dict[str, t_infra.Infra.InfraValue]
            ).validate_python(current)
            if key not in typed_current:
                return {}
            child: t_infra.Infra.InfraValue = typed_current[key]
            if child is None:
                return {}
            current = child
        if not isinstance(current, Mapping):
            return {}
        return TypeAdapter(dict[str, t_infra.Infra.InfraValue]).validate_python(current)

    @classmethod
    def _nested_int(
        cls,
        data: dict[str, t_infra.Infra.InfraValue],
        *keys: str,
        default: int = 0,
    ) -> int:
        target = cls._nested_mapping(data, *keys[:-1])
        raw: t_infra.Infra.InfraValue = target.get(keys[-1])
        if raw is None:
            return default
        return cls._as_int(raw, default)

    @classmethod
    def _result_exit_code(cls, result: p.Infra.CommandOutput) -> int:
        try:
            payload = TypeAdapter(dict[str, t_infra.Infra.InfraValue]).validate_python(
                vars(result),
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
        result = u.Infra.run_raw(cmd, cwd=cwd, timeout=timeout, env=env)
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
        bandit_data: dict[str, t_infra.Infra.InfraValue] = {}
        try:
            parsed = u.Infra.parse(result.stdout or "{}")
            if parsed.is_success and isinstance(parsed.value, Mapping):
                bandit_data = self._to_mapping(parsed.value)
            raw_results: list[dict[str, t_infra.Infra.InfraValue]] = (
                self._to_mapping_list(
                    bandit_data.get("results", []),
                )
            )
            issues.extend(
                m.Infra.Check.Issue(
                    file=self._as_str(raw_item.get("filename", "?"), "?"),
                    line=self._as_int(raw_item.get("line_number", 0)),
                    column=0,
                    code=self._as_str(raw_item.get("test_id", "")),
                    message=self._as_str(raw_item.get("issue_text", "")),
                    severity=self._as_str(
                        raw_item.get("issue_severity", "MEDIUM"),
                        "MEDIUM",
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
                ),
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
                ),
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
                    ),
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
                ),
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
                ),
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
                line_data = TypeAdapter(
                    dict[str, t_infra.Infra.InfraValue]
                ).validate_json(
                    stripped,
                )
            except ValidationError:
                continue
            try:
                severity = self._as_str(
                    line_data.get("severity", c.Infra.Toml.ERROR),
                    c.Infra.Toml.ERROR,
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
                        ),
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
        self,
        project_dir: Path,
        reports_dir: Path,
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
                parsed = u.Infra.parse(raw_text)
                if parsed.is_success and isinstance(parsed.value, Mapping):
                    parsed_map = self._to_mapping(parsed.value)
                    error_items: list[dict[str, t_infra.Infra.InfraValue]] = (
                        self._to_mapping_list(parsed_map.get("errors", []))
                    )
                elif parsed.is_success and isinstance(parsed.value, list):
                    error_items = self._to_mapping_list(parsed.value)
                else:
                    error_items = []
                issues.extend(
                    m.Infra.Check.Issue(
                        file=self._as_str(item.get("path"), "?"),
                        line=self._as_int(item.get("line"), 0),
                        column=self._as_int(item.get("column"), 0),
                        code=self._as_str(item.get("name"), ""),
                        message=self._as_str(item.get("description"), ""),
                        severity=self._as_str(item.get("severity"), c.Infra.Toml.ERROR),
                    )
                    for err in error_items
                    for item in [dict(err)]
                )
            except (TypeError, ValidationError):
                pass
        if not issues and self._result_exit_code(result) != 0:
            match = re.search(r"(\d+)\s+errors?", result.stderr + result.stdout)
            if match:
                count = int(match.group(1))
                issues = [
                    m.Infra.Check.Issue(
                        file="?",
                        line=0,
                        column=0,
                        code=c.Infra.Gates.PYREFLY,
                        message=f"Pyrefly reported {count} error(s)",
                    ),
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
            project_dir,
            self._existing_check_dirs(project_dir),
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
        pyright_parse_result = u.Infra.parse(result.stdout or "{}")
        pyright_data: dict[str, t_infra.Infra.InfraValue] = self._to_mapping(
            pyright_parse_result.value if pyright_parse_result.is_success else {},
        )
        try:
            raw_diagnostics: list[dict[str, t_infra.Infra.InfraValue]] = (
                self._to_mapping_list(
                    pyright_data.get("generalDiagnostics", []),
                )
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
                        ),
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
                        ),
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
        ruff_parse_result = u.Infra.parse(result.stdout or "[]")
        ruff_data: t_infra.Infra.InfraValue = (
            ruff_parse_result.value if ruff_parse_result.is_success else []
        )
        try:
            if isinstance(ruff_data, list):
                issues.extend(
                    m.Infra.Check.Issue(
                        file=str(entry.get("filename", "?")),
                        line=self._nested_int(dict(entry.items()), "location", "row"),
                        column=self._nested_int(
                            dict(entry.items()), "location", "column"
                        ),
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


def build_parser() -> argparse.ArgumentParser:
    """Build the workspace check CLI parser."""
    parser, subs = u.Infra.create_subcommand_parser(
        "flext-infra check",
        "FLEXT check utilities",
        subcommands={
            c.Infra.Verbs.RUN: "Run quality gates",
            "fix-pyrefly-config": "Repair [tool.pyrefly] blocks",
        },
        include_apply=True,
    )
    _ = subs[c.Infra.Verbs.RUN].add_argument(
        "--gates", default=c.Infra.Gates.DEFAULT_CSV
    )
    _ = subs[c.Infra.Verbs.RUN].add_argument(
        "--project",
        action="append",
        required=True,
    )
    _ = subs[c.Infra.Verbs.RUN].add_argument(
        "--reports-dir",
        default=f"{c.Infra.Reporting.REPORTS_DIR_NAME}/check",
    )
    _ = subs[c.Infra.Verbs.RUN].add_argument("--fail-fast", action="store_true")
    _ = subs["fix-pyrefly-config"].add_argument("projects", nargs="*")
    _ = subs["fix-pyrefly-config"].add_argument("--verbose", action="store_true")
    return parser


def run_cli(argv: list[str] | None = None) -> int:
    """Run the subcommand-based workspace check CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    cli = u.Infra.resolve(args)
    if args.command == c.Infra.Verbs.RUN:
        checker = FlextInfraWorkspaceChecker(workspace_root=cli.workspace)
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


def main(argv: list[str] | None = None) -> int:
    """Run the legacy workspace check CLI entrypoint."""
    parser = u.Infra.create_parser(
        "flext-infra check-workspace",
        "FLEXT Workspace Check",
        include_apply=False,
    )
    _ = parser.add_argument("projects", nargs="*")
    _ = parser.add_argument("--gates", default=c.Infra.Gates.DEFAULT_CSV)
    _ = parser.add_argument(
        "--reports-dir",
        default=f"{c.Infra.Reporting.REPORTS_DIR_NAME}/check",
    )
    _ = parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args(argv)
    if not args.projects:
        output.error("no projects specified")
        return 1
    checker = FlextInfraWorkspaceChecker()
    gates = FlextInfraWorkspaceChecker.parse_gate_csv(args.gates)
    reports_dir = Path(args.reports_dir).expanduser()
    if not reports_dir.is_absolute():
        reports_dir = (Path.cwd() / reports_dir).resolve()
    result = checker.run_projects(
        projects=args.projects,
        gates=gates,
        reports_dir=reports_dir,
        fail_fast=args.fail_fast,
    )
    if result.is_failure:
        output.error(result.error or "workspace check failed")
        return 2
    projects = result.value
    failed_projects = [project for project in projects if not project.passed]
    return 1 if failed_projects else 0


__all__ = ["FlextInfraWorkspaceChecker", "build_parser", "main", "run_cli"]
