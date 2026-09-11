"""jscpd duplicate-code detector — real cross-project clone findings, never silent.

Every clone jscpd finds is reported per project and ALSO emitted as a
``FlextSmellViolation`` warning on every run — warnings fire for all findings,
always, regardless of gate mode. ``c.Infra.DUPLICATION_GATE_MODE`` only decides
pass/fail: WARN is report-only while real findings are driven to zero project
by project (bead flext-qj62o), after which this flips to STRICT.

The jscpd config is rendered from this module's typed constants at scan time
and written under ``.reports/jscpd/`` — a runtime projection, never a second
hand-maintained source (the former committed ``.jscpd.json``/
``.jscpd-baseline.json`` pair is retired).
"""

from __future__ import annotations

import shutil
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, override

from flext_core import e as core_e
from flext_infra import c, m, u
from flext_infra.gates.base_gate import FlextInfraGate

if TYPE_CHECKING:
    from flext_infra import p, t


class FlextInfraDuplicationGate(FlextInfraGate):
    """Report jscpd clones per project from one process-cached workspace scan.

    A single jscpd scan covers every discovered workspace project's ``src``
    and ``tests`` trees so cross-project duplication clusters stay visible;
    per-project results are filtered by absolute path prefix. The scan output
    is cached per workspace root for the lifetime of the process (one scan
    per ``check`` run), matching ``FlextInfraSmellsGate``'s strategy.
    """

    gate_id: ClassVar[str] = "duplication"
    gate_name: ClassVar[str] = "Code Duplication"
    can_fix: ClassVar[bool] = False
    tool_name: ClassVar[str] = c.Infra.SARIF_TOOL_INFO["duplication"][0]
    tool_url: ClassVar[str] = c.Infra.SARIF_TOOL_INFO["duplication"][1]

    # flext-pulj: process results stay structural outside the Pydantic boundary.
    _scan_cache: ClassVar[dict[str, p.Cli.CommandOutput]] = {}

    @override
    def check(
        self, project_dir: Path, ctx: m.Infra.GateContext
    ) -> m.Infra.GateExecution:
        """One cached full-workspace jscpd scan, filtered to ``project_dir``."""
        _ = ctx
        started = time.monotonic()
        scan = self._workspace_scan()
        issues = self._issues_from_report(scan, project_dir)
        if not issues and scan.exit_code not in {0, 1}:
            issues = (self._tool_failure_issue(scan),)
        for issue in issues:
            warnings.warn(issue.formatted, core_e.SmellViolation, stacklevel=2)
        passed = c.Infra.DUPLICATION_GATE_MODE is c.Infra.GateMode.WARN or not issues
        return self._build_check_gate_execution(
            project_dir,
            passed=passed,
            issues=issues,
            raw_output=scan.stderr,
            started=started,
        )

    def _workspace_scan(self) -> p.Cli.CommandOutput:
        """Run the workspace jscpd scan once per process; a missing binary is VISIBLE."""
        # Fix-forward: FlextInfraGate exposes `_repository_root`, never
        # `_workspace_root` (see base_gate.py) — align with the base class.
        key = str(self._repository_root)
        cached = self._scan_cache.get(key)
        if cached is not None:
            return cached
        output = self._execute_scan()
        self._scan_cache[key] = output
        return output

    def _execute_scan(self) -> p.Cli.CommandOutput:
        """Render config, invoke jscpd, and load its JSON report from disk."""
        binary = self._resolve_binary()
        if binary is None:
            return m.Cli.CommandOutput(
                stdout="",
                stderr=(
                    f"{c.Infra.JSCPD_BINARY} not found on PATH; `make setup` "
                    "provisions it from codegen.toolchain.jscpd_version"
                ),
                # jscpd itself exits 1 when it finds clones, so an absent binary
                # must not borrow that code or the gate would read it as a scan.
                exit_code=c.Infra.PROCESS_COMMAND_NOT_FOUND_EXIT_CODE,
            )
        scope = self._render_scope_dirs()
        if not scope:
            return m.Cli.CommandOutput(stdout="{}", stderr="", exit_code=0)
        config_path = self._render_config()
        report_dir = self._repository_root / c.Infra.JSCPD_REPORT_DIRNAME
        cmd = (
            binary,
            "--no-gitignore",
            "--config",
            str(config_path),
            "--reporters",
            "json",
            "--output",
            str(report_dir),
            *scope,
        )
        result = self._run(cmd, self._repository_root, timeout=c.Infra.TIMEOUT_LONG)
        return self._load_report(report_dir, result)

    def _render_scope_dirs(self) -> t.StrSequence:
        """Every discovered workspace project's existing ``src``/``tests`` trees.

        Reuses the same workspace-topology discovery every other check-scoping
        path uses (``u.Infra.resolve_projects``) — never a second hardcoded
        project list.
        """
        discovered = u.Infra.resolve_projects(self._repository_root, ())
        if discovered.failure:
            return ()
        return tuple(
            str(project.path / candidate)
            for project in discovered.value
            for candidate in self._existing_check_dirs(project.path)
        )

    def _render_config(self) -> Path:
        """Materialize the jscpd config from this gate's typed SSOT.

        A generated-at-scan-time projection, never a second hand-edited
        source; regenerating with unchanged constants produces byte-identical
        content (idempotent).
        """
        ignore: list[t.JsonValue] = list(c.Infra.JSCPD_IGNORE_PATTERNS)
        config: t.JsonMapping = {
            "absolute": True,
            "noTips": True,
            "noColors": True,
            "mode": c.Infra.JSCPD_MODE,
            "ignore": ignore,
            "reporters": ["json"],
            "minLines": c.Infra.JSCPD_MIN_LINES,
        }
        config_path = (
            self._repository_root
            / c.Infra.JSCPD_REPORT_DIRNAME
            / c.Infra.JSCPD_CONFIG_FILENAME
        )
        _ = u.Cli.ensure_dir(config_path.parent).unwrap()
        _ = u.Cli.json_write(config_path, config).unwrap()
        return config_path

    @staticmethod
    def _load_report(
        report_dir: Path, result: p.Cli.CommandOutput
    ) -> p.Cli.CommandOutput:
        """Load the JSON report jscpd writes to disk; stdout carries console noise only."""
        report_path = report_dir / c.Infra.JSCPD_REPORT_FILENAME
        if not report_path.is_file():
            return result
        return m.Cli.CommandOutput(
            stdout=report_path.read_text(encoding="utf-8"),
            stderr=result.stderr,
            exit_code=result.exit_code,
        )

    @staticmethod
    def _resolve_binary() -> str | None:
        """Locate the mise-provisioned jscpd on PATH; None when absent."""
        return shutil.which(c.Infra.JSCPD_BINARY)

    def _tool_failure_issue(self, scan: p.Cli.CommandOutput) -> m.Infra.Issue:
        """Scanner absence/crash must never read as a clean pass."""
        return m.Infra.Issue(
            file=c.Infra.PYPROJECT_FILENAME,
            line=1,
            column=0,
            code=self.gate_id,
            message=scan.stderr or "jscpd execution failed",
            severity=self._severity(),
        )

    @staticmethod
    def _severity() -> str:
        """WARNING while report-only; ERROR once DUPLICATION_GATE_MODE is STRICT."""
        if c.Infra.DUPLICATION_GATE_MODE is c.Infra.GateMode.STRICT:
            return str(c.Infra.GateSeverity.ERROR.value)
        return str(c.Infra.GateSeverity.WARNING.value)

    @classmethod
    def _issues_from_report(
        cls, scan: p.Cli.CommandOutput, project_dir: Path
    ) -> tuple[m.Infra.Issue, ...]:
        """Extract one Issue per clone side that falls inside ``project_dir``."""
        parsed = u.Cli.json_parse(scan.stdout or "{}")
        empty_json: t.JsonValue = {}
        data = u.Cli.json_as_mapping(parsed.unwrap() if parsed.success else empty_json)
        prefix = str(project_dir)
        issues: list[m.Infra.Issue] = []
        for duplicate in u.Cli.json_deep_mapping_list(data, "duplicates"):
            first = u.Cli.json_deep_mapping(duplicate, "firstFile")
            second = u.Cli.json_deep_mapping(duplicate, "secondFile")
            first_name = u.Cli.json_pick_str(first, "name")
            second_name = u.Cli.json_pick_str(second, "name")
            if first_name.startswith(prefix):
                issues.append(
                    cls._issue_from_duplicate(
                        duplicate, first, first_name, second_name, project_dir
                    )
                )
            if second_name.startswith(prefix) and second_name != first_name:
                issues.append(
                    cls._issue_from_duplicate(
                        duplicate, second, second_name, first_name, project_dir
                    )
                )
        return tuple(issues)

    @classmethod
    def _issue_from_duplicate(
        cls,
        duplicate: t.JsonMapping,
        own_side: t.JsonMapping,
        own_name: str,
        other_name: str,
        project_dir: Path,
    ) -> m.Infra.Issue:
        """Map one clone side to an Issue naming its counterpart file/lines."""
        lines = u.Cli.json_pick_int(duplicate, "lines")
        tokens = u.Cli.json_pick_int(duplicate, "tokens")
        start_line = u.Cli.json_nested_int(own_side, "startLoc", "line", default=1)
        start_col = u.Cli.json_nested_int(own_side, "startLoc", "column")
        return m.Infra.Issue(
            file=own_name.removeprefix(f"{project_dir}/"),
            line=start_line,
            column=start_col,
            code=cls.gate_id,
            message=(
                f"{lines}-line ({tokens}-token) clone of {other_name} "
                f"— extend one owner, rewire consumers, delete the duplicate"
            ),
            severity=cls._severity(),
        )


__all__: list[str] = ["FlextInfraDuplicationGate"]
