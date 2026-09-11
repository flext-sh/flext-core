"""Abstraction-boundary quality gate (AGENTS.md §2.7).

Single SSOT replacing the two standalone scripts ``audit_banned_cli_libs.py``
and ``audit_flext_cli_concrete_imports.py``. All detection data (banned libs,
regex catalog, exemptions) lives in ``c.Infra.BOUNDARY_*`` (CONSTANTS-FIRST);
this class is the thin, data-driven scanner. For every project except
``flext-cli``/``flext-core`` it flags CLI-domain lib imports (``click`` exempt
in Singer-SDK boundary files), ``subprocess``, ``tomllib``/``tomlkit`` outside
``flext-infra``, direct ``json.``/``yaml.``/``csv.`` use, top-level ``u.Cli.print(``/
``sys.exit(``, and concrete ``FlextCli<X>`` imports outside src extension files.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, ClassVar, override

from flext_infra import c, m, u
from flext_infra.gates.base_gate import FlextInfraGate

if TYPE_CHECKING:
    from pathlib import Path

    from flext_infra import t


class FlextInfraAbstractionBoundaryGate(FlextInfraGate):
    """Block CLI-domain library leakage and concrete FlextCli imports in consumers."""

    gate_id: ClassVar[str] = "boundary"
    gate_name: ClassVar[str] = "Abstraction Boundary"
    can_fix: ClassVar[bool] = False
    tool_name: ClassVar[str] = c.Infra.SARIF_TOOL_INFO["boundary"][0]
    tool_url: ClassVar[str] = c.Infra.SARIF_TOOL_INFO["boundary"][1]

    @override
    def check(
        self, project_dir: Path, ctx: m.Infra.GateContext
    ) -> m.Infra.GateExecution:
        """Scan one project's Python sources for abstraction-boundary breaches."""
        _ = ctx
        started = time.monotonic()
        if project_dir.name in c.Infra.BOUNDARY_SKIP_PROJECTS:
            return self._skip_result(project_dir, started)
        files_result = u.Infra.iter_python_files(
            m.Infra.SourceScanRequest(project_roots=(project_dir,))
        )
        if files_result.failure:
            issue = m.Infra.Issue(
                file=project_dir.name,
                line=1,
                column=1,
                code=self.gate_id,
                message=files_result.error or "abstraction-boundary scan failed",
            )
            return self._build_check_gate_execution(
                project_dir,
                passed=False,
                issues=[issue],
                raw_output=issue.formatted,
                started=started,
            )
        issues = [
            issue
            for file_path in files_result.value
            for issue in self._scan_file(file_path, project_dir.name)
        ]
        return self._build_check_gate_execution(
            project_dir,
            passed=len(issues) == 0,
            issues=issues,
            raw_output="\n".join(issue.formatted for issue in issues),
            started=started,
        )

    def _scan_file(self, path: Path, project: str) -> t.SequenceOf[m.Infra.Issue]:
        """Return boundary violations for a single file via the c.Infra catalog."""
        read = u.Cli.files_read_text(path)
        if read.failure:
            return [self._issue(path, read.error or "unreadable source file")]
        text = read.value
        posix = str(path).replace("\\", "/")
        if any(frag in posix for frag in c.Infra.BOUNDARY_SELF_FILES):
            return []
        issues: t.MutableSequenceOf[m.Infra.Issue] = []
        click_ok = any(frag in posix for frag in c.Infra.BOUNDARY_CLICK_FILES)
        for lib, regex, replacement in c.Infra.BOUNDARY_BANNED_RULES:
            if lib == "click" and click_ok:
                continue
            if regex.search(text):
                issues.append(self._issue(path, f"imports `{lib}` — use {replacement}"))
        for regex, message in c.Infra.BOUNDARY_SIMPLE_RULES:
            if regex.search(text):
                issues.append(self._issue(path, message))
        if (
            c.Infra.BOUNDARY_TOML_RE.search(text)
            and project not in c.Infra.BOUNDARY_TOML_ALLOWED
        ):
            issues.append(
                self._issue(path, "imports tomllib/tomlkit — use cli.read_toml_file")
            )
        issues.extend(self._concrete_cli_issues(path, text, posix))
        return issues

    def _concrete_cli_issues(
        self, path: Path, text: str, posix: str
    ) -> t.SequenceOf[m.Infra.Issue]:
        """Flag concrete FlextCli<X> imports outside src extension files."""
        if "/src/" in posix and path.name in c.Infra.BOUNDARY_EXTENSION_FILES:
            return ()
        issues: t.MutableSequenceOf[m.Infra.Issue] = []
        for match in c.Infra.BOUNDARY_CONCRETE_IMPORT_RE.finditer(text):
            for name in c.Infra.BOUNDARY_FLEXT_CLI_CONCRETE_RE.findall(
                match.group("imports")
            ):
                issues.append(
                    self._issue(
                        path, f"imports concrete `{name}` (use cli/c/m/p/t/u/s)"
                    )
                )
        return issues

    def _issue(self, path: Path, message: str) -> m.Infra.Issue:
        """Build a boundary Issue anchored at the file head."""
        return m.Infra.Issue(
            file=str(path),
            line=1,
            column=1,
            code=self.gate_id,
            message=message,
            severity="ERROR",
        )


__all__: list[str] = ["FlextInfraAbstractionBoundaryGate"]
