"""FLEXT pyright quality gate."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from typing import TYPE_CHECKING, ClassVar, override

from flext_infra import c, m, u
from flext_infra.gates.base_gate import FlextInfraGate

if TYPE_CHECKING:
    from pathlib import Path

    from flext_infra import p, t


class FlextInfraPyrightGate(FlextInfraGate):
    """Pyright quality gate."""

    gate_id: ClassVar[str] = c.Infra.PYRIGHT
    gate_name: ClassVar[str] = "Pyright"
    can_fix: ClassVar[bool] = False
    tool_name: ClassVar[str] = c.Infra.SARIF_TOOL_INFO[c.Infra.PYRIGHT][0]
    tool_url: ClassVar[str] = c.Infra.SARIF_TOOL_INFO[c.Infra.PYRIGHT][1]

    @override
    def _get_check_dirs(
        self, project_dir: Path, ctx: m.Infra.GateContext
    ) -> t.StrSequence:
        """Use the project pyright config as SSOT when it exists."""
        _ = ctx
        if self._has_project_pyright_config(project_dir):
            return [c.Infra.PYRIGHT_PROJECT_ARG, c.Infra.PYRIGHT_PROJECT_CONFIG_TARGET]
        return super()._get_check_dirs(project_dir, ctx)

    @override
    def _build_check_command(
        self, project_dir: Path, ctx: m.Infra.GateContext, check_dirs: t.StrSequence
    ) -> t.StrSequence:
        """Build check command."""
        _ = project_dir
        return self._python_module_command(
            c.Infra.PYRIGHT,
            "--pythonpath",
            sys.executable,
            *check_dirs,
            *ctx.pyright_args,
            "--outputjson",
        )

    @staticmethod
    def _has_project_pyright_config(project_dir: Path) -> bool:
        """Return whether pyproject.toml declares [tool.pyright]."""
        doc = u.Cli.toml_read(project_dir / c.Infra.PYPROJECT_FILENAME)
        if doc is None:
            return False
        tool_table = u.Cli.toml_table_child(doc, c.Infra.TOOL)
        return (
            tool_table is not None
            and u.Cli.toml_table_child(tool_table, c.Infra.PYRIGHT) is not None
        )

    @override
    def _check_timeout(self, project_dir: Path, ctx: m.Infra.GateContext) -> int:
        """Check timeout."""
        _ = project_dir, ctx
        timeout: int = c.Infra.TIMEOUT_LONG
        return timeout

    @override
    def _parse_check_output(
        self, result: p.Cli.CommandOutput, project_dir: Path, ctx: m.Infra.GateContext
    ) -> tuple[bool, t.SequenceOf[m.Infra.Issue]]:
        """Parse check output."""
        _ = project_dir, ctx
        issues: t.MutableSequenceOf[m.Infra.Issue] = []
        empty: t.MappingKV[str, t.Infra.InfraValue] = {}
        parsed_result = u.Cli.json_parse(result.stdout or "{}")
        parsed = parsed_result.unwrap() if parsed_result.success else empty
        data = u.Cli.json_as_mapping(parsed) if isinstance(parsed, Mapping) else empty
        try:
            diagnostics = u.Cli.json_deep_mapping_list(
                data, c.Infra.PYRIGHT_DIAGNOSTICS_KEY
            )
            issues.extend(
                m.Infra.Issue(
                    file=u.Cli.json_pick_str(diag, "file", "?"),
                    line=u.Cli.json_nested_int(diag, "range", "start", "line") + 1,
                    column=u.Cli.json_nested_int(diag, "range", "start", "character")
                    + 1,
                    code=u.Cli.json_pick_str(diag, "rule"),
                    message=u.Cli.json_pick_str(diag, "message"),
                    severity=u.Cli.json_pick_str(diag, "severity", c.Infra.ERROR),
                )
                for diag in diagnostics
            )
        except c.EXC_VALIDATION_TYPE as err:
            issues.append(
                m.Infra.Issue(
                    file="<pyright-output>",
                    line=0,
                    column=0,
                    code="PARSE_ERROR",
                    message=f"Tool output parsing failed: {type(err).__name__}",
                    severity="ERROR",
                )
            )
            return False, issues
        if (not issues) and result.exit_code != 0:
            message = (result.stderr or result.stdout).strip()
            if not message:
                message = (
                    f"pyright exited with code {result.exit_code} "
                    "without JSON diagnostics"
                )
            issues.append(
                m.Infra.Issue(
                    file=c.Infra.PYPROJECT_FILENAME,
                    line=1,
                    column=1,
                    code="pyright-exec",
                    message=message,
                    severity=c.Infra.ERROR,
                )
            )
        return result.exit_code == 0, issues


__all__: list[str] = ["FlextInfraPyrightGate"]
