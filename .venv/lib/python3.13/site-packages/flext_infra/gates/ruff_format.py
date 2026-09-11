"""Ruff format gate implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, override

from flext_infra import c, m
from flext_infra.gates.base_gate import FlextInfraGate

if TYPE_CHECKING:
    from pathlib import Path

    from flext_infra import p, t


class FlextInfraRuffFormatGate(FlextInfraGate):
    """Gate for Ruff formatter checks and fixes."""

    gate_id: ClassVar[str] = c.Infra.FORMAT
    gate_name: ClassVar[str] = "Ruff Format"
    can_fix: ClassVar[bool] = True
    tool_name: ClassVar[str] = c.Infra.SARIF_TOOL_INFO[c.Infra.FORMAT][0]
    tool_url: ClassVar[str] = c.Infra.SARIF_TOOL_INFO[c.Infra.FORMAT][1]

    @override
    def _get_check_dirs(
        self, project_dir: Path, ctx: m.Infra.GateContext
    ) -> t.StrSequence:
        """Get check dirs."""
        _ = ctx
        return self._existing_check_dirs(project_dir) or ["."]

    @override
    def _build_check_command(
        self, project_dir: Path, ctx: m.Infra.GateContext, check_dirs: t.StrSequence
    ) -> t.StrSequence:
        """Build check command."""
        _ = project_dir, ctx
        return self._python_module_command(
            c.Infra.RUFF, c.Infra.FORMAT, "--check", *check_dirs, "--quiet"
        )

    @override
    def _parse_check_output(
        self, result: p.Cli.CommandOutput, project_dir: Path, ctx: m.Infra.GateContext
    ) -> tuple[bool, t.SequenceOf[m.Infra.Issue]]:
        """Parse check output."""
        _ = project_dir, ctx
        issues: t.MutableSequenceOf[m.Infra.Issue] = []
        if result.exit_code != 0 and result.stdout.strip():
            seen: t.Infra.StrSet = set()
            for line in result.stdout.strip().splitlines():
                raw = line.strip()
                if not raw:
                    continue
                match = c.Infra.RUFF_FORMAT_FILE_RE.match(raw)
                resolved = match.group(1).strip() if match else raw
                if not resolved or resolved in seen:
                    continue
                if match or (
                    resolved.endswith(c.Infra.EXT_PYTHON) and " " not in resolved
                ):
                    seen.add(resolved)
                    issues.append(
                        m.Infra.Issue(
                            file=resolved,
                            line=0,
                            column=0,
                            code=c.Infra.FORMAT,
                            message="Would be reformatted",
                        )
                    )
        return result.exit_code == 0, issues

    @override
    def _build_fix_command(
        self, project_dir: Path, ctx: m.Infra.GateContext, targets: t.StrSequence
    ) -> t.StrSequence:
        """Build fix command."""
        _ = project_dir, ctx
        return self._python_module_command(c.Infra.RUFF, c.Infra.FORMAT, *targets)


__all__: list[str] = ["FlextInfraRuffFormatGate"]
