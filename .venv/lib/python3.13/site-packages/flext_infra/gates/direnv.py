"""FLEXT direnv environment contract gate.

Two fail-closed stages per checked workspace: the static environment-file
contracts (see ``flext_infra.workspace.environment_contracts``) followed by a
real ``direnv exec`` activation smoke. A workspace without ``.envrc`` skips.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, ClassVar, override

from flext_infra import c, m, u
from flext_infra.gates.base_gate import FlextInfraGate
from flext_infra.workspace.environment_contracts import envrc_contract_violations

if TYPE_CHECKING:
    from pathlib import Path

    from flext_infra import p, t


class FlextInfraDirenvGate(FlextInfraGate):
    """Enforce direnv file contracts, then prove real activation."""

    gate_id: ClassVar[str] = "direnv"
    gate_name: ClassVar[str] = "DIRENV ENVIRONMENT CONTRACT"
    can_fix: ClassVar[bool] = False
    tool_name: ClassVar[str] = c.Infra.SARIF_TOOL_INFO["direnv"][0]
    tool_url: ClassVar[str] = c.Infra.SARIF_TOOL_INFO["direnv"][1]

    @override
    def check(
        self, project_dir: Path, ctx: m.Infra.GateContext
    ) -> m.Infra.GateExecution:
        """Run the static contracts, then the activation smoke."""
        started = time.monotonic()
        envrc = project_dir / c.Infra.ENVRC_FILENAME
        if not envrc.is_file():
            return self._skip_result(project_dir, started)
        content = u.Cli.files_read_text(envrc)
        if content.failure:
            issue = m.Infra.Issue(
                file=c.Infra.ENVRC_FILENAME,
                line=0,
                column=0,
                code="DIRENV_READ",
                message=content.error or f"cannot read {envrc}",
                severity="ERROR",
            )
            return self._build_check_gate_execution(
                project_dir,
                passed=False,
                issues=(issue,),
                raw_output=issue.message,
                started=started,
            )
        violations = envrc_contract_violations(content.value, root=project_dir)
        if violations:
            issues = tuple(
                m.Infra.Issue(
                    file=c.Infra.ENVRC_FILENAME,
                    line=0,
                    column=0,
                    code="DIRENV_CONTRACT",
                    message=violation,
                    severity="ERROR",
                )
                for violation in violations
            )
            return self._build_check_gate_execution(
                project_dir,
                passed=False,
                issues=issues,
                raw_output="\n".join(violations),
                started=started,
            )
        return super().check(project_dir, ctx)

    @override
    def _get_check_dirs(
        self, project_dir: Path, ctx: m.Infra.GateContext
    ) -> t.StrSequence:
        """One marker dir drives the base flow; the smoke targets the root."""
        _ = ctx
        return (
            (str(project_dir),)
            if (project_dir / c.Infra.ENVRC_FILENAME).exists()
            else ()
        )

    @override
    def _build_check_command(
        self, project_dir: Path, ctx: m.Infra.GateContext, check_dirs: t.StrSequence
    ) -> t.StrSequence:
        """Activate the workspace environment for one no-op command."""
        _ = ctx, check_dirs
        return (c.Infra.CLI_DIRENV, "exec", str(project_dir), "true")

    @override
    def _parse_check_output(
        self, result: p.Cli.CommandOutput, project_dir: Path, ctx: m.Infra.GateContext
    ) -> tuple[bool, t.SequenceOf[m.Infra.Issue]]:
        """Pass only on a zero-exit activation."""
        _ = project_dir, ctx
        if result.exit_code == 0:
            return True, ()
        detail = result.stderr.strip() or result.stdout.strip() or "direnv exec failed"
        return (
            False,
            (
                m.Infra.Issue(
                    file=c.Infra.ENVRC_FILENAME,
                    line=0,
                    column=0,
                    code="DIRENV_ACTIVATE",
                    message=detail,
                    severity="ERROR",
                ),
            ),
        )


__all__: tuple[str, ...] = ("FlextInfraDirenvGate",)
