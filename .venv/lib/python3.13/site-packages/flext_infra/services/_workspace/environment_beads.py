"""Beads-workspace environment sync: generated Gas City + Beads activation.

Owns the generated ``.envrc`` for workspaces whose activation is the Beads /
Gas City wiring (identity var, managed Dolt state) instead of the Python
package environment, plus the post-sync ``direnv allow`` that keeps every
managed root free of a stale allow.
"""

from __future__ import annotations

from typing import ClassVar, override

from flext_infra import c, m, p, r, t, u
from flext_infra.base import s
from flext_infra.services._workspace.environment import (
    FlextInfraWorkspaceEnvironmentMixin,
)
from flext_infra.workspace.environment_contracts import envrc_contract_violations


class FlextInfraWorkspaceBeadsEnvironmentMixin(FlextInfraWorkspaceEnvironmentMixin):
    """Generated beads-workspace activation + post-sync direnv allow."""

    _BEADS_ENVRC_TEMPLATE: ClassVar[str] = ".envrc.beads-workspace"

    @classmethod
    @override
    def sync_environment_files(
        cls,
        request: m.Infra.WorkspaceEnvironmentSyncRequest,
        *,
        runner: p.Cli.CommandRunner | None = None,
    ) -> p.Result[m.Infra.WorkspaceEnvironmentSyncResult]:
        """Dispatch beads workspaces, then heal the direnv allow state."""
        if request.beads is not None:
            beads_result = cls._sync_beads_environment(request)
            if beads_result.failure:
                return beads_result
            allow_result = cls._allow_direnv_if_requested(request, runner=runner)
            if allow_result.failure:
                return r[m.Infra.WorkspaceEnvironmentSyncResult].fail(
                    allow_result.error or "direnv allow failed"
                )
            return beads_result
        result = super().sync_environment_files(request)
        if result.failure:
            return result
        allow_result = cls._allow_direnv_if_requested(request, runner=runner)
        if allow_result.failure:
            return r[m.Infra.WorkspaceEnvironmentSyncResult].fail(
                allow_result.error or "direnv allow failed"
            )
        return result

    @classmethod
    def _sync_beads_environment(
        cls, request: m.Infra.WorkspaceEnvironmentSyncRequest
    ) -> p.Result[m.Infra.WorkspaceEnvironmentSyncResult]:
        """Sync one generated beads-workspace ``.envrc`` end to end."""
        result_type = m.Infra.WorkspaceEnvironmentSyncResult
        beads = request.beads
        if beads is None:
            return r[result_type].fail("beads environment sync requires beads spec")
        rendered = cls._render_environment_template(
            cls._BEADS_ENVRC_TEMPLATE, context=beads
        )
        if rendered.failure:
            return r[result_type].fail(
                rendered.error or "beads-workspace template render failed"
            )
        violations = envrc_contract_violations(
            rendered.value, root=request.repository_root, resolve_home=False
        )
        if violations:
            return r[result_type].fail(
                "generated beads-workspace .envrc violates contracts: "
                + "; ".join(violations)
            )
        envrc = request.repository_root / c.Infra.ENVRC_FILENAME
        written = cls._write_generated_text(
            envrc, rendered.value, apply=request.apply, force=request.force
        )
        if written.failure:
            return r[result_type].fail(written.error or ".envrc write failed")
        changed = (envrc,) if written.value else ()
        return r[result_type].ok(result_type(changed_files=changed))

    @classmethod
    def _allow_direnv_if_requested(
        cls,
        request: m.Infra.WorkspaceEnvironmentSyncRequest,
        *,
        runner: p.Cli.CommandRunner | None = None,
    ) -> p.Result[bool]:
        """Run ``direnv allow`` for one applied sync that owns the envrc."""
        envrc = request.repository_root / c.Infra.ENVRC_FILENAME
        if not request.apply or not request.allow_direnv or not envrc.is_file():
            return r[bool].ok(False)
        runner_service = runner or u.Cli
        result = runner_service.run_raw(
            (c.Infra.CLI_DIRENV, "allow", str(request.repository_root)),
            cwd=request.repository_root,
            timeout=c.Infra.TIMEOUT_DEFAULT,
        )
        if result.failure:
            return r[bool].fail(result.error or "direnv allow execution failed")
        output = result.value
        if output.exit_code != 0:
            return r[bool].fail(
                f"direnv allow failed for {request.repository_root}: "
                f"{output.stderr.strip() or output.stdout.strip()}"
            )
        return r[bool].ok(True)


class FlextInfraWorkspaceEnvironmentSync(
    FlextInfraWorkspaceBeadsEnvironmentMixin, s[t.JsonDict]
):
    """CLI-facing service composing the full environment sync surface."""

    @classmethod
    def execute_request(
        cls, request: m.Infra.WorkspaceEnvironmentSyncRequest
    ) -> p.Result[t.Cli.ResultValue]:
        """Run one sync request through the composed mixin surface."""
        result = cls.sync_environment_files(request)
        if result.failure:
            return r[t.Cli.ResultValue].fail(result.error or "environment sync failed")
        return r[t.Cli.ResultValue].ok(
            tuple(str(path) for path in result.value.changed_files)
        )


__all__: tuple[str, ...] = (
    "FlextInfraWorkspaceBeadsEnvironmentMixin",
    "FlextInfraWorkspaceEnvironmentSync",
)
