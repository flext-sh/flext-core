"""Sandbox-backed multi-project orchestration.

Mirrors the live workspace into a disposable sandbox (typically
``flext`` → ``flext-base`` via rsync), runs the requested make verb on
every discovered project there, and on success replays the same verb
against the live workspace. On sandbox failure the live workspace is
never touched and the sandbox itself is reverted to git HEAD per
project so the next run starts clean.

Composes the existing ``FlextInfraOrchestratorService`` for the
per-project make execution; this service only adds the snapshot,
validation, propagation, and rollback edges around it.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, override

from flext_core import r
from flext_infra import m, u
from flext_infra._utilities.snapshot import FlextInfraUtilitiesSnapshot
from flext_infra.workspace.orchestrator import FlextInfraOrchestratorService

if TYPE_CHECKING:
    from pathlib import Path

    from flext_infra import p


class FlextInfraSandboxOrchestrator(FlextInfraOrchestratorService):
    """Run a make verb in ``sandbox_root`` first; replay in ``self.root`` on success."""

    sandbox_root: Annotated[
        Path,
        m.Field(
            description="Disposable sandbox repository root (default: <root>-base)"
        ),
    ]

    @property
    def _sandbox_path(self) -> Path:
        """Sandbox path."""
        return self.sandbox_root.resolve()

    def _snapshot(self) -> p.Result[Path]:
        """Snapshot."""
        return FlextInfraUtilitiesSnapshot.rsync(
            src=self.root.resolve(), dst=self._sandbox_path
        )

    def _orchestrate_in(self, repository_root: Path) -> p.Result[bool]:
        """Orchestrate in."""
        clone = self.model_copy(update={"repository_root": repository_root})
        return FlextInfraOrchestratorService.execute(clone)

    def _rollback_sandbox(self) -> p.Result[bool]:
        """Rollback sandbox."""
        for project_root in u.Infra.discover_project_roots(self._sandbox_path):
            restored = u.Infra.git_checkout_restore(
                m.Infra.GitRepoRequest(repo_root=project_root)
            )
            if restored.failure:
                return r[bool].fail(
                    restored.error or f"sandbox rollback ({project_root.name}) failed"
                )
        return r[bool].ok(True)

    @override
    def execute(self) -> p.Result[bool]:
        """Snapshot → run-in-sandbox → propagate-to-live or rollback."""
        snapshot_result = self._snapshot()
        if snapshot_result.failure:
            return r[bool].fail(snapshot_result.error or "sandbox snapshot failed")
        sandbox_result = self._orchestrate_in(self._sandbox_path)
        if sandbox_result.failure:
            rollback = self._rollback_sandbox()
            if rollback.failure:
                return r[bool].fail(
                    f"sandbox failed and rollback failed: "
                    f"{sandbox_result.error or '<no detail>'} | "
                    f"{rollback.error or '<no detail>'}"
                )
            return r[bool].fail(
                f"sandbox '{self.verb}' failed; sandbox reverted, live workspace untouched: "
                f"{sandbox_result.error or '<no detail>'}"
            )
        return self._orchestrate_in(self.root.resolve())


__all__: list[str] = ["FlextInfraSandboxOrchestrator"]
