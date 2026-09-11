"""Destination-local staging for complete newest-Mise artifact sets."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from flext_core import r
from flext_infra import config, m
from flext_infra.codegen import _mise_artifacts_candidates as candidates
from flext_infra.codegen import _mise_artifacts_files as files
from flext_infra.codegen import _mise_artifacts_process as process
from flext_infra.codegen._mise_artifacts_runtime import FlextInfraMiseRuntime

if TYPE_CHECKING:
    from flext_infra import p


class FlextInfraMiseStaging:
    """Build every replacement before the journal permits live publication."""

    def __init__(self, owner: p.Infra.MiseArtifactsOwner) -> None:
        self._owner = owner
        self._runtime = FlextInfraMiseRuntime(owner)

    def stage(
        self, plan: m.Infra.MiseToolchainWorkspacePlan, *, credential_command: str
    ) -> p.Result[tuple[m.Infra.MiseToolchainPublication, ...]]:
        """Generate, hydrate, and validate all destination-local candidates."""
        receipt = self._runtime.latest_receipt(
            plan.projects[0], credential_command=credential_command
        )
        if receipt.failure:
            return r[tuple[m.Infra.MiseToolchainPublication, ...]].from_failure(receipt)
        runtime_scratch = plan.projects[0].layout.transaction_root / "runtime"
        environment = process.credential_environment(
            runtime_scratch, credential_command
        )
        launcher = receipt.value / "bin" / ("mise.cmd" if os.name == "nt" else "mise")
        credential = process.validate_credential_source(
            launcher, cwd=runtime_scratch, env=environment
        )
        if credential.failure:
            return r[tuple[m.Infra.MiseToolchainPublication, ...]].from_failure(
                credential
            )
        receipt_states = candidates.receipt_states(receipt.value)
        if receipt_states.failure:
            return r[tuple[m.Infra.MiseToolchainPublication, ...]].from_failure(
                receipt_states
            )
        stages: list[Path] = []
        for project in plan.projects:
            stage_root = project.layout.transaction_root / "stage"
            staged = self._stage_project(
                project,
                stage_root=stage_root,
                launcher=launcher,
                receipt_states=receipt_states.value,
                environment=environment,
            )
            if staged.failure:
                return r[tuple[m.Infra.MiseToolchainPublication, ...]].from_failure(
                    staged
                )
            stages.append(stage_root)
        return candidates.publication_plan(plan.projects, tuple(stages))

    @staticmethod
    def _lock_resolution_is_required(
        project: m.Infra.MiseToolchainProjectState,
    ) -> bool:
        """Return whether this project's lock must be resolved from the network.

        ``mise lock --bump`` re-resolves every declared selector against its
        remote registry. That is required exactly when the declaration changed
        or no lock exists yet. When the rendered configuration is byte-identical
        to the published one and a lock is already present, the lock already
        answers that declaration: re-resolving would make generation depend on
        the network and on upstream release timing, so the same sources would
        stop producing the same bytes.
        """
        if project.artifacts.lock.content is None:
            return True
        return project.config.before.content != project.config.replacement_content

    def _stage_project(
        self,
        project: m.Infra.MiseToolchainProjectState,
        *,
        stage_root: Path,
        launcher: Path,
        receipt_states: tuple[m.Cli.AtomicFileState, ...],
        environment: dict[str, str],
    ) -> p.Result[bool]:
        """Build and validate one project without reading mutable source bytes."""
        try:
            (stage_root / "bin").mkdir(parents=True, exist_ok=False)
        except OSError as exc:
            return r[bool].fail_op(
                f"create Mise stage for {project.layout.selector}", exc
            )
        config_write = process.write_new(
            stage_root / files.CONFIG_SPEC[0],
            project.config.replacement_content,
            project.config.replacement_mode,
        )
        if config_write.failure:
            return config_write
        for source, (name, mode) in zip(
            receipt_states, files.ARTIFACT_SPECS[:2], strict=True
        ):
            if source.content is None:
                return r[bool].fail(f"validated Mise receipt is absent: {name}")
            copied = process.write_new(stage_root / name, source.content, mode)
            if copied.failure:
                return copied
        lock_before = project.artifacts.lock
        if lock_before.content is not None:
            copied_lock = process.write_new(
                stage_root / "mise.lock", lock_before.content, 0o644
            )
            if copied_lock.failure:
                return copied_lock
        if self._lock_resolution_is_required(project):
            project_environment = dict(environment)
            project_environment.update({
                "MISE_CEILING_PATHS": str(stage_root.parent),
                "MISE_TRUSTED_CONFIG_PATHS": str(stage_root),
            })
            locked = process.run(
                (
                    str(launcher),
                    "-C",
                    str(stage_root),
                    "lock",
                    "--bump",
                    "--platform",
                    ",".join(config.Infra.codegen.toolchain.mise_lock_platforms),
                ),
                cwd=stage_root,
                env=project_environment,
                operation=f"Mise lock generation for {project.layout.selector}",
            )
            if locked.failure:
                return r[bool].from_failure(locked)
        hydrated = self._owner.hydrate_lock_checksums_at(stage_root)
        if hydrated.failure:
            return r[bool].fail(
                hydrated.error
                or f"Mise checksum hydration failed for {project.layout.selector}"
            )
        normalized = candidates.normalize_lock_mode(stage_root / "mise.lock")
        if normalized.failure:
            return normalized
        validated = self._owner.validate_artifacts(
            stage_root, config_sources=project.config.sources
        )
        if validated.failure:
            return r[bool].fail(
                validated.error
                or f"Mise artifact validation failed for {project.layout.selector}"
            )
        return r[bool].ok(True)


__all__: list[str] = ["FlextInfraMiseStaging"]
