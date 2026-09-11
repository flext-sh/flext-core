"""Thin transaction coordinator for newest Mise artifact publication."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from filelock import FileLock, Timeout
from flext_core import r
from flext_infra import m, settings
from flext_infra.codegen import _mise_artifacts_journal as journal_io
from flext_infra.codegen import _mise_artifacts_publication as publication
from flext_infra.codegen import _mise_artifacts_state as state
from flext_infra.codegen import _mise_artifacts_verification as verify
from flext_infra.codegen._mise_artifacts_recovery import FlextInfraMiseRecovery
from flext_infra.codegen._mise_artifacts_staging import FlextInfraMiseStaging
from flext_infra.codegen._mise_artifacts_workspace import FlextInfraMiseWorkspacePlanner

if TYPE_CHECKING:
    from flext_infra import p


class FlextInfraCodegenMiseArtifactTransaction:
    """Coordinate one workspace-wide transaction through focused owners."""

    def __init__(self, owner: p.Infra.MiseArtifactsOwner) -> None:
        self._owner = owner
        self._planner = FlextInfraMiseWorkspacePlanner(owner)
        self._recovery = FlextInfraMiseRecovery()
        self._staging = FlextInfraMiseStaging(owner)

    def validate(
        self, config_plans: tuple[m.Infra.CodegenFilePlan, ...] = ()
    ) -> p.Result[bool]:
        """Validate one coherent live snapshot under the existing lock."""
        return self.run_locked(
            prepare=False,
            operation=lambda scope_root: self.validate_locked(scope_root, config_plans),
        )

    def validate_locked(
        self, scope_root: Path, config_plans: tuple[m.Infra.CodegenFilePlan, ...] = ()
    ) -> p.Result[bool]:
        """Validate a plan while the generation owner holds the workspace lock."""
        layout_result = (
            self._planner.layout_for_config_plans(scope_root, config_plans)
            if config_plans
            else self._planner.layout(scope_root)
        )
        if layout_result.failure:
            return r[bool].from_failure(layout_result)
        selected = self._planner.select_layout(layout_result.value, config_plans)
        if selected.failure:
            return r[bool].from_failure(selected)
        layout = selected.value
        journal = state.journal_state(layout)
        if journal.failure:
            return r[bool].from_failure(journal)
        if journal.value.content is not None:
            return r[bool].fail("pending Mise transaction requires apply-mode recovery")
        residue = state.transaction_residue(layout)
        if residue:
            return r[bool].fail(
                f"unpublished Mise staging requires apply-mode cleanup: {residue[0]}"
            )
        plan = self._planner.snapshot(layout, config_plans)
        if plan.failure:
            return r[bool].from_failure(plan)
        return verify.live(self._owner, plan.value)

    def execute(
        self, config_plans: tuple[m.Infra.CodegenFilePlan, ...]
    ) -> p.Result[bool]:
        """Recover if necessary, then publish one locked atomic workspace set."""
        published = self.run_locked(
            prepare=True,
            operation=lambda scope_root: self.publish_locked(scope_root, config_plans),
        )
        if published.failure:
            return r[bool].from_failure(published)
        return r[bool].ok(True)

    def run_locked[T](
        self, *, prepare: bool, operation: Callable[[Path], p.Result[T]]
    ) -> p.Result[T]:
        """Run one generation operation under the stable workspace lock."""
        scope_root = self._planner.scope_root()
        if scope_root.failure:
            return r[T].from_failure(scope_root)
        # The coordination root is transaction state, never a managed write, and
        # it is gitignored: a fresh clone has none. Creating it before the lock
        # in every mode is what lets `gen check` serialize against a concurrent
        # publisher on a checkout that has never published.
        prepared = state.prepare_common_state_root(scope_root.value)
        if prepared.failure:
            return r[T].from_failure(prepared)
        lock_path = state.validate_lock_path(scope_root.value, require_existing=False)
        if lock_path.failure:
            return r[T].from_failure(lock_path)
        try:
            with FileLock(str(lock_path.value), timeout=0):
                return self._run_authenticated(
                    scope_root.value, prepare=prepare, operation=operation
                )
        except Timeout:
            return r[T].fail(
                "another Mise artifact transaction owns the workspace: "
                f"{lock_path.value}"
            )
        except OSError as exc:
            return r[T].fail_op("execute Mise toolchain transaction", exc)

    def _run_authenticated[T](
        self,
        scope_root: Path,
        *,
        prepare: bool,
        operation: Callable[[Path], p.Result[T]],
    ) -> p.Result[T]:
        """Authenticate lock and layout before entering caller-owned work."""
        authenticated = state.validate_lock_path(scope_root, require_existing=True)
        if authenticated.failure:
            return r[T].from_failure(authenticated)
        if prepare:
            reconciled = self._reconcile(scope_root)
            if reconciled.failure:
                return r[T].from_failure(reconciled)
        return operation(scope_root)

    def publish_locked(
        self, scope_root: Path, config_plans: tuple[m.Infra.CodegenFilePlan, ...]
    ) -> p.Result[tuple[Path, ...]]:
        """Publish one planned bundle while its caller holds the stable lock."""
        result_type = r[tuple[Path, ...]]
        layout_result = self._planner.layout_for_config_plans(scope_root, config_plans)
        if layout_result.failure:
            return result_type.from_failure(layout_result)
        layout = layout_result.value
        plan = self._planner.snapshot(layout, config_plans)
        if plan.failure:
            return result_type.from_failure(plan)
        credential_command = settings.Infra.mise_github_credential_command
        if credential_command is None or not credential_command.strip():
            return result_type.fail(
                "MISE_GITHUB_CREDENTIAL_COMMAND is required for Mise lock publication"
            )
        journal_before = state.journal_state(layout)
        if journal_before.failure:
            return result_type.from_failure(journal_before)
        if journal_before.value.content is not None:
            return result_type.fail("Mise journal appeared after locked recovery")
        roots = state.prepare_state_roots(layout)
        if roots.failure:
            return result_type.from_failure(roots)
        residue = state.transaction_residue(layout)
        if residue:
            return result_type.fail(
                f"Mise transaction residue has no journal authority: {residue[0]}"
            )
        staging_journal = journal_io.begin(plan.value)
        if staging_journal.failure:
            return result_type.from_failure(staging_journal)
        staging_topology = verify.journal_topology(layout, staging_journal.value)
        if staging_topology.failure:
            return result_type.from_failure(staging_topology)
        staging_state = journal_io.write(
            layout, staging_journal.value, expected=journal_before.value
        )
        if staging_state.failure:
            return result_type.from_failure(staging_state)
        created = state.create_transaction_roots(layout)
        if created.failure:
            return result_type.from_failure(
                self._recover_failure(
                    scope_root, created.error or "cannot create Mise transaction roots"
                )
            )
        staged = self._staging.stage(
            plan.value, credential_command=credential_command.strip()
        )
        if staged.failure:
            return result_type.from_failure(
                self._recover_failure(
                    scope_root, staged.error or "cannot stage Mise artifacts"
                )
            )
        sources = verify.sources(plan.value)
        if sources.failure:
            return result_type.from_failure(
                self._recover_failure(
                    scope_root, sources.error or "Mise sources changed during staging"
                )
            )
        destinations = verify.destinations(plan.value)
        if destinations.failure:
            return result_type.from_failure(
                self._recover_failure(
                    scope_root,
                    destinations.error or "Mise destinations changed during staging",
                )
            )
        prepared_journal = journal_io.prepare(plan.value, staged.value)
        if prepared_journal.failure:
            return result_type.from_failure(
                self._recover_failure(
                    scope_root,
                    prepared_journal.error or "cannot prepare Mise recovery journal",
                )
            )
        prepared_topology = verify.journal_topology(layout, prepared_journal.value)
        if prepared_topology.failure:
            return result_type.from_failure(
                self._recover_failure(
                    scope_root,
                    prepared_topology.error
                    or "Mise prepared journal topology is invalid",
                )
            )
        source_barrier = verify.sources(plan.value, prepared_journal.value)
        if source_barrier.failure:
            return result_type.from_failure(
                self._recover_failure(
                    scope_root,
                    source_barrier.error or "Mise sources changed before publication",
                )
            )
        destination_barrier = verify.destinations(plan.value)
        if destination_barrier.failure:
            return result_type.from_failure(
                self._recover_failure(
                    scope_root,
                    destination_barrier.error
                    or "Mise destinations changed before publication",
                )
            )
        prepared_state = journal_io.write(
            layout, prepared_journal.value, expected=staging_state.value
        )
        if prepared_state.failure:
            return result_type.from_failure(
                self._handle_journal_write_failure(
                    scope_root,
                    prepared_state.error or "cannot publish prepared Mise journal",
                )
            )
        published = publication.publish(
            self._owner, plan.value, prepared_journal.value, staged.value
        )
        if published.failure:
            return result_type.from_failure(
                self._recover_failure(
                    scope_root, published.error or "Mise artifact publication failed"
                )
            )
        precommit = verify.live(self._owner, plan.value, staged.value)
        if precommit.failure:
            return result_type.from_failure(
                self._recover_failure(
                    scope_root, precommit.error or "Mise pre-commit validation failed"
                )
            )
        source_commit = verify.sources(plan.value, prepared_journal.value)
        if source_commit.failure:
            return result_type.from_failure(
                self._recover_failure(
                    scope_root,
                    source_commit.error or "Mise sources changed before commit",
                )
            )
        committed_journal = journal_io.commit(prepared_journal.value)
        if committed_journal.failure:
            return result_type.from_failure(
                self._recover_failure(
                    scope_root,
                    committed_journal.error or "cannot validate Mise journal commit",
                )
            )
        committed_state = journal_io.write(
            layout, committed_journal.value, expected=prepared_state.value
        )
        if committed_state.failure:
            return result_type.from_failure(
                self._recover_failure(
                    scope_root, committed_state.error or "cannot commit Mise journal"
                )
            )
        changed = tuple(
            item.before.path
            for item in staged.value
            if (
                item.before.content != item.replacement.content
                or item.before.mode != item.replacement.mode
            )
        )
        cleaned = journal_io.cleanup(layout, committed_state.value)
        if cleaned.failure:
            return result_type.from_failure(cleaned)
        return result_type.ok(changed)

    def _reconcile(self, scope_root: Path) -> p.Result[bool]:
        journal = state.journal_state_for_scope(scope_root)
        if journal.failure:
            return r[bool].from_failure(journal)
        if journal.value.content is not None:
            recovered = self._recover(scope_root)
            if recovered.failure:
                return recovered
        return r[bool].ok(True)

    def _handle_journal_write_failure(
        self, scope_root: Path, failure: str
    ) -> p.Result[bool]:
        observed = state.journal_state_for_scope(scope_root)
        if observed.failure:
            return r[bool].fail(
                f"{failure}; journal inspection failed: {observed.error}"
            )
        if observed.value.content is None:
            return r[bool].fail(f"{failure}; durable journal disappeared")
        return self._recover_failure(scope_root, failure)

    def _recover_failure(self, scope_root: Path, failure: str) -> p.Result[bool]:
        recovered = self._recover(scope_root)
        if recovered.failure:
            return r[bool].fail(f"{failure}; recovery failed: {recovered.error}")
        return r[bool].fail(failure)

    def _recover(self, scope_root: Path) -> p.Result[bool]:
        """Recover only the topology authorized by the durable common journal."""
        loaded = journal_io.read_scope(scope_root)
        if loaded.failure:
            return r[bool].from_failure(loaded)
        journal, journal_state = loaded.value
        layout = self._planner.layout_from_selectors(scope_root, journal.projects)
        if layout.failure:
            return r[bool].from_failure(layout)
        return self._recovery.execute(layout.value, journal, journal_state)


__all__: list[str] = ["FlextInfraCodegenMiseArtifactTransaction"]
