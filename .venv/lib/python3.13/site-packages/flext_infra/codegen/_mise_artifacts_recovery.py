"""Crash recovery for prepared or committed Mise artifact journals."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core import r
from flext_infra import m
from flext_infra.codegen import _mise_artifacts_files as files
from flext_infra.codegen import _mise_artifacts_journal as journal_io
from flext_infra.codegen import _mise_artifacts_process as process
from flext_infra.codegen import _mise_artifacts_state as state
from flext_infra.codegen import _mise_artifacts_verification as verify

if TYPE_CHECKING:
    from flext_infra import p


class FlextInfraMiseRecovery:
    """Restore only exact states attributable to one durable journal."""

    def execute(
        self,
        layout: m.Infra.MiseToolchainWorkspaceLayout,
        journal: m.Infra.MiseToolchainJournal,
        journal_state: m.Cli.AtomicFileState,
    ) -> p.Result[bool]:
        """Recover one already parsed journal against its persisted topology."""
        topology = verify.journal_topology(layout, journal)
        if topology.failure:
            return topology
        roots = state.validate_transaction_roots(layout)
        if roots.failure:
            return roots
        if journal.state == "staging":
            return journal_io.cleanup(layout, journal_state)
        classified = self._classify(layout, journal)
        if classified.failure:
            return r[bool].from_failure(classified)
        if journal.state == "prepared":
            candidates = self._prepare_restore_candidates(layout, classified.value)
            if candidates.failure:
                return r[bool].from_failure(candidates)
            restored = self._restore(classified.value, candidates.value)
            if restored.failure:
                return restored
            exact = self._verify_originals(layout, journal)
            if exact.failure:
                return exact
        return journal_io.cleanup(layout, journal_state)

    def _classify(
        self,
        layout: m.Infra.MiseToolchainWorkspaceLayout,
        journal: m.Infra.MiseToolchainJournal,
    ) -> p.Result[tuple[m.Infra.MiseToolchainRecoveryAction, ...]]:
        """Classify every live target before preparing any recovery effect."""
        result_type = r[tuple[m.Infra.MiseToolchainRecoveryAction, ...]]
        actions: list[m.Infra.MiseToolchainRecoveryAction] = []
        for entry in journal.entries:
            target = files.resolve_relative(
                layout.scope_root, entry.path, purpose="Mise artifact"
            )
            if target.failure:
                return result_type.from_failure(target)
            current = files.read_state(target.value, required=False)
            if current.failure:
                return result_type.from_failure(current)
            current_identity = self._identity(current.value)
            replacement = (entry.replacement_sha256, entry.replacement_mode)
            original = (
                (entry.original_sha256, entry.original_mode)
                if entry.original_exists
                else (None, None)
            )
            if journal.state == "committed":
                if current_identity != replacement:
                    return result_type.fail(
                        f"committed Mise artifact changed: {entry.path}"
                    )
                operation = "noop"
            elif current_identity == original:
                operation = "noop"
            elif current_identity != replacement:
                return result_type.fail(
                    f"Mise artifact has foreign state during recovery: {entry.path}"
                )
            elif entry.original_exists:
                operation = "restore"
            else:
                operation = "delete"
            actions.append(
                m.Infra.MiseToolchainRecoveryAction(
                    entry=entry, current=current.value, operation=operation
                )
            )
        return result_type.ok(tuple(actions))

    def _prepare_restore_candidates(
        self,
        layout: m.Infra.MiseToolchainWorkspaceLayout,
        actions: tuple[m.Infra.MiseToolchainRecoveryAction, ...],
    ) -> p.Result[tuple[m.Infra.MiseToolchainPublication | None, ...]]:
        """Prepare every deterministic restore candidate before live effects."""
        result_type = r[tuple[m.Infra.MiseToolchainPublication | None, ...]]
        candidates: list[m.Infra.MiseToolchainPublication | None] = []
        for action in actions:
            if action.operation != "restore":
                candidates.append(None)
                continue
            prepared = self._restore_publication(layout, action)
            if prepared.failure:
                return result_type.from_failure(prepared)
            candidates.append(prepared.value)
        return result_type.ok(tuple(candidates))

    @staticmethod
    def _restore_publication(
        layout: m.Infra.MiseToolchainWorkspaceLayout,
        action: m.Infra.MiseToolchainRecoveryAction,
    ) -> p.Result[m.Infra.MiseToolchainPublication]:
        entry = action.entry
        if (
            entry.original_backup is None
            or entry.original_sha256 is None
            or entry.original_mode is None
        ):
            return r[m.Infra.MiseToolchainPublication].fail(
                f"Mise recovery tuple is incomplete: {entry.path}"
            )
        backup_path = files.resolve_relative(
            layout.scope_root, entry.original_backup, purpose="Mise recovery backup"
        )
        if backup_path.failure:
            return r[m.Infra.MiseToolchainPublication].from_failure(backup_path)
        backup = files.read_state(backup_path.value, required=True)
        if backup.failure or backup.value.content is None:
            return r[m.Infra.MiseToolchainPublication].fail(
                backup.error or f"Mise recovery backup is absent: {entry.path}"
            )
        if (
            backup.value.mode != files.JOURNAL_MODE
            or files.digest(backup.value.content) != entry.original_sha256
        ):
            return r[m.Infra.MiseToolchainPublication].fail(
                f"Mise recovery backup identity differs: {entry.path}"
            )
        candidate_path = backup_path.value.with_suffix(".restore")
        candidate = files.read_state(candidate_path, required=False)
        if candidate.failure:
            return r[m.Infra.MiseToolchainPublication].from_failure(candidate)
        if candidate.value.content is None:
            created = process.write_new(
                candidate_path, backup.value.content, entry.original_mode
            )
            if created.failure:
                return r[m.Infra.MiseToolchainPublication].fail(
                    created.error or f"cannot prepare Mise restore: {entry.path}"
                )
            candidate = files.read_state(candidate_path, required=True)
            if candidate.failure:
                return r[m.Infra.MiseToolchainPublication].from_failure(candidate)
        if (
            candidate.value.content != backup.value.content
            or candidate.value.mode != entry.original_mode
        ):
            return r[m.Infra.MiseToolchainPublication].fail(
                f"Mise restore candidate identity differs: {entry.path}"
            )
        return r[m.Infra.MiseToolchainPublication].ok(
            m.Infra.MiseToolchainPublication(
                before=action.current, replacement=candidate.value
            )
        )

    @staticmethod
    def _restore(
        actions: tuple[m.Infra.MiseToolchainRecoveryAction, ...],
        candidates: tuple[m.Infra.MiseToolchainPublication | None, ...],
    ) -> p.Result[bool]:
        paired = tuple(zip(actions, candidates, strict=True))
        for action, candidate in reversed(paired):
            if action.operation == "restore":
                if candidate is None:
                    return r[bool].fail(
                        f"Mise restore candidate is absent: {action.entry.path}"
                    )
                restored = files.write_publication(candidate)
                if restored.failure:
                    return r[bool].fail(
                        restored.error or f"Mise restore failed: {action.entry.path}"
                    )
            elif action.operation == "delete":
                removed = files.delete_state(action.current)
                if removed.failure:
                    return r[bool].fail(
                        removed.error
                        or f"Mise rollback delete failed: {action.entry.path}"
                    )
        return r[bool].ok(True)

    @staticmethod
    def _verify_originals(
        layout: m.Infra.MiseToolchainWorkspaceLayout,
        journal: m.Infra.MiseToolchainJournal,
    ) -> p.Result[bool]:
        for entry in journal.entries:
            target = files.resolve_relative(
                layout.scope_root, entry.path, purpose="Mise artifact"
            )
            if target.failure:
                return r[bool].from_failure(target)
            current = files.read_state(target.value, required=False)
            expected = (
                (entry.original_sha256, entry.original_mode)
                if entry.original_exists
                else (None, None)
            )
            if (
                current.failure
                or FlextInfraMiseRecovery._identity(current.value) != expected
            ):
                return r[bool].fail(
                    current.error or f"Mise artifact was not restored: {entry.path}"
                )
        return r[bool].ok(True)

    @staticmethod
    def _identity(state: m.Cli.AtomicFileState) -> tuple[str | None, int | None]:
        return (
            None if state.content is None else files.digest(state.content),
            state.mode,
        )


__all__: list[str] = ["FlextInfraMiseRecovery"]
