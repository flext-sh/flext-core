"""Typed crash journal and recovery-backup publication for Mise artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from flext_core import r
from flext_infra import c, m, u
from flext_infra.codegen import _mise_artifacts_files as files
from flext_infra.codegen import _mise_artifacts_process as process
from flext_infra.codegen import _mise_artifacts_state as state

if TYPE_CHECKING:
    from flext_infra import p


def begin(
    plan: m.Infra.MiseToolchainWorkspacePlan,
) -> p.Result[m.Infra.MiseToolchainJournal]:
    """Build the durable staging authority before any disposable root exists."""
    sources: list[m.Infra.MiseToolchainJournalSource] = []
    for project in plan.projects:
        for source in project.config.sources:
            encoded = _journal_source(plan, source)
            if encoded.failure:
                return r[m.Infra.MiseToolchainJournal].from_failure(encoded)
            sources.append(encoded.value)
    try:
        return r[m.Infra.MiseToolchainJournal].ok(
            m.Infra.MiseToolchainJournal(
                version=3,
                state="staging",
                projects=tuple(project.layout.selector for project in plan.projects),
                sources=tuple(sources),
                entries=(),
            )
        )
    except c.ValidationError as exc:
        return r[m.Infra.MiseToolchainJournal].fail_op(
            "validate staging Mise journal", exc
        )


def prepare(
    plan: m.Infra.MiseToolchainWorkspacePlan,
    publications: tuple[m.Infra.MiseToolchainPublication, ...],
) -> p.Result[m.Infra.MiseToolchainJournal]:
    """Write every original backup and build the complete prepared journal."""
    staging = begin(plan)
    if staging.failure:
        return r[m.Infra.MiseToolchainJournal].from_failure(staging)
    entries: list[m.Infra.MiseToolchainJournalEntry] = []
    recovery_roots: set[Path] = set()
    for index, publication in enumerate(publications):
        entry = _journal_entry(
            plan, publication, index=index, recovery_roots=recovery_roots
        )
        if entry.failure:
            return r[m.Infra.MiseToolchainJournal].from_failure(entry)
        entries.append(entry.value)
    try:
        return r[m.Infra.MiseToolchainJournal].ok(
            m.Infra.MiseToolchainJournal(
                version=3,
                state="prepared",
                projects=staging.value.projects,
                sources=staging.value.sources,
                entries=tuple(entries),
            )
        )
    except c.ValidationError as exc:
        return r[m.Infra.MiseToolchainJournal].fail_op(
            "validate prepared Mise journal", exc
        )


def commit(
    journal: m.Infra.MiseToolchainJournal,
) -> p.Result[m.Infra.MiseToolchainJournal]:
    """Validate the only legal prepared-to-committed journal transition."""
    if journal.state != "prepared":
        return r[m.Infra.MiseToolchainJournal].fail(
            "only a prepared Mise journal can be committed"
        )
    try:
        return r[m.Infra.MiseToolchainJournal].ok(
            m.Infra.MiseToolchainJournal(
                version=3,
                state="committed",
                projects=journal.projects,
                sources=journal.sources,
                entries=journal.entries,
            )
        )
    except c.ValidationError as exc:
        return r[m.Infra.MiseToolchainJournal].fail_op(
            "validate committed Mise journal", exc
        )


def write(
    layout: m.Infra.MiseToolchainWorkspaceLayout,
    journal: m.Infra.MiseToolchainJournal,
    *,
    expected: m.Cli.AtomicFileState,
) -> p.Result[m.Cli.AtomicFileState]:
    """Create or transition the common journal with exact bytes and mode."""
    content = journal.model_dump_json(indent=2).encode(c.Cli.ENCODING_DEFAULT)
    journal_path = layout.state_root / files.JOURNAL_NAME
    if expected.path != journal_path:
        return r[m.Cli.AtomicFileState].fail(
            "Mise journal expected state belongs to another path"
        )
    written = u.Cli.atomic_write_binary_file_guarded(
        journal_path,
        content,
        expected_bytes=expected.content,
        expected_mode=expected.mode,
        permission_mode=files.JOURNAL_MODE,
    )
    if written.failure:
        return r[m.Cli.AtomicFileState].fail(
            written.error or "cannot publish Mise transaction journal"
        )
    observed = state.journal_state(layout)
    if observed.failure:
        return r[m.Cli.AtomicFileState].from_failure(observed)
    if observed.value.content != content or observed.value.mode != files.JOURNAL_MODE:
        return r[m.Cli.AtomicFileState].fail(
            "published Mise journal differs from staged bytes or mode"
        )
    return observed


def read(
    layout: m.Infra.MiseToolchainWorkspaceLayout,
) -> p.Result[tuple[m.Infra.MiseToolchainJournal, m.Cli.AtomicFileState]]:
    """Parse the exact regular common journal through schema v3 only."""
    return read_scope(layout.scope_root)


def read_scope(
    scope_root: Path,
) -> p.Result[tuple[m.Infra.MiseToolchainJournal, m.Cli.AtomicFileState]]:
    """Parse the common journal without consulting current workspace topology."""
    snapshot = state.journal_state_for_scope(scope_root)
    result_type = r[tuple[m.Infra.MiseToolchainJournal, m.Cli.AtomicFileState]]
    if snapshot.failure:
        return result_type.from_failure(snapshot)
    if snapshot.value.content is None:
        return result_type.fail("Mise transaction journal is absent")
    if snapshot.value.mode != files.JOURNAL_MODE:
        return result_type.fail("Mise transaction journal mode is not 0600")
    try:
        journal = m.Infra.MiseToolchainJournal.model_validate_json(
            snapshot.value.content
        )
    except c.ValidationError as exc:
        return result_type.fail_op("validate Mise transaction journal", exc)
    return result_type.ok((journal, snapshot.value))


def cleanup(
    layout: m.Infra.MiseToolchainWorkspaceLayout, journal_state: m.Cli.AtomicFileState
) -> p.Result[bool]:
    """Retain recovery authority until every disposable root is removed."""
    roots = state.remove_transaction_roots(layout)
    if roots.failure:
        return roots
    removed = files.delete_state(journal_state)
    if removed.failure:
        return r[bool].fail(removed.error or "cannot remove Mise journal")
    return r[bool].ok(True)


def _journal_source(
    plan: m.Infra.MiseToolchainWorkspacePlan, source: m.Cli.AtomicFileState
) -> p.Result[m.Infra.MiseToolchainJournalSource]:
    if source.content is None or source.mode is None:
        return r[m.Infra.MiseToolchainJournalSource].fail(
            f"journal source is absent: {source.path}"
        )
    selector = files.workspace_relative(plan.layout.scope_root, source.path)
    if selector.failure:
        return r[m.Infra.MiseToolchainJournalSource].from_failure(selector)
    return r[m.Infra.MiseToolchainJournalSource].ok(
        m.Infra.MiseToolchainJournalSource(
            path=selector.value, sha256=files.digest(source.content), mode=source.mode
        )
    )


def _journal_entry(
    plan: m.Infra.MiseToolchainWorkspacePlan,
    publication: m.Infra.MiseToolchainPublication,
    *,
    index: int,
    recovery_roots: set[Path],
) -> p.Result[m.Infra.MiseToolchainJournalEntry]:
    before = publication.before
    replacement = publication.replacement
    if replacement.content is None or replacement.mode is None:
        return r[m.Infra.MiseToolchainJournalEntry].fail(
            f"Mise staged replacement is absent: {replacement.path}"
        )
    selector = files.workspace_relative(plan.layout.scope_root, before.path)
    if selector.failure:
        return r[m.Infra.MiseToolchainJournalEntry].from_failure(selector)
    backup_selector: str | None = None
    original_sha: str | None = None
    if before.content is not None and before.mode is not None:
        project = next(
            (
                item
                for item in plan.projects
                if before.path
                in {
                    item.config.before.path,
                    item.artifacts.unix_launcher.path,
                    item.artifacts.windows_launcher.path,
                    item.artifacts.lock.path,
                }
            ),
            None,
        )
        if project is None:
            return r[m.Infra.MiseToolchainJournalEntry].fail(
                f"Mise artifact has no project owner: {before.path}"
            )
        recovery_root = project.layout.transaction_root / "recovery"
        if recovery_root not in recovery_roots:
            try:
                recovery_root.mkdir(exist_ok=False)
            except OSError as exc:
                return r[m.Infra.MiseToolchainJournalEntry].fail_op(
                    "create Mise recovery directory", exc
                )
            recovery_roots.add(recovery_root)
        backup = recovery_root / f"{index:04d}.original"
        written = process.write_new(backup, before.content, files.JOURNAL_MODE)
        if written.failure:
            return r[m.Infra.MiseToolchainJournalEntry].fail(
                written.error or f"cannot back up Mise artifact: {before.path}"
            )
        relative_backup = files.workspace_relative(plan.layout.scope_root, backup)
        if relative_backup.failure:
            return r[m.Infra.MiseToolchainJournalEntry].from_failure(relative_backup)
        backup_selector = relative_backup.value
        original_sha = files.digest(before.content)
    return r[m.Infra.MiseToolchainJournalEntry].ok(
        m.Infra.MiseToolchainJournalEntry(
            path=selector.value,
            original_exists=before.content is not None,
            original_backup=backup_selector,
            original_sha256=original_sha,
            original_mode=before.mode,
            replacement_sha256=files.digest(replacement.content),
            replacement_mode=replacement.mode,
        )
    )


__all__: list[str] = [
    "begin",
    "cleanup",
    "commit",
    "prepare",
    "read",
    "read_scope",
    "write",
]
