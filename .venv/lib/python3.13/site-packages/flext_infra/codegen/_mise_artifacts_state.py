"""Persistent coordination-state lifecycle for Mise transactions."""

from __future__ import annotations

import shutil
import os
import stat
from pathlib import Path
from typing import TYPE_CHECKING

from flext_core import r
from flext_infra import m
from flext_infra.codegen import _mise_artifacts_files as files

if TYPE_CHECKING:
    from flext_infra import p


def prepare_state_roots(layout: m.Infra.MiseToolchainWorkspaceLayout) -> p.Result[bool]:
    """Create every configured persistent state root before transaction effects."""
    for project in layout.projects:
        prepared = _prepare_state_root(project.root)
        if prepared.failure:
            return prepared
        try:
            staging_device = project.transaction_root.parent.stat().st_dev
            # The staged artifacts are promoted into these directories, so a
            # repository that has never carried one (a governed member whose
            # `bin/` this transaction is about to create) owns it here. Without
            # it the device probe below reports ENOENT instead of comparing
            # filesystems.
            for artifact in (
                project.artifacts.unix_launcher,
                project.artifacts.windows_launcher,
            ):
                artifact.parent.mkdir(parents=True, exist_ok=True)
            destination_devices = {
                artifact.parent.stat().st_dev
                for artifact in (
                    project.artifacts.config,
                    project.artifacts.unix_launcher,
                    project.artifacts.windows_launcher,
                    project.artifacts.lock,
                )
            }
            if destination_devices != {staging_device}:
                return r[bool].fail(
                    f"Mise state is not on destination filesystem: {project.selector}"
                )
        except OSError as exc:
            return r[bool].fail_op("inspect Mise state filesystem", exc)
    return r[bool].ok(True)


def prepare_common_state_root(scope_root: Path) -> p.Result[bool]:
    """Create only the umbrella coordination root before acquiring its lock."""
    return _prepare_state_root(scope_root)


def validate_lock_path(scope_root: Path, *, require_existing: bool) -> p.Result[Path]:
    """Authenticate the persistent lock path before FileLock opens it."""
    lock_path = scope_root / files.STATE_DIRECTORY / files.LOCK_NAME
    try:
        parent = lock_path.parent.lstat()
    except OSError as exc:
        return r[Path].fail_op("inspect Mise lock parent", exc)
    if not stat.S_ISDIR(parent.st_mode) or _is_reparse(parent):
        return r[Path].fail(f"Mise lock parent is not physical: {lock_path.parent}")
    if not lock_path.exists() and not lock_path.is_symlink():
        if require_existing:
            return r[Path].fail(f"Mise transaction lock is absent: {lock_path}")
        return r[Path].ok(lock_path)
    try:
        state = lock_path.lstat()
    except OSError as exc:
        return r[Path].fail_op("inspect Mise transaction lock", exc)
    if not stat.S_ISREG(state.st_mode) or state.st_nlink != 1 or _is_reparse(state):
        return r[Path].fail(f"Mise transaction lock is not physical: {lock_path}")
    return r[Path].ok(lock_path)


def journal_state(
    layout: m.Infra.MiseToolchainWorkspaceLayout,
) -> p.Result[m.Cli.AtomicFileState]:
    """Read the common journal without creating its parent in check mode."""
    return journal_state_for_scope(layout.scope_root)


def journal_state_for_scope(scope_root: Path) -> p.Result[m.Cli.AtomicFileState]:
    """Read the common journal before mutable workspace topology is loaded."""
    state_root = scope_root / files.STATE_DIRECTORY
    journal = state_root / files.JOURNAL_NAME
    if not state_root.exists() and not state_root.is_symlink():
        return r[m.Cli.AtomicFileState].ok(m.Cli.AtomicFileState(path=journal))
    return files.read_state(journal, required=False)


def transaction_residue(
    layout: m.Infra.MiseToolchainWorkspaceLayout,
) -> tuple[Path, ...]:
    """Return every existing or aliased per-project transaction root."""
    return tuple(
        project.transaction_root
        for project in layout.projects
        if project.transaction_root.exists() or project.transaction_root.is_symlink()
    )


def create_transaction_roots(
    layout: m.Infra.MiseToolchainWorkspaceLayout,
) -> p.Result[bool]:
    """Create one fresh private transaction root on each destination filesystem."""
    residue = transaction_residue(layout)
    if residue:
        return r[bool].fail(f"Mise transaction staging already exists: {residue[0]}")
    created: list[tuple[Path, tuple[int, int]]] = []
    try:
        for project in layout.projects:
            project.transaction_root.mkdir(mode=0o700, exist_ok=False)
            state = project.transaction_root.lstat()
            created.append((project.transaction_root, (state.st_dev, state.st_ino)))
    except OSError as exc:
        cleanup = _remove_exact(tuple(created))
        if cleanup.failure:
            return r[bool].fail(
                f"create Mise transaction roots failed ({exc}); "
                f"cleanup failed ({cleanup.error})"
            )
        return r[bool].fail_op("create Mise transaction roots", exc)
    return r[bool].ok(True)


def remove_transaction_roots(
    layout: m.Infra.MiseToolchainWorkspaceLayout,
) -> p.Result[bool]:
    """Remove only preflighted literal transaction roots owned by this plan."""
    targets: list[tuple[Path, tuple[int, int]]] = []
    for project in layout.projects:
        valid = _validate_transaction_root(project.transaction_root)
        if valid.failure:
            return r[bool].from_failure(valid)
        if valid.value is not None:
            targets.append((project.transaction_root, valid.value))
    return _remove_exact(tuple(targets))


def validate_transaction_roots(
    layout: m.Infra.MiseToolchainWorkspaceLayout,
) -> p.Result[bool]:
    """Authenticate every existing staging and recovery root."""
    for project in layout.projects:
        transaction = _validate_transaction_root(project.transaction_root)
        if transaction.failure:
            return r[bool].from_failure(transaction)
        if transaction.value is None:
            continue
        recovery_root = project.transaction_root / "recovery"
        if not recovery_root.exists() and not recovery_root.is_symlink():
            continue
        try:
            recovery_state = recovery_root.lstat()
        except OSError as exc:
            return r[bool].fail_op("inspect Mise recovery root", exc)
        if not stat.S_ISDIR(recovery_state.st_mode) or _is_reparse(recovery_state):
            return r[bool].fail(f"Mise recovery root is not physical: {recovery_root}")
    return r[bool].ok(True)


def _prepare_state_root(project_root: Path) -> p.Result[bool]:
    cursor = project_root.absolute()
    try:
        for part in files.STATE_DIRECTORY.parts:
            cursor /= part
            if not cursor.exists() and not cursor.is_symlink():
                cursor.mkdir(mode=0o700, exist_ok=False)
            state = cursor.lstat()
            if not stat.S_ISDIR(state.st_mode) or _is_reparse(state):
                return r[bool].fail(f"Mise state path is not physical: {cursor}")
    except OSError as exc:
        return r[bool].fail_op("prepare persistent Mise state", exc)
    return r[bool].ok(True)


def _validate_transaction_root(target: Path) -> p.Result[tuple[int, int] | None]:
    if not target.exists() and not target.is_symlink():
        return r[tuple[int, int] | None].ok(None)
    if target.name != files.TRANSACTION_DIR_NAME or target.is_symlink():
        return r[tuple[int, int] | None].fail(
            f"refusing invalid Mise transaction target: {target}"
        )
    try:
        state = target.lstat()
    except OSError as exc:
        return r[tuple[int, int] | None].fail_op("inspect Mise transaction target", exc)
    if not stat.S_ISDIR(state.st_mode) or _is_reparse(state):
        return r[tuple[int, int] | None].fail(
            f"Mise transaction target is not physical: {target}"
        )
    return r[tuple[int, int] | None].ok((state.st_dev, state.st_ino))


def _remove_exact(targets: tuple[tuple[Path, tuple[int, int]], ...]) -> p.Result[bool]:
    try:
        for target, expected_identity in targets:
            state = target.lstat()
            if (
                (state.st_dev, state.st_ino) != expected_identity
                or not stat.S_ISDIR(state.st_mode)
                or _is_reparse(state)
            ):
                return r[bool].fail(
                    f"Mise transaction target changed before cleanup: {target}"
                )
            shutil.rmtree(target)
    except OSError as exc:
        return r[bool].fail_op("remove exact Mise transaction staging", exc)
    return r[bool].ok(True)


def _is_reparse(state: os.stat_result) -> bool:
    attributes = getattr(state, "st_file_attributes", 0)
    marker = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(attributes & marker)


__all__: list[str] = [
    "create_transaction_roots",
    "journal_state",
    "journal_state_for_scope",
    "prepare_common_state_root",
    "prepare_state_roots",
    "remove_transaction_roots",
    "transaction_residue",
    "validate_lock_path",
    "validate_transaction_roots",
]
