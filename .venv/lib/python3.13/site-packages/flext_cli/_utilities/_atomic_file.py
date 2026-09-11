"""Private atomic-file publication primitive for ``u.Cli`` file helpers."""

from __future__ import annotations

import errno
import os
from dataclasses import dataclass
from pathlib import Path

# Why: merge origin/0.12.0-dev (#131) into fix/flext-uatvz-atomic-files — #131
# merged a concurrent lane's unfixed tip of this branch directly into
# 0.12.0-dev with failing CI (module-qualified `file_mode.*`/`file_path.*`/
# `file_descriptor.*`/`file_state.*` calls with no corresponding module
# import — NameError at import time). This lane's explicit-import,
# gate-green implementation is kept as-is.
from flext_cli._utilities._atomic_file_cleanup import remove_failed_temporary
from flext_cli._utilities._atomic_file_descriptor import (
    ParentDescriptor,
    parent_descriptor,
    replace_entry,
)
from flext_cli._utilities._atomic_file_mode import (
    NO_MODE_PRECONDITION,
    NoModePrecondition,
    publication_mode,
    validate_guarded_mode_tuple,
    validate_mode_precondition,
)
from flext_cli._utilities._atomic_file_path import validate_atomic_path
from flext_cli._utilities._atomic_file_publish_checks import (
    require_distinct_inode,
    require_identity,
    validate_devices,
    validate_publication,
)
from flext_cli._utilities._atomic_file_state import (
    assert_destination_unchanged,
    assert_temporary_owned,
)
from flext_cli._utilities._atomic_file_state import (
    destination_state as file_destination_state,
)
from flext_cli._utilities._atomic_file_state import identity as file_identity
from flext_cli._utilities._atomic_file_state import (
    permission_state as file_permission_state,
)
from flext_cli._utilities._atomic_file_state import (
    validate_precondition as validate_state_precondition,
)
from flext_cli._utilities._atomic_file_temporary import (
    create_descriptor,
    require_mode_capability,
    temporary_path,
    write_and_sync,
)


class _NoPrecondition:
    """Marker for callers that do not supply an expected version."""


_NO_PRECONDITION = _NoPrecondition()


def write_atomic_bytes(
    path: Path,
    content: bytes,
    *,
    expected_bytes: bytes | _NoPrecondition | None = _NO_PRECONDITION,
    expected_mode: int | NoModePrecondition | None = NO_MODE_PRECONDITION,
    permission_mode: int | None = None,
) -> None:
    """Replace bytes and mode for a uniquely owned regular destination.

    ``expected_bytes`` is an exact raw-byte precondition for a caller that holds
    the same exclusive cooperative lock from planning through publication. The
    descriptor-bound replace is not compare-and-swap against actors that ignore
    that lock. The staged file is synced; parent-directory durability across
    power loss is outside this primitive's contract.
    """
    path = validate_atomic_path(path)
    guarded, expected_content = _parse_precondition(path, expected_bytes)
    if not guarded and expected_mode is not NO_MODE_PRECONDITION:
        message = "expected_mode requires an expected_bytes precondition"
        raise OSError(errno.EINVAL, message, path)
    if guarded:
        validate_guarded_mode_tuple(path, expected_content, expected_mode)
    with parent_descriptor(path, replace=True, unlink=True) as parent:
        expected = (
            file_destination_state(path, parent=parent)
            if guarded
            else file_permission_state(path, parent=parent)
        )
        validate_state_precondition(
            path, expected, expected_content, enabled=guarded, parent=parent
        )
        validate_mode_precondition(path, expected, expected_mode)
        target_mode = publication_mode(expected, permission_mode)
        require_mode_capability(path, target_mode)
        _stage_and_publish(
            parent, path, content, expected, target_mode, guarded=guarded
        )


def _parse_precondition(
    path: Path, expected_bytes: bytes | _NoPrecondition | None
) -> tuple[bool, bytes | None]:
    if expected_bytes is _NO_PRECONDITION:
        return False, None
    if isinstance(expected_bytes, _NoPrecondition):
        message = "expected_bytes sentinel must be the canonical singleton"
        raise OSError(errno.EINVAL, message, path)
    return True, expected_bytes


@dataclass
class _StagingProgress:
    """Mutable cleanup state shared across the staged-write/publish sequence.

    The two halves below can raise at any point after acquiring a resource;
    this holder lets the outer ``try`` retrieve exactly what was acquired so
    far without inflating that clause past the enforced statement budget.
    """

    descriptor: int | None = None
    staged_identity: tuple[int, int] | None = None
    replacement_completed: bool = False


def _stage_and_publish(
    parent: ParentDescriptor,
    destination: Path,
    content: bytes,
    expected: os.stat_result | None,
    target_mode: int | None,
    *,
    guarded: bool,
) -> None:
    temporary = temporary_path(parent)
    progress = _StagingProgress()
    try:
        _write_and_publish_staged(
            parent,
            destination,
            temporary,
            content,
            expected,
            target_mode,
            progress,
            guarded=guarded,
        )
    except BaseException as operation_error:
        if not progress.replacement_completed:
            remove_failed_temporary(
                parent,
                temporary,
                progress.staged_identity,
                progress.descriptor,
                operation_error,
            )
        raise


def _write_and_publish_staged(
    parent: ParentDescriptor,
    destination: Path,
    temporary: Path,
    content: bytes,
    expected: os.stat_result | None,
    target_mode: int | None,
    progress: _StagingProgress,
    *,
    guarded: bool,
) -> None:
    staged_mode, staged_identity = _write_staged(
        parent, temporary, content, target_mode, progress
    )
    staged_state = _validate_staged(
        parent, temporary, content, staged_mode, staged_identity
    )
    _publish_staged(
        parent,
        destination,
        temporary,
        content,
        expected,
        staged_mode,
        staged_identity,
        staged_state,
        progress,
        guarded=guarded,
    )


def _write_staged(
    parent: ParentDescriptor,
    temporary: Path,
    content: bytes,
    target_mode: int | None,
    progress: _StagingProgress,
) -> tuple[int, tuple[int, int]]:
    progress.descriptor = create_descriptor(parent, temporary)
    staged_identity = file_identity(os.fstat(progress.descriptor))
    progress.staged_identity = staged_identity
    assert_temporary_owned(temporary, staged_identity, parent=parent)
    staged_mode = write_and_sync(progress.descriptor, temporary, content, target_mode)
    os.close(progress.descriptor)
    progress.descriptor = None
    return staged_mode, staged_identity


def _publish_staged(
    parent: ParentDescriptor,
    destination: Path,
    temporary: Path,
    content: bytes,
    expected: os.stat_result | None,
    staged_mode: int,
    staged_identity: tuple[int, int],
    staged_state: os.stat_result,
    progress: _StagingProgress,
    *,
    guarded: bool,
) -> None:
    require_distinct_inode(destination, expected, staged_identity)
    validate_devices(destination, parent, expected, temporary, parent, staged_state)
    if guarded:
        assert_destination_unchanged(destination, expected, parent=parent)
    assert_temporary_owned(temporary, staged_identity, parent=parent)
    replace_entry(parent, temporary, parent, destination)
    progress.replacement_completed = True
    validate_publication(
        parent, destination, parent, temporary, content, staged_mode, staged_identity
    )


def _validate_staged(
    parent: ParentDescriptor,
    temporary: Path,
    content: bytes,
    mode: int,
    identity: tuple[int, int],
) -> os.stat_result:
    state = file_destination_state(temporary, parent=parent)
    if state is None:
        message = f"atomic temporary disappeared before publication: {temporary}"
        raise FileNotFoundError(errno.ENOENT, message, temporary)
    require_identity(temporary, state, identity)
    validate_state_precondition(temporary, state, content, enabled=True, parent=parent)
    validate_mode_precondition(temporary, state, mode)
    return state


__all__: list[str] = ["write_atomic_bytes"]
