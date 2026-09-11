"""Filesystem-state authentication for the atomic publication owner."""

from __future__ import annotations

import errno
import os
import stat
from pathlib import Path

from flext_cli._utilities._atomic_file_descriptor import (
    ParentDescriptor,
    assert_parent_unchanged,
    entry_stat,
)
from flext_cli._utilities._atomic_file_descriptor import (
    parent_descriptor as open_parent_descriptor,
)
from flext_cli._utilities._atomic_file_path import identity as path_identity
from flext_cli._utilities._atomic_file_path import (
    is_reparse_point,
    validate_parent_path,
)
from flext_cli._utilities._atomic_file_read import read_descriptor_bytes, state_key


def validate_parent(parent: Path) -> os.stat_result:
    """Return the physical state of one normalized immediate parent."""
    return validate_parent_path(parent)


def destination_state(
    path: Path, *, parent: ParentDescriptor | None = None
) -> os.stat_result | None:
    """Return an authorized destination snapshot without following links."""
    if parent is None:
        with open_parent_descriptor(path) as opened:
            state = destination_state(path, parent=opened)
            assert_parent_unchanged(opened)
            return state
    try:
        state = entry_stat(parent, path)
    except FileNotFoundError:
        return None
    _validate_regular_state(path, state)
    return state


def permission_state(
    path: Path, *, parent: ParentDescriptor | None = None
) -> os.stat_result | None:
    """Return mode provenance only when one legacy destination owns its inode.

    Unlike ``destination_state``, an existing non-regular or multiply-linked
    destination is not a precondition failure here: the caller has no
    explicit-bytes lock to enforce, so it inherits no permission mode from
    that name and proceeds to replace it unconditionally.
    """
    try:
        return destination_state(path, parent=parent)
    except OSError:
        return None


def validate_precondition(
    path: Path,
    state: os.stat_result | None,
    expected_bytes: bytes | None,
    *,
    enabled: bool,
    parent: ParentDescriptor | None = None,
) -> None:
    """Authenticate an explicit raw-byte version before staging."""
    if not enabled:
        return
    if expected_bytes is None:
        if state is not None:
            message = f"atomic destination exists but absence was required: {path}"
            raise FileExistsError(errno.EEXIST, message, path)
        return
    if state is None:
        message = f"atomic destination is missing: {path}"
        raise FileNotFoundError(errno.ENOENT, message, path)
    if read_authenticated_bytes(path, state, parent=parent) != expected_bytes:
        message = f"atomic destination content changed before write: {path}"
        raise OSError(errno.ESTALE, message, path)


def assert_temporary_owned(
    temporary: Path,
    expected_identity: tuple[int, int],
    *,
    parent: ParentDescriptor | None = None,
) -> None:
    """Require the staged pathname to retain its uniquely owned inode."""
    if parent is None:
        with open_parent_descriptor(temporary) as opened:
            assert_temporary_owned(temporary, expected_identity, parent=opened)
            return
    try:
        state = entry_stat(parent, temporary)
    except FileNotFoundError as exc:
        message = f"atomic temporary disappeared before publication: {temporary}"
        raise FileNotFoundError(errno.ENOENT, message, temporary) from exc
    if identity(state) != expected_identity:
        message = f"atomic temporary identity changed before publication: {temporary}"
        raise OSError(errno.ESTALE, message, temporary)
    _validate_regular_state(temporary, state)
    assert_parent_unchanged(parent)


def assert_destination_unchanged(
    path: Path,
    expected: os.stat_result | None,
    *,
    parent: ParentDescriptor | None = None,
) -> None:
    """Fail before publication when destination identity or state changed."""
    if parent is None:
        with open_parent_descriptor(path) as opened:
            assert_destination_unchanged(path, expected, parent=opened)
            return
    current = destination_state(path, parent=parent)
    if expected is None:
        if current is not None:
            message = f"atomic destination appeared during write: {path}"
            raise FileExistsError(errno.EEXIST, message, path)
    elif current is None or state_key(current) != state_key(expected):
        message = f"atomic destination changed during write: {path}"
        raise OSError(errno.ESTALE, message, path)
    assert_parent_unchanged(parent)


def identity(state: os.stat_result) -> tuple[int, int]:
    """Return the filesystem identity shared by descriptor and pathname stats."""
    return path_identity(state)


def read_authenticated_bytes(
    path: Path, expected: os.stat_result, *, parent: ParentDescriptor | None = None
) -> bytes:
    """Read exact bytes and prove the descriptor remains bound to its pathname."""
    if parent is None:
        with open_parent_descriptor(path) as opened:
            return read_authenticated_bytes(path, expected, parent=opened)
    content = read_descriptor_bytes(parent, path, expected)
    assert_destination_unchanged(path, expected, parent=parent)
    return content


def _validate_regular_state(path: Path, state: os.stat_result) -> None:
    if not stat.S_ISREG(state.st_mode) or is_reparse_point(state):
        message = f"atomic destination is not a regular file: {path}"
        raise OSError(errno.EINVAL, message, path)
    if state.st_nlink != 1:
        message = f"atomic destination has {state.st_nlink} hard links: {path}"
        raise OSError(errno.EMLINK, message, path)


__all__: list[str] = [
    "assert_destination_unchanged",
    "assert_temporary_owned",
    "destination_state",
    "identity",
    "permission_state",
    "read_authenticated_bytes",
    "validate_parent",
    "validate_precondition",
]
