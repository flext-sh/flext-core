"""Descriptor-bound parent ownership for atomic file operations."""

from __future__ import annotations

import errno
import os
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from flext_cli._utilities._atomic_file_path import (
    identity,
    validate_atomic_path,
    validate_directory_state,
    validate_parent_path,
)


@dataclass(slots=True, frozen=True)
class ParentDescriptor:
    """One open physical parent directory bound to its lexical pathname."""

    path: Path
    descriptor: int
    state: os.stat_result


@contextmanager
def parent_descriptor(
    path: Path, *, replace: bool = False, unlink: bool = False
) -> Generator[ParentDescriptor]:
    """Yield one authenticated parent descriptor with required OS capabilities."""
    validated = validate_atomic_path(path)
    _require_capabilities(replace=replace, unlink=unlink)
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    nofollow_flag = getattr(os, "O_NOFOLLOW", 0)
    flags = (
        os.O_RDONLY
        | directory_flag
        | nofollow_flag
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_BINARY", 0)
    )
    descriptor = os.open(validated.parent, flags)
    try:
        state = os.fstat(descriptor)
        validate_directory_state(validated.parent, state)
        handle = ParentDescriptor(validated.parent, descriptor, state)
        assert_parent_unchanged(handle)
    except BaseException as operation_error:
        _close_after_failure(descriptor, validated.parent, operation_error)
        raise
    try:
        yield handle
        assert_parent_unchanged(handle)
    except BaseException as operation_error:
        _close_after_failure(descriptor, validated.parent, operation_error)
        raise
    os.close(descriptor)


def assert_parent_unchanged(parent: ParentDescriptor) -> None:
    """Require descriptor and pathname to retain the opened directory identity."""
    descriptor_state = os.fstat(parent.descriptor)
    validate_directory_state(parent.path, descriptor_state)
    pathname_state = validate_parent_path(parent.path)
    expected = identity(parent.state)
    if identity(descriptor_state) != expected or identity(pathname_state) != expected:
        message = f"atomic file parent identity changed: {parent.path}"
        raise OSError(errno.ESTALE, message, parent.path)


def entry_stat(parent: ParentDescriptor, path: Path) -> os.stat_result:
    """Read one final entry relative to its authenticated parent descriptor."""
    _require_entry(parent, path)
    return os.stat(path.name, dir_fd=parent.descriptor, follow_symlinks=False)


def open_entry(
    parent: ParentDescriptor, path: Path, flags: int, *, mode: int | None = None
) -> int:
    """Open one final entry relative to its authenticated parent descriptor."""
    _require_entry(parent, path)
    nofollow_flag = getattr(os, "O_NOFOLLOW", 0)
    guarded_flags = flags | nofollow_flag
    if mode is None:
        return os.open(path.name, guarded_flags, dir_fd=parent.descriptor)
    return os.open(path.name, guarded_flags, mode, dir_fd=parent.descriptor)


def unlink_entry(parent: ParentDescriptor, path: Path) -> None:
    """Unlink one entry from the still-authorized physical parent."""
    _require_entry(parent, path)
    assert_parent_unchanged(parent)
    os.unlink(path.name, dir_fd=parent.descriptor)


def replace_entry(
    source_parent: ParentDescriptor,
    source: Path,
    destination_parent: ParentDescriptor,
    destination: Path,
) -> None:
    """Replace one entry using only authenticated directory descriptors."""
    _require_entry(source_parent, source)
    _require_entry(destination_parent, destination)
    assert_parent_unchanged(source_parent)
    assert_parent_unchanged(destination_parent)
    os.replace(
        source.name,
        destination.name,
        src_dir_fd=source_parent.descriptor,
        dst_dir_fd=destination_parent.descriptor,
    )


def _require_capabilities(*, replace: bool, unlink: bool) -> None:
    operations: list[tuple[str, object]] = [("open", os.open), ("stat", os.stat)]
    if replace:
        # CPython exposes ``src_dir_fd``/``dst_dir_fd`` on ``os.replace`` via
        # the same renameat primitive as ``os.rename``, but records only
        # ``os.rename`` in ``supports_dir_fd`` on supported POSIX runtimes.
        operations.append(("replace", os.rename))
    if unlink:
        operations.append(("unlink", os.unlink))
    missing = [
        name for name, operation in operations if operation not in os.supports_dir_fd
    ]
    if os.stat not in os.supports_follow_symlinks:
        missing.append("stat(follow_symlinks=False)")
    if not getattr(os, "O_DIRECTORY", 0):
        missing.append("O_DIRECTORY")
    if not getattr(os, "O_NOFOLLOW", 0):
        missing.append("O_NOFOLLOW")
    if missing:
        unsupported = sorted(set(missing))
        message = f"descriptor-bound atomic files are unsupported: {unsupported}"
        raise OSError(errno.ENOTSUP, message)


def _require_entry(parent: ParentDescriptor, path: Path) -> None:
    validated = validate_atomic_path(path)
    if validated.parent != parent.path:
        message = f"atomic file does not belong to authenticated parent: {path}"
        raise OSError(errno.EINVAL, message, path)


def _close_after_failure(
    descriptor: int, path: Path, operation_error: BaseException
) -> None:
    try:
        os.close(descriptor)
    except OSError as close_error:
        message = (
            f"atomic operation failed ({operation_error}); close failed ({close_error})"
        )
        group_message = "atomic operation and descriptor close failed"
        if isinstance(operation_error, Exception):
            causes = ExceptionGroup(group_message, [operation_error, close_error])
            raise OSError(errno.EIO, message, path) from causes
        raise BaseExceptionGroup(
            group_message, [operation_error, close_error]
        ) from close_error


__all__: list[str] = [
    "ParentDescriptor",
    "assert_parent_unchanged",
    "entry_stat",
    "open_entry",
    "parent_descriptor",
    "replace_entry",
    "unlink_entry",
]
