"""Stable descriptor reads for atomic file state authentication."""

from __future__ import annotations

import errno
import os
from pathlib import Path

from flext_cli._utilities._atomic_file_descriptor import ParentDescriptor, open_entry


def read_descriptor_bytes(
    parent: ParentDescriptor, path: Path, expected: os.stat_result
) -> bytes:
    """Read all bytes while one descriptor retains the expected exact state."""
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = open_entry(parent, path, flags)
    try:
        content = _read_stable_descriptor(descriptor, path, expected)
    except BaseException as operation_error:
        _close_after_failure(descriptor, path, operation_error)
        raise
    os.close(descriptor)
    return content


def state_key(state: os.stat_result) -> tuple[int, ...]:
    """Return fields that identify one authorized regular-file version."""
    return (
        state.st_dev,
        state.st_ino,
        state.st_mode,
        state.st_nlink,
        state.st_uid,
        state.st_gid,
        state.st_size,
        state.st_mtime_ns,
        state.st_ctime_ns,
        getattr(state, "st_file_attributes", 0),
        getattr(state, "st_reparse_tag", 0),
    )


def _read_stable_descriptor(
    descriptor: int, path: Path, expected: os.stat_result
) -> bytes:
    if state_key(os.fstat(descriptor)) != state_key(expected):
        _raise_changed(path)
    chunks: list[bytes] = []
    while chunk := os.read(descriptor, 1024 * 1024):
        chunks.append(chunk)
    if state_key(os.fstat(descriptor)) != state_key(expected):
        _raise_changed(path)
    return b"".join(chunks)


def _close_after_failure(
    descriptor: int, path: Path, operation_error: BaseException
) -> None:
    try:
        os.close(descriptor)
    except OSError as close_error:
        message = (
            f"atomic read failed ({operation_error}); close failed ({close_error})"
        )
        group_message = "atomic read and descriptor close failed"
        if isinstance(operation_error, Exception):
            causes = ExceptionGroup(group_message, [operation_error, close_error])
            raise OSError(errno.EIO, message, path) from causes
        raise BaseExceptionGroup(
            group_message, [operation_error, close_error]
        ) from close_error


def _raise_changed(path: Path) -> None:
    message = f"atomic destination changed during authenticated read: {path}"
    raise OSError(errno.ESTALE, message, path)


__all__: list[str] = ["read_descriptor_bytes", "state_key"]
