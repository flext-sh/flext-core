"""Descriptor-owned staging for one atomic file replacement."""

from __future__ import annotations

import errno
import os
import secrets
import stat
from pathlib import Path

from flext_cli._utilities._atomic_file_descriptor import ParentDescriptor, open_entry
from flext_cli._utilities._atomic_file_mode import assert_observed_mode

_SECURE_CREATE_MODE = 0o600


def temporary_path(parent: ParentDescriptor) -> Path:
    """Return one unpredictable sibling name without probing or retrying."""
    return parent.path / f".flext-atomic-{secrets.token_hex(16)}.tmp"


def require_mode_capability(path: Path, permission_mode: int | None) -> None:
    """Fail before staging if an exact requested mode cannot use its descriptor."""
    if permission_mode is not None and os.chmod not in os.supports_fd:
        message = "descriptor permission changes are unsupported"
        raise OSError(errno.ENOTSUP, message, path)


def create_descriptor(parent: ParentDescriptor, temporary: Path) -> int:
    """Create one exclusive, securely permissioned sibling through ``dir_fd``."""
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_BINARY", 0)
    )
    return open_entry(parent, temporary, flags, mode=_SECURE_CREATE_MODE)


def write_and_sync(
    descriptor: int, temporary: Path, content: bytes, permission_mode: int | None
) -> int:
    """Write exact bytes, materialize exact mode, and sync the open inode."""
    remaining = memoryview(content)
    while remaining:
        written = os.write(descriptor, remaining)
        if written == 0:
            message = f"atomic temporary write made no progress: {temporary}"
            raise OSError(errno.EIO, message, temporary)
        remaining = remaining[written:]
    if permission_mode is not None:
        os.chmod(descriptor, permission_mode)
        assert_observed_mode(temporary, os.fstat(descriptor), permission_mode)
    os.fsync(descriptor)
    return stat.S_IMODE(os.fstat(descriptor).st_mode)


__all__: list[str] = [
    "create_descriptor",
    "require_mode_capability",
    "temporary_path",
    "write_and_sync",
]
