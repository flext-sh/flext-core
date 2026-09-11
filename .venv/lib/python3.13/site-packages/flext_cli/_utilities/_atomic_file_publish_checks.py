"""Physical identity and publication-result checks for staged files."""

from __future__ import annotations

import errno
import os
from pathlib import Path

from flext_cli._utilities._atomic_file_descriptor import ParentDescriptor
from flext_cli._utilities._atomic_file_mode import validate_mode_precondition
from flext_cli._utilities._atomic_file_state import (
    destination_state as file_destination_state,
)
from flext_cli._utilities._atomic_file_state import (
    identity as file_identity,
    read_authenticated_bytes,
)

_IDENTITY_TUPLE_LENGTH = 2


def validate_expected_identity(
    path: Path, expected_bytes: bytes | None, expected_identity: tuple[int, int] | None
) -> None:
    """Require destination presence and physical identity together."""
    if (expected_bytes is None) != (expected_identity is None):
        message = "destination bytes and physical identity must be present together"
        raise OSError(errno.EINVAL, message, path)
    if expected_identity is not None:
        validate_identity(path, expected_identity, label="expected_identity")


def validate_identity(path: Path, value: tuple[int, int], *, label: str) -> None:
    """Require one strict non-negative device and inode pair."""
    if (
        len(value) != _IDENTITY_TUPLE_LENGTH
        or any(isinstance(item, bool) for item in value)
        or any(item < 0 for item in value)
    ):
        message = f"{label} must be a non-negative device and inode pair"
        raise OSError(errno.EINVAL, message, path)


def require_identity(
    path: Path, state: os.stat_result | None, expected: tuple[int, int] | None
) -> None:
    """Require an observed physical identity to match its caller snapshot."""
    observed = None if state is None else file_identity(state)
    if observed != expected:
        message = f"atomic file physical identity changed: {path}"
        raise OSError(errno.ESTALE, message, path)


def require_distinct_inode(
    destination: Path,
    destination_state: os.stat_result | None,
    staged_identity: tuple[int, int],
) -> None:
    """Reject lexical aliases that identify the same physical file."""
    if (
        destination_state is not None
        and file_identity(destination_state) == staged_identity
    ):
        message = "staged file and atomic destination share one inode"
        raise OSError(errno.EINVAL, message, destination)


def validate_devices(
    destination: Path,
    destination_parent: ParentDescriptor,
    destination_state: os.stat_result | None,
    staged: Path,
    staged_parent: ParentDescriptor,
    staged_state: os.stat_result,
) -> None:
    """Require both entries and parents to occupy one filesystem."""
    device = destination_parent.state.st_dev
    if (
        staged_parent.state.st_dev != device
        or staged_state.st_dev != staged_parent.state.st_dev
        or (destination_state is not None and destination_state.st_dev != device)
    ):
        message = f"atomic staged and destination entries span filesystems: {staged}"
        raise OSError(errno.EXDEV, message, destination)


def validate_publication(
    destination_parent: ParentDescriptor,
    destination: Path,
    staged_parent: ParentDescriptor,
    staged: Path,
    staged_bytes: bytes,
    staged_mode: int,
    staged_identity: tuple[int, int],
) -> os.stat_result:
    """Prove replacement consumed the staged name and retained its exact state."""
    if file_destination_state(staged, parent=staged_parent) is not None:
        message = f"atomic staged file still exists after publication: {staged}"
        raise OSError(errno.ESTALE, message, staged)
    published = file_destination_state(destination, parent=destination_parent)
    if published is None or file_identity(published) != staged_identity:
        message = f"published atomic file has another identity: {destination}"
        raise OSError(errno.ESTALE, message, destination)
    if (
        read_authenticated_bytes(destination, published, parent=destination_parent)
        != staged_bytes
    ):
        message = f"published atomic file bytes differ: {destination}"
        raise OSError(errno.ESTALE, message, destination)
    validate_mode_precondition(destination, published, staged_mode)
    return published


__all__: list[str] = [
    "require_distinct_inode",
    "require_identity",
    "validate_devices",
    "validate_expected_identity",
    "validate_identity",
    "validate_publication",
]
