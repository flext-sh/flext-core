"""Guarded publication of caller-owned staged files."""

from __future__ import annotations

import errno
import os
from pathlib import Path

from flext_cli._utilities._atomic_file_descriptor import (
    parent_descriptor,
    replace_entry,
)
from flext_cli._utilities._atomic_file_mode import (
    validate_guarded_mode_tuple,
    validate_mode,
    validate_mode_precondition,
)
from flext_cli._utilities._atomic_file_path import validate_atomic_path
from flext_cli._utilities._atomic_file_publish_checks import (
    require_distinct_inode,
    require_identity,
    validate_devices,
    validate_expected_identity,
    validate_identity,
    validate_publication,
)
from flext_cli._utilities._atomic_file_state import (
    assert_destination_unchanged,
    assert_temporary_owned,
)
from flext_cli._utilities._atomic_file_state import (
    destination_state as file_destination_state,
)
from flext_cli._utilities._atomic_file_state import (
    validate_precondition as validate_state_precondition,
)


def publish_guarded_staged_file(
    destination: Path,
    staged: Path,
    *,
    expected_bytes: bytes | None,
    expected_mode: int | None,
    expected_identity: tuple[int, int] | None,
    staged_bytes: bytes,
    staged_mode: int,
    staged_identity: tuple[int, int],
) -> os.stat_result:
    """Move one exact caller-owned staged file over an exact destination state.

    The caller owns both the cooperative lock and staged-file lifecycle.  A
    failure before replacement leaves the staged path in place. A completed
    replace consumes it; any subsequent validation failure requires the caller's
    journal to classify the destination. Parent-directory power-loss durability
    is outside this primitive's contract.
    """
    destination = validate_atomic_path(destination)
    staged = validate_atomic_path(staged)
    if destination == staged:
        message = "staged file and atomic destination must differ"
        raise OSError(errno.EINVAL, message, destination)
    validate_guarded_mode_tuple(destination, expected_bytes, expected_mode)
    required_staged_mode = validate_mode(staged_mode, label="staged_mode")
    if required_staged_mode is None:
        message = "staged_mode is required"
        raise OSError(errno.EINVAL, message, staged)
    validate_expected_identity(destination, expected_bytes, expected_identity)
    validate_identity(staged, staged_identity, label="staged_identity")
    with (
        parent_descriptor(destination, replace=True) as destination_parent,
        parent_descriptor(staged, replace=True) as staged_parent,
    ):
        destination_state = file_destination_state(
            destination, parent=destination_parent
        )
        require_identity(destination, destination_state, expected_identity)
        validate_state_precondition(
            destination,
            destination_state,
            expected_bytes,
            enabled=True,
            parent=destination_parent,
        )
        validate_mode_precondition(destination, destination_state, expected_mode)
        staged_state = file_destination_state(staged, parent=staged_parent)
        if staged_state is None:
            message = f"atomic staged file is missing: {staged}"
            raise FileNotFoundError(errno.ENOENT, message, staged)
        require_identity(staged, staged_state, staged_identity)
        require_distinct_inode(destination, destination_state, staged_identity)
        validate_devices(
            destination,
            destination_parent,
            destination_state,
            staged,
            staged_parent,
            staged_state,
        )
        validate_state_precondition(
            staged, staged_state, staged_bytes, enabled=True, parent=staged_parent
        )
        validate_mode_precondition(staged, staged_state, required_staged_mode)
        assert_destination_unchanged(
            destination, destination_state, parent=destination_parent
        )
        assert_temporary_owned(staged, staged_identity, parent=staged_parent)
        replace_entry(staged_parent, staged, destination_parent, destination)
        return validate_publication(
            destination_parent,
            destination,
            staged_parent,
            staged,
            staged_bytes,
            required_staged_mode,
            staged_identity,
        )


__all__: list[str] = ["publish_guarded_staged_file"]
