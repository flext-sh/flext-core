"""Guarded deletion for transaction rollback under a caller-held lock."""

from __future__ import annotations

import errno
from pathlib import Path

from flext_cli._utilities._atomic_file_descriptor import parent_descriptor, unlink_entry
from flext_cli._utilities._atomic_file_mode import validate_mode_precondition
from flext_cli._utilities._atomic_file_state import (
    assert_destination_unchanged,
    destination_state,
    validate_precondition,
)


def remove_guarded_file(
    path: Path, *, expected_bytes: bytes, expected_mode: int
) -> None:
    """Unlink the exact regular-file version authorized by the caller."""
    with parent_descriptor(path, unlink=True) as parent:
        expected = destination_state(path, parent=parent)
        validate_precondition(
            path, expected, expected_bytes, enabled=True, parent=parent
        )
        validate_mode_precondition(path, expected, expected_mode)
        assert_destination_unchanged(path, expected, parent=parent)
        unlink_entry(parent, path)
        if destination_state(path, parent=parent) is not None:
            message = f"atomic destination still exists after delete: {path}"
            raise OSError(errno.ESTALE, message, path)


__all__: list[str] = ["remove_guarded_file"]
