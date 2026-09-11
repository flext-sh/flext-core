"""Lexical and physical path validation for atomic file owners."""

from __future__ import annotations

import errno
import os
import stat
from pathlib import Path


def validate_atomic_path(path: Path) -> Path:
    """Require one absolute, normalized file pathname without traversal."""
    normalized = Path(os.path.normpath(path))
    if (
        not path.is_absolute()
        or not path.name
        or ".." in path.parts
        or normalized != path
    ):
        message = f"atomic file path is not absolute and normalized: {path}"
        raise OSError(errno.EINVAL, message, path)
    return path


def validate_parent_path(parent: Path) -> os.stat_result:
    """Return one physical directory state without following its final entry."""
    normalized = Path(os.path.normpath(parent))
    if not parent.is_absolute() or ".." in parent.parts or normalized != parent:
        message = f"atomic file parent is not absolute and normalized: {parent}"
        raise OSError(errno.EINVAL, message, parent)
    try:
        state = parent.lstat()
    except FileNotFoundError as exc:
        message = f"atomic destination parent is missing: {parent}"
        raise FileNotFoundError(errno.ENOENT, message, parent) from exc
    validate_directory_state(parent, state)
    return state


def validate_directory_state(path: Path, state: os.stat_result) -> None:
    """Require one non-reparse physical directory state."""
    if not stat.S_ISDIR(state.st_mode) or is_reparse_point(state):
        message = f"atomic destination parent is not a real directory: {path}"
        raise NotADirectoryError(errno.ENOTDIR, message, path)


def identity(state: os.stat_result) -> tuple[int, int]:
    """Return the physical device and inode identity of one filesystem object."""
    return (state.st_dev, state.st_ino)


def is_reparse_point(state: os.stat_result) -> bool:
    """Identify Windows reparse-point aliases without platform branching."""
    attributes = getattr(state, "st_file_attributes", 0)
    marker = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(attributes & marker)


__all__: list[str] = [
    "identity",
    "is_reparse_point",
    "validate_atomic_path",
    "validate_directory_state",
    "validate_parent_path",
]
