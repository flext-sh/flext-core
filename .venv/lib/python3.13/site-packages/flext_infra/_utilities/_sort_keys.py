"""Typed sort keys shared by filesystem owners."""

from __future__ import annotations

from pathlib import Path


def path_depth(path: Path) -> int:
    """Return the number of components in a path."""
    return len(path.parts)


def path_depth_then_text(path: Path) -> tuple[int, str]:
    """Order paths by depth and then their stable POSIX representation."""
    return path_depth(path), path.as_posix()


# Internal owner: direct module imports are intentional; no facade ABI is published.
__all__: tuple[str, ...] = ()
