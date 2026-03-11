"""Path resolution utilities for workspace navigation.

Provides static utility methods for path resolution with r error handling,
replacing bare functions with a utility class.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import os
from pathlib import Path

from flext_core import r
from flext_infra import c


class FlextInfraUtilitiesPaths:
    """Utility class for workspace path resolution.

    Provides static methods for path resolution with r-wrapped error handling,
    replacing bare functions from ``scripts/libs/paths.py``.
    """

    @staticmethod
    def workspace_root(path: str | Path = ".") -> r[Path]:
        """Resolve and return the absolute path to the workspace root.

        Checks ``FLEXT_WORKSPACE_ROOT`` env var first, then falls back to
        resolving *path*.

        Args:
            path: A starting path, defaults to the current directory.

        Returns:
            r[Path] with the resolved absolute path.

        """
        try:
            env_root = os.getenv("FLEXT_WORKSPACE_ROOT")
            if env_root:
                candidate = Path(env_root).expanduser().resolve()
                if candidate.is_dir():
                    return r[Path].ok(candidate)
            resolved = Path(path).resolve()
            return r[Path].ok(resolved)
        except (OSError, RuntimeError, TypeError) as exc:
            return r[Path].fail(f"failed to resolve workspace root: {exc}")

    @staticmethod
    def workspace_root_from_file(file: str | Path) -> r[Path]:
        """Resolve workspace root by walking up from file location.

        Finds the first directory containing .git, Makefile, and pyproject.toml.

        Args:
            file: Path to a file (usually __file__).

        Returns:
            r[Path] with absolute path to workspace root,
            or failure if not found.

        """
        try:
            current = Path(file).resolve()
            if current.is_file():
                current = current.parent
            for parent in [current, *list(current.parents)]:
                if all(
                    (parent / marker).exists()
                    for marker in c.Infra.Paths.WORKSPACE_MARKERS
                ):
                    return r[Path].ok(parent)
            return r[Path].fail(
                f"workspace root not found (looking for {c.Infra.Paths.WORKSPACE_MARKERS}) starting from {file}",
            )
        except (OSError, RuntimeError, TypeError) as exc:
            return r[Path].fail(f"failed to resolve workspace root: {exc}")

    @staticmethod
    def resolve_workspace_root(file: str | Path) -> Path:
        """Resolve workspace root by walking up from file location to .gitmodules.

        This specific resolution pattern is used by orchestration services
        to identify the FLEXT workspace root.

        Args:
            file: Path to a file (usually __file__).

        Returns:
            Absolute path to workspace root.

        """
        here = Path(file).resolve()
        for candidate in here.parents:
            if (candidate / c.Infra.Files.GITMODULES).exists():
                return candidate
        return here.parents[4]


__all__ = ["FlextInfraUtilitiesPaths"]
