"""Path resolution service for workspace navigation.

Wraps path resolution functions with FlextResult error handling,
replacing bare functions with a service class.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

from flext_core.result import FlextResult, r
from flext_infra.constants import c

_WORKSPACE_MARKERS: frozenset[str] = frozenset(
    {
        ".git",
        c.Files.MAKEFILE_FILENAME,
        c.Files.PYPROJECT_FILENAME,
    }
)


class PathResolver:
    """Infrastructure service for workspace path resolution.

    Provides FlextResult-wrapped path resolution, replacing the bare
    functions from ``scripts/libs/paths.py``.
    """

    def workspace_root(self, path: str | Path = ".") -> FlextResult[Path]:
        """Resolve and return the absolute path to the workspace root.

        Args:
            path: A starting path, defaults to the current directory.

        Returns:
            FlextResult[Path] with the resolved absolute path.

        """
        try:
            resolved = Path(path).resolve()
            return r[Path].ok(resolved)
        except (OSError, RuntimeError, TypeError) as exc:
            return r[Path].fail(f"failed to resolve workspace root: {exc}")

    def repo_root_from_script(self, script_file: str | Path) -> FlextResult[Path]:
        """Resolve the repository root based on a script file location.

        Args:
            script_file: Path to the script file (usually __file__).

        Returns:
            FlextResult[Path] with the absolute path to the repo root.

        """
        try:
            resolved = Path(script_file).resolve().parents[1]
            return r[Path].ok(resolved)
        except (IndexError, OSError, TypeError) as exc:
            return r[Path].fail(f"failed to resolve repo root: {exc}")

    def workspace_root_from_file(self, file: str | Path) -> FlextResult[Path]:
        """Resolve workspace root by walking up from file location.

        Finds the first directory containing .git, Makefile, and pyproject.toml.

        Args:
            file: Path to a file (usually __file__).

        Returns:
            FlextResult[Path] with absolute path to workspace root,
            or failure if not found.

        """
        try:
            current = Path(file).resolve()
            if current.is_file():
                current = current.parent

            for parent in [current, *list(current.parents)]:
                if all((parent / marker).exists() for marker in _WORKSPACE_MARKERS):
                    return r[Path].ok(parent)

            return r[Path].fail(
                f"workspace root not found (looking for {_WORKSPACE_MARKERS}) "
                f"starting from {file}",
            )
        except (OSError, RuntimeError, TypeError) as exc:
            return r[Path].fail(f"failed to resolve workspace root: {exc}")


__all__ = ["PathResolver"]
