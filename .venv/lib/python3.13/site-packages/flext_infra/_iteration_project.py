"""Project root resolution iteration mixin.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_infra import c

if TYPE_CHECKING:
    from pathlib import Path


class FlextInfraUtilitiesIterationProject:
    """Static helpers for resolving project roots from file paths."""

    @staticmethod
    def resolve_project_root(file_path: Path) -> Path | None:
        """Walk up from file_path to find the project root containing pyproject.toml."""
        current = file_path.parent
        for _ in range(10):
            if (current / c.Infra.PYPROJECT_FILENAME).is_file():
                return current
            parent = current.parent
            if parent == current:
                break
            current = parent
        return None


__all__: list[str] = ["FlextInfraUtilitiesIterationProject"]
