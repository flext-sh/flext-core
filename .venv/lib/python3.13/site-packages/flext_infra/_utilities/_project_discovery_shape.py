"""Project shape helpers for flext-infra discovery utilities.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_infra._utilities.dependencies import FlextInfraUtilitiesDependencies
from flext_infra._utilities.pyproject import FlextInfraUtilitiesPyproject
from flext_infra.constants import c

if TYPE_CHECKING:
    from pathlib import Path


class FlextInfraUtilitiesProjectDiscoveryShapeMixin:
    """Private project-shape predicates for workspace project discovery."""

    @staticmethod
    def _looks_like_project(
        path: Path,
        *,
        effective_scan_dirs: frozenset[str],
        configured_project_set: frozenset[str],
    ) -> bool:
        """Return whether one path matches the canonical governed project shape."""
        if not path.is_dir():
            return False
        pyproject_path = path / c.Infra.PYPROJECT_FILENAME
        if pyproject_path.exists() and (
            path.name in configured_project_set
            or (path / c.Infra.MAKEFILE_FILENAME).exists()
        ):
            return True
        payload = FlextInfraUtilitiesPyproject.pyproject_payload(pyproject_path)
        if not payload:
            return False
        dependency_names: set[str] = set(
            FlextInfraUtilitiesDependencies.declared_dependency_names_from_payload(
                payload
            )
        )
        if c.Infra.PKG_CORE in dependency_names:
            return True
        if effective_scan_dirs:
            return any((path / dir_name).is_dir() for dir_name in effective_scan_dirs)
        return True


__all__: list[str] = ["FlextInfraUtilitiesProjectDiscoveryShapeMixin"]
