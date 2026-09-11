"""Namespace enforcement configuration helpers for flext-infra utilities.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_infra._utilities.git import FlextInfraUtilitiesGit
from flext_infra._utilities.pyproject import FlextInfraUtilitiesPyproject
from flext_infra.constants import c

if TYPE_CHECKING:
    from pathlib import Path

    from flext_infra.typings import t


class FlextInfraUtilitiesNamespaceConfig:
    """Static helpers for reading namespace enforcement configuration."""

    @staticmethod
    def namespace_meta(project_root: Path) -> t.JsonMapping:
        """Return optional ``tool.flext.namespace`` metadata for one project."""
        flext_meta = FlextInfraUtilitiesPyproject.tool_flext_meta(project_root)
        namespace = flext_meta.get("namespace")
        return namespace if isinstance(namespace, dict) else {}

    @staticmethod
    def namespace_enabled(project_root: Path) -> bool:
        """Return whether namespace enforcement is enabled for a project."""
        enabled = FlextInfraUtilitiesNamespaceConfig.namespace_meta(project_root).get(
            "enabled", True
        )
        return enabled if isinstance(enabled, bool) else True

    @staticmethod
    def namespace_scan_dirs(project_root: Path) -> frozenset[str]:
        """Return configured scan dirs for namespace enforcement.

        Priority:
        1. Explicit ``[tool.flext.namespace] scan_dirs`` in pyproject.toml.
        2. Git-tracked top-level directories that exist on disk (dynamic).
        3. Fixed candidate list filtered by ``is_dir()`` (fallback).
        """
        configured = FlextInfraUtilitiesNamespaceConfig.namespace_meta(
            project_root
        ).get("scan_dirs")
        if isinstance(configured, list):
            normalized = frozenset(
                str(item).strip() for item in configured if str(item).strip()
            )
            if normalized:
                return normalized
        tracked = FlextInfraUtilitiesGit.git_tracked_top_level_dir_names(project_root)
        if tracked is not None:
            excluded = c.Infra.COMMON_EXCLUDED_DIRS | {
                name for name in tracked if name.startswith(".")
            }
            dynamic = frozenset(
                name
                for name in tracked
                if name not in excluded and (project_root / name).is_dir()
            )
            if dynamic:
                return dynamic
        candidates = ("docs", "examples", "scripts", "src", "tests")
        return frozenset(name for name in candidates if (project_root / name).is_dir())

    @staticmethod
    def namespace_include_dynamic_dirs(project_root: Path) -> bool:
        """Return whether namespace enforcement should scan non-canonical dirs."""
        include_dynamic_dirs = FlextInfraUtilitiesNamespaceConfig.namespace_meta(
            project_root
        ).get("include_dynamic_dirs")
        return include_dynamic_dirs if isinstance(include_dynamic_dirs, bool) else False


__all__: list[str] = ["FlextInfraUtilitiesNamespaceConfig"]
