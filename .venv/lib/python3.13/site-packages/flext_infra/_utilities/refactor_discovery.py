"""Refactor file discovery utilities.

Centralizes file filtering, project/workspace file collection, and
rope-based nested class propagation helpers.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import fnmatch
from collections.abc import Iterator
from pathlib import Path

from flext_cli import u
from flext_infra._utilities.project_discovery import FlextInfraUtilitiesProjectDiscovery
from flext_infra.constants import c
from flext_infra.iteration import FlextInfraUtilitiesIteration
from flext_infra.models import m
from flext_infra.typings import t


class FlextInfraUtilitiesRefactorDiscovery:
    """File collection and project discovery helpers for refactor services."""

    @staticmethod
    def _resolve_refactor_config(
        settings: t.MappingKV[str, t.Infra.InfraValue],
    ) -> m.Infra.RefactorConfig:
        """Resolve the typed refactor config through the shared CLI DSL."""
        validated: m.Infra.RefactorConfig = m.Infra.RefactorConfig.model_validate(
            u.Cli.rules_resolve_scope(
                dict(settings),
                scope_key=c.Infra.RK_REFACTOR,
                allowed_keys=c.Infra.REFACTOR_CONFIG_KEYS,
            )
        )
        return validated

    @staticmethod
    def filter_refactor_files(
        files: t.SequenceOf[Path],
        *,
        base_path: Path,
        pattern: str = c.Infra.EXT_PYTHON_GLOB,
        ignore_patterns: set[str] | None = None,
        allowed_extensions: set[str] | None = None,
    ) -> Iterator[Path]:
        """Filter candidate files by glob pattern, ignore list, and extension."""
        ign = ignore_patterns or set()
        ext = allowed_extensions or {c.Infra.EXT_PYTHON}
        for f in files:
            if not fnmatch.fnmatch(f.name, pattern):
                continue
            if f.suffix not in ext:
                continue
            rel = str(f.relative_to(base_path))
            if any(fnmatch.fnmatch(rel, ip) for ip in ign):
                continue
            yield f

    @staticmethod
    def _configured_scan_files(
        project: Path, scan_dirs: t.StrSequence
    ) -> t.SequenceOf[Path]:
        """Return files from the exact refactor-owned project directories."""
        files: set[Path] = set()
        for directory_name in scan_dirs:
            files.update(
                FlextInfraUtilitiesIteration.iter_directory_python_files(
                    project / directory_name
                )
            )
        return tuple(sorted(files))

    @staticmethod
    def collect_refactor_project_files(
        settings: t.MappingKV[str, t.Infra.InfraValue],
        project: Path,
        *,
        pattern: str = c.Infra.EXT_PYTHON_GLOB,
    ) -> t.MutableSequenceOf[Path] | None:
        """Iterate and filter Python files under a project.

        Returns None on error.
        """
        refactor_config = FlextInfraUtilitiesRefactorDiscovery._resolve_refactor_config(
            settings
        )
        files = FlextInfraUtilitiesRefactorDiscovery._configured_scan_files(
            project, refactor_config.project_scan_dirs
        )
        ign = refactor_config.ignore_patterns
        ext = refactor_config.file_extensions
        return list(
            FlextInfraUtilitiesRefactorDiscovery.filter_refactor_files(
                files,
                base_path=project,
                pattern=pattern,
                ignore_patterns=set(ign),
                allowed_extensions=set(ext),
            )
        )

    @staticmethod
    def collect_refactor_workspace_files(
        settings: t.MappingKV[str, t.Infra.InfraValue],
        repository_root: Path,
        *,
        pattern: str = c.Infra.EXT_PYTHON_GLOB,
    ) -> t.SequenceOf[Path]:
        """Collect all candidate files under workspace projects."""
        root = repository_root.resolve()
        refactor_config = FlextInfraUtilitiesRefactorDiscovery._resolve_refactor_config(
            settings
        )
        scan_dirs = frozenset(refactor_config.project_scan_dirs)
        projects = FlextInfraUtilitiesProjectDiscovery.discover_project_roots(
            repository_root=root, scan_dirs=scan_dirs or None
        )
        ign = refactor_config.ignore_patterns
        ext = refactor_config.file_extensions
        ignore_patterns = set(ign)
        allowed_extensions = set(ext)
        all_files: t.MutableSequenceOf[Path] = []
        for proj in projects:
            files = FlextInfraUtilitiesRefactorDiscovery._configured_scan_files(
                proj, refactor_config.project_scan_dirs
            )
            all_files.extend(
                FlextInfraUtilitiesRefactorDiscovery.filter_refactor_files(
                    files,
                    base_path=proj,
                    pattern=pattern,
                    ignore_patterns=ignore_patterns,
                    allowed_extensions=allowed_extensions,
                )
            )
        return all_files

    @staticmethod
    def discover_refactor_projects(
        settings: t.MappingKV[str, t.Infra.InfraValue], repository_root: Path
    ) -> t.SequenceOf[Path]:
        """Discover workspace projects using the typed refactor config."""
        root = repository_root.resolve()
        refactor_config = FlextInfraUtilitiesRefactorDiscovery._resolve_refactor_config(
            settings
        )
        scan_dirs = frozenset(refactor_config.project_scan_dirs)
        return FlextInfraUtilitiesProjectDiscovery.discover_project_roots(
            repository_root=root, scan_dirs=scan_dirs or None
        )


__all__: list[str] = ["FlextInfraUtilitiesRefactorDiscovery"]
