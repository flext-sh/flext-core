"""Public API for flext-infra.

Provides access to infrastructure services for workspace management, validation,
dependency handling, and build orchestration in the FLEXT ecosystem.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from flext_core import cleanup_submodule_namespace, lazy_getattr

if TYPE_CHECKING:
    from flext_infra import (
        KNOWN_VERBS,
        REPORTS_DIR_NAME,
        BaseMkGenerator,
        BaseMkGenerator as FlextInfraBaseMkGenerator,
        CommandRunner,
        CommandRunner as FlextInfraCommandRunner,
        DiscoveryService,
        DiscoveryService as FlextInfraDiscoveryService,
        FlextInfraConstants,
        FlextInfraConstants as c,
        FlextInfraModels,
        FlextInfraModels as m,
        FlextInfraPatterns,
        FlextInfraProtocols,
        FlextInfraProtocols as p,
        FlextInfraTypes,
        FlextInfraTypes as t,
        FlextInfraUtilities,
        FlextInfraUtilities as u,
        GitService,
        GitService as FlextInfraGitService,
        JsonService,
        JsonService as FlextInfraJsonService,
        PathResolver,
        PathResolver as FlextInfraPathResolver,
        ProjectSelector,
        ProjectSelector as FlextInfraProjectSelector,
        ReleaseOrchestrator,
        ReleaseOrchestrator as FlextInfraReleaseOrchestrator,
        ReportingService,
        ReportingService as FlextInfraReportingService,
        TemplateEngine,
        TemplateEngine as BaseMkTemplateEngine,
        TemplateEngine as FlextInfraTemplateEngine,
        TomlService,
        TomlService as FlextInfraTomlService,
        VersioningService,
        VersioningService as FlextInfraVersioningService,
        __version__,
        __version_info__,
    )

# Lazy import mapping: export_name -> (module_path, attr_name)
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "BaseMkGenerator": ("flext_infra.basemk", "BaseMkGenerator"),
    "BaseMkTemplateEngine": ("flext_infra.basemk", "TemplateEngine"),
    "CommandRunner": ("flext_infra.subprocess", "CommandRunner"),
    "DiscoveryService": ("flext_infra.discovery", "DiscoveryService"),
    "FlextInfraBaseMkGenerator": ("flext_infra.basemk", "BaseMkGenerator"),
    "FlextInfraCommandRunner": ("flext_infra.subprocess", "CommandRunner"),
    "FlextInfraConstants": ("flext_infra.constants", "FlextInfraConstants"),
    "FlextInfraDiscoveryService": ("flext_infra.discovery", "DiscoveryService"),
    "FlextInfraGitService": ("flext_infra.git", "GitService"),
    "FlextInfraJsonService": ("flext_infra.json_io", "JsonService"),
    "FlextInfraModels": ("flext_infra.models", "FlextInfraModels"),
    "FlextInfraPathResolver": ("flext_infra.paths", "PathResolver"),
    "FlextInfraPatterns": ("flext_infra.patterns", "FlextInfraPatterns"),
    "FlextInfraProjectSelector": ("flext_infra.selection", "ProjectSelector"),
    "FlextInfraProtocols": ("flext_infra.protocols", "FlextInfraProtocols"),
    "FlextInfraReleaseOrchestrator": ("flext_infra.release", "ReleaseOrchestrator"),
    "FlextInfraReportingService": ("flext_infra.reporting", "ReportingService"),
    "FlextInfraTemplateEngine": ("flext_infra.templates", "TemplateEngine"),
    "FlextInfraTomlService": ("flext_infra.toml_io", "TomlService"),
    "FlextInfraTypes": ("flext_infra.typings", "FlextInfraTypes"),
    "FlextInfraUtilities": ("flext_infra.utilities", "FlextInfraUtilities"),
    "FlextInfraVersioningService": ("flext_infra.versioning", "VersioningService"),
    "GitService": ("flext_infra.git", "GitService"),
    "JsonService": ("flext_infra.json_io", "JsonService"),
    "KNOWN_VERBS": ("flext_infra.reporting", "KNOWN_VERBS"),
    "PathResolver": ("flext_infra.paths", "PathResolver"),
    "ProjectSelector": ("flext_infra.selection", "ProjectSelector"),
    "REPORTS_DIR_NAME": ("flext_infra.reporting", "REPORTS_DIR_NAME"),
    "ReleaseOrchestrator": ("flext_infra.release", "ReleaseOrchestrator"),
    "ReportingService": ("flext_infra.reporting", "ReportingService"),
    "TemplateEngine": ("flext_infra.templates", "TemplateEngine"),
    "TomlService": ("flext_infra.toml_io", "TomlService"),
    "VersioningService": ("flext_infra.versioning", "VersioningService"),
    "__version__": ("flext_infra.__version__", "__version__"),
    "__version_info__": ("flext_infra.__version__", "__version_info__"),
    "c": ("flext_infra.constants", "FlextInfraConstants"),
    "m": ("flext_infra.models", "FlextInfraModels"),
    "p": ("flext_infra.protocols", "FlextInfraProtocols"),
    "t": ("flext_infra.typings", "FlextInfraTypes"),
    "u": ("flext_infra.utilities", "FlextInfraUtilities"),
}

__all__ = [
    "KNOWN_VERBS",
    "REPORTS_DIR_NAME",
    "BaseMkGenerator",
    "BaseMkTemplateEngine",
    "CommandRunner",
    "DiscoveryService",
    "FlextInfraBaseMkGenerator",
    "FlextInfraCommandRunner",
    "FlextInfraConstants",
    "FlextInfraDiscoveryService",
    "FlextInfraGitService",
    "FlextInfraJsonService",
    "FlextInfraModels",
    "FlextInfraPathResolver",
    "FlextInfraPatterns",
    "FlextInfraProjectSelector",
    "FlextInfraProtocols",
    "FlextInfraReleaseOrchestrator",
    "FlextInfraReportingService",
    "FlextInfraTemplateEngine",
    "FlextInfraTomlService",
    "FlextInfraTypes",
    "FlextInfraUtilities",
    "FlextInfraVersioningService",
    "GitService",
    "JsonService",
    "PathResolver",
    "ProjectSelector",
    "ReleaseOrchestrator",
    "ReportingService",
    "TemplateEngine",
    "TomlService",
    "VersioningService",
    "__version__",
    "__version_info__",
    "c",
    "m",
    "p",
    "t",
    "u",
]


def __getattr__(name: str) -> Any:  # noqa: ANN401
    """Lazy-load module attributes on first access (PEP 562)."""
    return lazy_getattr(name, _LAZY_IMPORTS, globals(), __name__)


def __dir__() -> list[str]:
    """Return list of available attributes for dir() and autocomplete."""
    return sorted(__all__)


cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)
