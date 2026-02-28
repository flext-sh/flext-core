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
    from flext_infra.__version__ import __version__, __version_info__
    from flext_infra.basemk import (
        FlextInfraBaseMkGenerator,
        FlextInfraBaseMkTemplateEngine,
    )
    from flext_infra.constants import FlextInfraConstants, FlextInfraConstants as c
    from flext_infra.discovery import (
        DiscoveryService,
        DiscoveryService as FlextInfraDiscoveryService,
    )
    from flext_infra.git import GitService, GitService as FlextInfraGitService
    from flext_infra.json_io import JsonService, JsonService as FlextInfraJsonService
    from flext_infra.models import FlextInfraModels, FlextInfraModels as m
    from flext_infra.output import (
        FlextInfraOutput as FlextInfraOutput,
    )
    from flext_infra.paths import PathResolver, PathResolver as FlextInfraPathResolver
    from flext_infra.patterns import FlextInfraPatterns
    from flext_infra.protocols import FlextInfraProtocols, FlextInfraProtocols as p
    from flext_infra.release import (
        ReleaseOrchestrator,
        ReleaseOrchestrator as FlextInfraReleaseOrchestrator,
    )
    from flext_infra.reporting import (
        KNOWN_VERBS,
        REPORTS_DIR_NAME,
        ReportingService,
        ReportingService as FlextInfraReportingService,
    )
    from flext_infra.selection import (
        ProjectSelector,
        ProjectSelector as FlextInfraProjectSelector,
    )
    from flext_infra.subprocess import (
        CommandRunner,
        CommandRunner as FlextInfraCommandRunner,
    )
    from flext_infra.templates import (
        FlextInfraTemplateEngine as FlextInfraTemplateEngine,
    )
    from flext_infra.toml_io import TomlService, TomlService as FlextInfraTomlService
    from flext_infra.typings import FlextInfraTypes, FlextInfraTypes as t
    from flext_infra.utilities import FlextInfraUtilities, FlextInfraUtilities as u
    from flext_infra.versioning import (
        VersioningService,
        VersioningService as FlextInfraVersioningService,
    )

# Lazy import mapping: export_name -> (module_path, attr_name)
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "FlextInfraBaseMkGenerator": ("flext_infra.basemk", "FlextInfraBaseMkGenerator"),
    "FlextInfraBaseMkTemplateEngine": ("flext_infra.basemk", "FlextInfraBaseMkTemplateEngine"),
    "FlextInfraCheckConfigFixer": ("flext_infra.check.services", "FlextInfraConfigFixer"),
    "FlextInfraCheckWorkspaceChecker": ("flext_infra.check.services", "FlextInfraWorkspaceChecker"),
    "FlextInfraDiscoveryService": ("flext_infra.discovery", "FlextInfraDiscoveryService"),
    "FlextInfraCommandRunner": ("flext_infra.subprocess", "FlextInfraCommandRunner"),
    "FlextInfraConstants": ("flext_infra.constants", "FlextInfraConstants"),

    "FlextInfraGitService": ("flext_infra.git", "FlextInfraGitService"),
    "FlextInfraJsonService": ("flext_infra.json_io", "FlextInfraJsonService"),
    "FlextInfraModels": ("flext_infra.models", "FlextInfraModels"),
    "FlextInfraPathResolver": ("flext_infra.paths", "FlextInfraPathResolver"),
    "FlextInfraPatterns": ("flext_infra.patterns", "FlextInfraPatterns"),
    "FlextInfraProjectSelector": ("flext_infra.selection", "FlextInfraProjectSelector"),
    "FlextInfraProtocols": ("flext_infra.protocols", "FlextInfraProtocols"),
    "FlextInfraReleaseOrchestrator": ("flext_infra.release", "ReleaseOrchestrator"),
    "FlextInfraReportingService": ("flext_infra.reporting", "FlextInfraReportingService"),
    "FlextInfraTemplateEngine": ("flext_infra.templates", "FlextInfraTemplateEngine"),
    "FlextInfraTomlService": ("flext_infra.toml_io", "FlextInfraTomlService"),
    "FlextInfraTypes": ("flext_infra.typings", "FlextInfraTypes"),
    "FlextInfraUtilities": ("flext_infra.utilities", "FlextInfraUtilities"),
    "FlextInfraVersioningService": ("flext_infra.versioning", "FlextInfraVersioningService"),
    "KNOWN_VERBS": ("flext_infra.reporting", "KNOWN_VERBS"),

    "REPORTS_DIR_NAME": ("flext_infra.reporting", "REPORTS_DIR_NAME"),
    "ReleaseOrchestrator": ("flext_infra.release", "ReleaseOrchestrator"),

    "FlextInfraOutput": ("flext_infra.output", "FlextInfraOutput"),

    "__version__": ("flext_infra.__version__", "__version__"),
    "__version_info__": ("flext_infra.__version__", "__version_info__"),
    "c": ("flext_infra.constants", "FlextInfraConstants"),
    "m": ("flext_infra.models", "FlextInfraModels"),
    "p": ("flext_infra.protocols", "FlextInfraProtocols"),
    "t": ("flext_infra.typings", "FlextInfraTypes"),
    "u": ("flext_infra.utilities", "FlextInfraUtilities"),
    "output": ("flext_infra.output", "output"),
    # Short aliases for internal consumers (backward compat)
    "CommandRunner": ("flext_infra.subprocess", "FlextInfraCommandRunner"),
    "DiscoveryService": ("flext_infra.discovery", "FlextInfraDiscoveryService"),
    "GitService": ("flext_infra.git", "FlextInfraGitService"),
    "JsonService": ("flext_infra.json_io", "FlextInfraJsonService"),
    "PathResolver": ("flext_infra.paths", "FlextInfraPathResolver"),
    "ProjectSelector": ("flext_infra.selection", "FlextInfraProjectSelector"),
    "ReportingService": ("flext_infra.reporting", "FlextInfraReportingService"),
    "TemplateEngine": ("flext_infra.templates", "FlextInfraTemplateEngine"),
    "TomlService": ("flext_infra.toml_io", "FlextInfraTomlService"),
    "VersioningService": ("flext_infra.versioning", "FlextInfraVersioningService"),
}

__all__ = [
    "KNOWN_VERBS",
    "REPORTS_DIR_NAME",
    "CommandRunner",
    "DiscoveryService",
    "FlextInfraBaseMkGenerator",
    "FlextInfraBaseMkTemplateEngine",
    "FlextInfraCheckConfigFixer",
    "FlextInfraCheckWorkspaceChecker",
    "FlextInfraCommandRunner",
    "FlextInfraConstants",
    "FlextInfraDiscoveryService",
    "FlextInfraGitService",
    "FlextInfraJsonService",
    "FlextInfraModels",
    "FlextInfraOutput",
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
    "TomlService",
    "VersioningService",
    "__version__",
    "__version_info__",
    "c",
    "m",
    "output",
    "p",
    "t",
    "u",
]


def __getattr__(name: str) -> Any:  # noqa: ANN401
    """Lazy-load module attributes on first access (PEP 562)."""
    return lazy_getattr(name, _LAZY_IMPORTS, globals(), __name__)


cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)
