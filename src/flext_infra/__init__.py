"""Public API for flext-infra.

Provides access to infrastructure services for workspace management, validation,
dependency handling, and build orchestration in the FLEXT ecosystem.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from flext_core.lazy import cleanup_submodule_namespace, lazy_getattr

if TYPE_CHECKING:
    from flext_infra.__version__ import __version__, __version_info__
    from flext_infra.basemk import (
        FlextInfraBaseMkGenerator,
        FlextInfraBaseMkTemplateEngine,
    )
    from flext_infra.check.services import (
        FlextInfraConfigFixer as FlextInfraCheckConfigFixer,
        FlextInfraWorkspaceChecker as FlextInfraCheckWorkspaceChecker,
    )
    from flext_infra.codegen import FlextInfraCodegenLazyInit
    from flext_infra.constants import FlextInfraConstants, FlextInfraConstants as c
    from flext_infra.container import (
        configure_flext_infra_dependencies,
        get_flext_infra_container,
        get_flext_infra_service,
    )
    from flext_infra.core.basemk_validator import FlextInfraBaseMkValidator
    from flext_infra.core.inventory import FlextInfraInventoryService
    from flext_infra.core.pytest_diag import FlextInfraPytestDiagExtractor
    from flext_infra.core.scanner import FlextInfraTextPatternScanner
    from flext_infra.core.skill_validator import FlextInfraSkillValidator
    from flext_infra.core.stub_chain import FlextInfraStubSupplyChain
    from flext_infra.discovery import FlextInfraDiscoveryService
    from flext_infra.dispatcher import FlextInfraDispatcher
    from flext_infra.git import FlextInfraGitService
    from flext_infra.github import (
        FlextInfraPrManager,
        FlextInfraPrWorkspaceManager,
        FlextInfraWorkflowLinter,
        FlextInfraWorkflowSyncer,
        SyncOperation,
    )
    from flext_infra.json_io import FlextInfraJsonService
    from flext_infra.maintenance import FlextInfraPythonVersionEnforcer
    from flext_infra.models import FlextInfraModels, FlextInfraModels as m
    from flext_infra.output import FlextInfraOutput, output
    from flext_infra.paths import FlextInfraPathResolver
    from flext_infra.patterns import FlextInfraPatterns
    from flext_infra.protocols import FlextInfraProtocols, FlextInfraProtocols as p
    from flext_infra.release import FlextInfraReleaseOrchestrator
    from flext_infra.reporting import (
        KNOWN_VERBS,
        REPORTS_DIR_NAME,
        FlextInfraReportingService,
    )
    from flext_infra.selection import FlextInfraProjectSelector
    from flext_infra.subprocess import FlextInfraCommandRunner
    from flext_infra.templates import FlextInfraTemplateEngine
    from flext_infra.toml_io import FlextInfraTomlService
    from flext_infra.typings import FlextInfraTypes, FlextInfraTypes as t
    from flext_infra.utilities import FlextInfraUtilities, FlextInfraUtilities as u
    from flext_infra.versioning import FlextInfraVersioningService

# Lazy import mapping: export_name -> (module_path, attr_name)
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "FlextInfraBaseMkGenerator": ("flext_infra.basemk", "FlextInfraBaseMkGenerator"),
    "FlextInfraBaseMkTemplateEngine": (
        "flext_infra.basemk",
        "FlextInfraBaseMkTemplateEngine",
    ),
    "FlextInfraBaseMkValidator": (
        "flext_infra.core.basemk_validator",
        "FlextInfraBaseMkValidator",
    ),
    "FlextInfraCheckConfigFixer": (
        "flext_infra.check.services",
        "FlextInfraConfigFixer",
    ),
    "FlextInfraCheckWorkspaceChecker": (
        "flext_infra.check.services",
        "FlextInfraWorkspaceChecker",
    ),
    "FlextInfraCommandRunner": ("flext_infra.subprocess", "FlextInfraCommandRunner"),
    "FlextInfraConstants": ("flext_infra.constants", "FlextInfraConstants"),
    "FlextInfraDiscoveryService": (
        "flext_infra.discovery",
        "FlextInfraDiscoveryService",
    ),
    "FlextInfraDispatcher": ("flext_infra.dispatcher", "FlextInfraDispatcher"),
    "FlextInfraGitService": ("flext_infra.git", "FlextInfraGitService"),
    "FlextInfraInventoryService": (
        "flext_infra.core.inventory",
        "FlextInfraInventoryService",
    ),
    "FlextInfraJsonService": ("flext_infra.json_io", "FlextInfraJsonService"),
    "FlextInfraCodegenLazyInit": (
        "flext_infra.codegen",
        "FlextInfraCodegenLazyInit",
    ),
    "FlextInfraModels": ("flext_infra.models", "FlextInfraModels"),
    "FlextInfraOutput": ("flext_infra.output", "FlextInfraOutput"),
    "FlextInfraPathResolver": ("flext_infra.paths", "FlextInfraPathResolver"),
    "FlextInfraPatterns": ("flext_infra.patterns", "FlextInfraPatterns"),
    "FlextInfraPrManager": ("flext_infra.github", "FlextInfraPrManager"),
    "FlextInfraPrWorkspaceManager": (
        "flext_infra.github",
        "FlextInfraPrWorkspaceManager",
    ),
    "FlextInfraProjectSelector": ("flext_infra.selection", "FlextInfraProjectSelector"),
    "FlextInfraProtocols": ("flext_infra.protocols", "FlextInfraProtocols"),
    "FlextInfraPytestDiagExtractor": (
        "flext_infra.core.pytest_diag",
        "FlextInfraPytestDiagExtractor",
    ),
    "FlextInfraPythonVersionEnforcer": (
        "flext_infra.maintenance",
        "FlextInfraPythonVersionEnforcer",
    ),
    "FlextInfraReleaseOrchestrator": (
        "flext_infra.release",
        "FlextInfraReleaseOrchestrator",
    ),
    "FlextInfraReportingService": (
        "flext_infra.reporting",
        "FlextInfraReportingService",
    ),
    "FlextInfraSkillValidator": (
        "flext_infra.core.skill_validator",
        "FlextInfraSkillValidator",
    ),
    "FlextInfraStubSupplyChain": (
        "flext_infra.core.stub_chain",
        "FlextInfraStubSupplyChain",
    ),
    "FlextInfraTemplateEngine": ("flext_infra.templates", "FlextInfraTemplateEngine"),
    "FlextInfraTextPatternScanner": (
        "flext_infra.core.scanner",
        "FlextInfraTextPatternScanner",
    ),
    "FlextInfraTomlService": ("flext_infra.toml_io", "FlextInfraTomlService"),
    "FlextInfraTypes": ("flext_infra.typings", "FlextInfraTypes"),
    "FlextInfraUtilities": ("flext_infra.utilities", "FlextInfraUtilities"),
    "FlextInfraVersioningService": (
        "flext_infra.versioning",
        "FlextInfraVersioningService",
    ),
    "FlextInfraWorkflowLinter": ("flext_infra.github", "FlextInfraWorkflowLinter"),
    "FlextInfraWorkflowSyncer": ("flext_infra.github", "FlextInfraWorkflowSyncer"),
    "KNOWN_VERBS": ("flext_infra.reporting", "KNOWN_VERBS"),
    "REPORTS_DIR_NAME": ("flext_infra.reporting", "REPORTS_DIR_NAME"),
    "SyncOperation": ("flext_infra.github", "SyncOperation"),
    "__version__": ("flext_infra.__version__", "__version__"),
    "__version_info__": ("flext_infra.__version__", "__version_info__"),
    "c": ("flext_infra.constants", "FlextInfraConstants"),
    "configure_flext_infra_dependencies": (
        "flext_infra.container",
        "configure_flext_infra_dependencies",
    ),
    "get_flext_infra_container": ("flext_infra.container", "get_flext_infra_container"),
    "get_flext_infra_service": ("flext_infra.container", "get_flext_infra_service"),
    "m": ("flext_infra.models", "FlextInfraModels"),
    "output": ("flext_infra.output", "output"),
    "p": ("flext_infra.protocols", "FlextInfraProtocols"),
    "t": ("flext_infra.typings", "FlextInfraTypes"),
    "u": ("flext_infra.utilities", "FlextInfraUtilities"),
}

__all__ = [
    "KNOWN_VERBS",
    "REPORTS_DIR_NAME",
    "FlextInfraBaseMkGenerator",
    "FlextInfraBaseMkTemplateEngine",
    "FlextInfraBaseMkValidator",
    "FlextInfraCheckConfigFixer",
    "FlextInfraCheckWorkspaceChecker",
    "FlextInfraCodegenLazyInit",
    "FlextInfraCommandRunner",
    "FlextInfraConstants",
    "FlextInfraDiscoveryService",
    "FlextInfraDispatcher",
    "FlextInfraGitService",
    "FlextInfraInventoryService",
    "FlextInfraJsonService",
    "FlextInfraModels",
    "FlextInfraOutput",
    "FlextInfraPathResolver",
    "FlextInfraPatterns",
    "FlextInfraPrManager",
    "FlextInfraPrWorkspaceManager",
    "FlextInfraProjectSelector",
    "FlextInfraProtocols",
    "FlextInfraPytestDiagExtractor",
    "FlextInfraPythonVersionEnforcer",
    "FlextInfraReleaseOrchestrator",
    "FlextInfraReportingService",
    "FlextInfraSkillValidator",
    "FlextInfraStubSupplyChain",
    "FlextInfraTemplateEngine",
    "FlextInfraTextPatternScanner",
    "FlextInfraTomlService",
    "FlextInfraTypes",
    "FlextInfraUtilities",
    "FlextInfraVersioningService",
    "FlextInfraWorkflowLinter",
    "FlextInfraWorkflowSyncer",
    "SyncOperation",
    "__version__",
    "__version_info__",
    "c",
    "configure_flext_infra_dependencies",
    "get_flext_infra_container",
    "get_flext_infra_service",
    "m",
    "output",
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
