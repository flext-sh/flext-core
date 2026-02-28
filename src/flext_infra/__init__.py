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
    from flext_infra.codegen import FlextInfraLazyInitGenerator
    from flext_infra.constants import FlextInfraConstants, FlextInfraConstants as c
    from flext_infra.discovery import FlextInfraDiscoveryService
    from flext_infra.git import FlextInfraGitService
    from flext_infra.json_io import FlextInfraJsonService
    from flext_infra.models import FlextInfraModels, FlextInfraModels as m
    from flext_infra.output import FlextInfraOutput
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
    from flext_infra.github import (
        FlextInfraPrManager,
        FlextInfraPrWorkspaceManager,
        FlextInfraWorkflowLinter,
        FlextInfraWorkflowSyncer,
        SyncOperation,
    )

# Lazy import mapping: export_name -> (module_path, attr_name)
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "FlextInfraBaseMkGenerator": ("flext_infra.basemk", "FlextInfraBaseMkGenerator"),
    "FlextInfraBaseMkTemplateEngine": ("flext_infra.basemk", "FlextInfraBaseMkTemplateEngine"),
    "FlextInfraCheckConfigFixer": ("flext_infra.check.services", "FlextInfraConfigFixer"),
    "FlextInfraCheckWorkspaceChecker": ("flext_infra.check.services", "FlextInfraWorkspaceChecker"),
    "FlextInfraBaseMkValidator": (
        "flext_infra.core.basemk_validator",
        "FlextInfraBaseMkValidator",
    ),
    "FlextInfraInventoryService": (
        "flext_infra.core.inventory",
        "FlextInfraInventoryService",
    ),
    "FlextInfraPytestDiagExtractor": (
        "flext_infra.core.pytest_diag",
        "FlextInfraPytestDiagExtractor",
    ),
    "FlextInfraSkillValidator": (
        "flext_infra.core.skill_validator",
        "FlextInfraSkillValidator",
    ),
    "FlextInfraStubSupplyChain": (
        "flext_infra.core.stub_chain",
        "FlextInfraStubSupplyChain",
    ),
    "FlextInfraTextPatternScanner": (
        "flext_infra.core.scanner",
        "FlextInfraTextPatternScanner",
    ),

    "FlextInfraCommandRunner": ("flext_infra.subprocess", "FlextInfraCommandRunner"),
    "FlextInfraConstants": ("flext_infra.constants", "FlextInfraConstants"),
    "FlextInfraDiscoveryService": ("flext_infra.discovery", "FlextInfraDiscoveryService"),
    "FlextInfraGitService": ("flext_infra.git", "FlextInfraGitService"),
    "FlextInfraJsonService": ("flext_infra.json_io", "FlextInfraJsonService"),
    "FlextInfraLazyInitGenerator": ("flext_infra.codegen", "FlextInfraLazyInitGenerator"),
    "FlextInfraModels": ("flext_infra.models", "FlextInfraModels"),
    "FlextInfraOutput": ("flext_infra.output", "FlextInfraOutput"),
    "FlextInfraPathResolver": ("flext_infra.paths", "FlextInfraPathResolver"),
    "FlextInfraPatterns": ("flext_infra.patterns", "FlextInfraPatterns"),
    "FlextInfraProjectSelector": ("flext_infra.selection", "FlextInfraProjectSelector"),
    "FlextInfraProtocols": ("flext_infra.protocols", "FlextInfraProtocols"),
    "FlextInfraReleaseOrchestrator": ("flext_infra.release", "FlextInfraReleaseOrchestrator"),
    "FlextInfraReportingService": ("flext_infra.reporting", "FlextInfraReportingService"),
    "FlextInfraTemplateEngine": ("flext_infra.templates", "FlextInfraTemplateEngine"),
    "FlextInfraTomlService": ("flext_infra.toml_io", "FlextInfraTomlService"),
    "FlextInfraTypes": ("flext_infra.typings", "FlextInfraTypes"),
    "FlextInfraUtilities": ("flext_infra.utilities", "FlextInfraUtilities"),
    "FlextInfraVersioningService": ("flext_infra.versioning", "FlextInfraVersioningService"),
    "FlextInfraPrManager": ("flext_infra.github", "FlextInfraPrManager"),
    "FlextInfraPrWorkspaceManager": ("flext_infra.github", "FlextInfraPrWorkspaceManager"),
    "FlextInfraWorkflowLinter": ("flext_infra.github", "FlextInfraWorkflowLinter"),
    "FlextInfraWorkflowSyncer": ("flext_infra.github", "FlextInfraWorkflowSyncer"),
    "SyncOperation": ("flext_infra.github", "SyncOperation"),
    "KNOWN_VERBS": ("flext_infra.reporting", "KNOWN_VERBS"),
    "REPORTS_DIR_NAME": ("flext_infra.reporting", "REPORTS_DIR_NAME"),
    "__version__": ("flext_infra.__version__", "__version__"),
    "__version_info__": ("flext_infra.__version__", "__version_info__"),
    "c": ("flext_infra.constants", "FlextInfraConstants"),
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
    "FlextInfraCheckConfigFixer",
    "FlextInfraCheckWorkspaceChecker",
    "FlextInfraBaseMkValidator",
    "FlextInfraInventoryService",
    "FlextInfraPytestDiagExtractor",
    "FlextInfraSkillValidator",
    "FlextInfraStubSupplyChain",
    "FlextInfraTextPatternScanner",

    "FlextInfraCommandRunner",
    "FlextInfraConstants",
    "FlextInfraDiscoveryService",
    "FlextInfraGitService",
    "FlextInfraJsonService",
    "FlextInfraLazyInitGenerator",
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
    "FlextInfraPrManager",
    "FlextInfraPrWorkspaceManager",
    "FlextInfraWorkflowLinter",
    "FlextInfraWorkflowSyncer",
    "SyncOperation",
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


def __dir__() -> list[str]:
    """Return list of available attributes for dir() and autocomplete."""
    return sorted(__all__)


cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)
