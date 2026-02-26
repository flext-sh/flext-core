"""Public API for flext-infra.

Provides access to infrastructure services for workspace management, validation,
dependency handling, and build orchestration in the FLEXT ecosystem.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Type hints only - not loaded at runtime
    from flext_infra.__version__ import __version__, __version_info__
    from flext_infra.basemk import (
        BaseMkGenerator,
        TemplateEngine as BaseMkTemplateEngine,
    )
    from flext_infra.constants import FlextInfraConstants as c
    from flext_infra.discovery import DiscoveryService
    from flext_infra.git import GitService
    from flext_infra.json_io import JsonService
    from flext_infra.models import FlextInfraModels as m
    from flext_infra.paths import PathResolver
    from flext_infra.patterns import FlextInfraPatterns
    from flext_infra.protocols import FlextInfraProtocols as p
    from flext_infra.release import ReleaseOrchestrator
    from flext_infra.reporting import KNOWN_VERBS, REPORTS_DIR_NAME, ReportingService
    from flext_infra.selection import ProjectSelector
    from flext_infra.subprocess import CommandRunner
    from flext_infra.templates import TemplateEngine
    from flext_infra.toml_io import TomlService
    from flext_infra.typings import FlextInfraTypes as t
    from flext_infra.utilities import FlextInfraUtilities as u
    from flext_infra.versioning import VersioningService

# Lazy import mapping: export_name -> (module_path, attr_name)
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # Version info
    "__version__": ("flext_infra.__version__", "__version__"),
    "__version_info__": ("flext_infra.__version__", "__version_info__"),
    # Facade classes and aliases
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
    # Aliases
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


def __getattr__(name: str) -> object:
    """Lazy-load module attributes on first access (PEP 562).
    
    This defers all imports until actually needed, reducing startup time
    from ~1.2s to <50ms for bare `import flext_infra`.
    
    Handles submodule namespace pollution: when a submodule like
    flext_infra.__version__ is imported, Python adds it to the parent
    module's namespace. We need to check _LAZY_IMPORTS first to ensure
    we return the attribute, not the submodule.
    """
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path)
        value = getattr(module, attr_name)
        # Cache in globals() to avoid repeated lookups
        globals()[name] = value
        return value
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    """Return list of available attributes for dir() and autocomplete."""
    return sorted(__all__)


# Clean up submodule namespace pollution
# When submodules like flext_infra.__version__ are imported, Python adds them
# to the parent module's namespace. We remove them to force __getattr__ usage.
def _cleanup_submodule_namespace() -> None:
    """Remove submodules from namespace to force __getattr__ usage."""
    import sys
    
    # Get the current module
    current_module = sys.modules[__name__]
    
    # List of submodule names that might pollute the namespace
    submodule_names = [
        "__version__",  # flext_infra.__version__
        "basemk",
        "constants",
        "discovery",
        "git",
        "json_io",
        "models",
        "paths",
        "patterns",
        "protocols",
        "release",
        "reporting",
        "selection",
        "subprocess",
        "templates",
        "toml_io",
        "typings",
        "utilities",
        "versioning",
    ]
    
    # Remove submodules from the module's namespace
    for submodule_name in submodule_names:
        if hasattr(current_module, submodule_name):
            attr = getattr(current_module, submodule_name)
            # Only remove if it's a module (not our lazy-loaded values)
            if isinstance(attr, type(sys)):
                delattr(current_module, submodule_name)


_cleanup_submodule_namespace()
