# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make codegen
#
"""Utilities package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import cleanup_submodule_namespace, lazy_getattr

if TYPE_CHECKING:
    from flext_core.typings import FlextTypes
    from tests.infra.unit._utilities.test_discovery_consolidated import (
        TestDiscoveryDiscoverProjects,
        TestDiscoveryFindAllPyprojectFiles,
        TestDiscoveryIterPythonFiles,
        TestDiscoveryProjectRoots,
    )
    from tests.infra.unit._utilities.test_formatting import TestFormattingRunRuffFix
    from tests.infra.unit._utilities.test_iteration import (
        TestIterWorkspacePythonModules,
    )
    from tests.infra.unit._utilities.test_parsing import (
        TestParsingModuleAst,
        TestParsingModuleCst,
    )
    from tests.infra.unit._utilities.test_safety import (
        TestSafetyCheckpoint,
        TestSafetyRollback,
        TestSafetyWorkspaceValidation,
    )
    from tests.infra.unit._utilities.test_scanning import (
        MockScanner,
        TestScanFileBatch,
        TestScanModels,
        TestScanModels as m,
    )

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "MockScanner": ("tests.infra.unit._utilities.test_scanning", "MockScanner"),
    "TestDiscoveryDiscoverProjects": (
        "tests.infra.unit._utilities.test_discovery_consolidated",
        "TestDiscoveryDiscoverProjects",
    ),
    "TestDiscoveryFindAllPyprojectFiles": (
        "tests.infra.unit._utilities.test_discovery_consolidated",
        "TestDiscoveryFindAllPyprojectFiles",
    ),
    "TestDiscoveryIterPythonFiles": (
        "tests.infra.unit._utilities.test_discovery_consolidated",
        "TestDiscoveryIterPythonFiles",
    ),
    "TestDiscoveryProjectRoots": (
        "tests.infra.unit._utilities.test_discovery_consolidated",
        "TestDiscoveryProjectRoots",
    ),
    "TestFormattingRunRuffFix": (
        "tests.infra.unit._utilities.test_formatting",
        "TestFormattingRunRuffFix",
    ),
    "TestIterWorkspacePythonModules": (
        "tests.infra.unit._utilities.test_iteration",
        "TestIterWorkspacePythonModules",
    ),
    "TestParsingModuleAst": (
        "tests.infra.unit._utilities.test_parsing",
        "TestParsingModuleAst",
    ),
    "TestParsingModuleCst": (
        "tests.infra.unit._utilities.test_parsing",
        "TestParsingModuleCst",
    ),
    "TestSafetyCheckpoint": (
        "tests.infra.unit._utilities.test_safety",
        "TestSafetyCheckpoint",
    ),
    "TestSafetyRollback": (
        "tests.infra.unit._utilities.test_safety",
        "TestSafetyRollback",
    ),
    "TestSafetyWorkspaceValidation": (
        "tests.infra.unit._utilities.test_safety",
        "TestSafetyWorkspaceValidation",
    ),
    "TestScanFileBatch": (
        "tests.infra.unit._utilities.test_scanning",
        "TestScanFileBatch",
    ),
    "TestScanModels": ("tests.infra.unit._utilities.test_scanning", "TestScanModels"),
    "m": ("tests.infra.unit._utilities.test_scanning", "TestScanModels"),
}

__all__ = [
    "MockScanner",
    "TestDiscoveryDiscoverProjects",
    "TestDiscoveryFindAllPyprojectFiles",
    "TestDiscoveryIterPythonFiles",
    "TestDiscoveryProjectRoots",
    "TestFormattingRunRuffFix",
    "TestIterWorkspacePythonModules",
    "TestParsingModuleAst",
    "TestParsingModuleCst",
    "TestSafetyCheckpoint",
    "TestSafetyRollback",
    "TestSafetyWorkspaceValidation",
    "TestScanFileBatch",
    "TestScanModels",
    "m",
]


def __getattr__(name: str) -> FlextTypes.ModuleExport:
    """Lazy-load module attributes on first access (PEP 562)."""
    return lazy_getattr(name, _LAZY_IMPORTS, globals(), __name__)


def __dir__() -> list[str]:
    """Return list of available attributes for dir() and autocomplete."""
    return sorted(__all__)


cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)
