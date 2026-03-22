# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make codegen
#
"""Integration package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import cleanup_submodule_namespace, lazy_getattr

if TYPE_CHECKING:
    from flext_core import FlextTypes

    from . import patterns as patterns
    from .patterns.test_advanced_patterns import TestAdvancedPatterns, TestFunction
    from .patterns.test_architectural_patterns import TestArchitecturalPatterns
    from .patterns.test_patterns_commands import TestPatternsCommands
    from .patterns.test_patterns_logging import EXPECTED_BULK_SIZE, TestPatternsLogging
    from .patterns.test_patterns_testing import TestPatternsTesting
    from .test_config_integration import TestFlextSettingsSingletonIntegration
    from .test_infra_integration import TestInfraIntegration
    from .test_integration import TestLibraryIntegration
    from .test_migration_validation import TestMigrationValidation
    from .test_refactor_nesting_file import (
        pytestmark,
        test_class_nesting_refactor_single_file_end_to_end,
    )
    from .test_refactor_nesting_idempotency import TestIdempotency
    from .test_refactor_nesting_project import TestProjectLevelRefactor
    from .test_refactor_nesting_workspace import TestWorkspaceLevelRefactor
    from .test_refactor_policy_mro import TestRefactorPolicyMRO
    from .test_service import TestService
    from .test_system import TestCompleteFlextSystemIntegration

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "EXPECTED_BULK_SIZE": (
        "tests.integration.patterns.test_patterns_logging",
        "EXPECTED_BULK_SIZE",
    ),
    "TestAdvancedPatterns": (
        "tests.integration.patterns.test_advanced_patterns",
        "TestAdvancedPatterns",
    ),
    "TestArchitecturalPatterns": (
        "tests.integration.patterns.test_architectural_patterns",
        "TestArchitecturalPatterns",
    ),
    "TestCompleteFlextSystemIntegration": (
        "tests.integration.test_system",
        "TestCompleteFlextSystemIntegration",
    ),
    "TestFlextSettingsSingletonIntegration": (
        "tests.integration.test_config_integration",
        "TestFlextSettingsSingletonIntegration",
    ),
    "TestFunction": (
        "tests.integration.patterns.test_advanced_patterns",
        "TestFunction",
    ),
    "TestIdempotency": (
        "tests.integration.test_refactor_nesting_idempotency",
        "TestIdempotency",
    ),
    "TestInfraIntegration": (
        "tests.integration.test_infra_integration",
        "TestInfraIntegration",
    ),
    "TestLibraryIntegration": (
        "tests.integration.test_integration",
        "TestLibraryIntegration",
    ),
    "TestMigrationValidation": (
        "tests.integration.test_migration_validation",
        "TestMigrationValidation",
    ),
    "TestPatternsCommands": (
        "tests.integration.patterns.test_patterns_commands",
        "TestPatternsCommands",
    ),
    "TestPatternsLogging": (
        "tests.integration.patterns.test_patterns_logging",
        "TestPatternsLogging",
    ),
    "TestPatternsTesting": (
        "tests.integration.patterns.test_patterns_testing",
        "TestPatternsTesting",
    ),
    "TestProjectLevelRefactor": (
        "tests.integration.test_refactor_nesting_project",
        "TestProjectLevelRefactor",
    ),
    "TestRefactorPolicyMRO": (
        "tests.integration.test_refactor_policy_mro",
        "TestRefactorPolicyMRO",
    ),
    "TestService": ("tests.integration.test_service", "TestService"),
    "TestWorkspaceLevelRefactor": (
        "tests.integration.test_refactor_nesting_workspace",
        "TestWorkspaceLevelRefactor",
    ),
    "patterns": ("tests.integration.patterns", ""),
    "pytestmark": ("tests.integration.test_refactor_nesting_file", "pytestmark"),
    "test_class_nesting_refactor_single_file_end_to_end": (
        "tests.integration.test_refactor_nesting_file",
        "test_class_nesting_refactor_single_file_end_to_end",
    ),
}

__all__ = [
    "EXPECTED_BULK_SIZE",
    "TestAdvancedPatterns",
    "TestArchitecturalPatterns",
    "TestCompleteFlextSystemIntegration",
    "TestFlextSettingsSingletonIntegration",
    "TestFunction",
    "TestIdempotency",
    "TestInfraIntegration",
    "TestLibraryIntegration",
    "TestMigrationValidation",
    "TestPatternsCommands",
    "TestPatternsLogging",
    "TestPatternsTesting",
    "TestProjectLevelRefactor",
    "TestRefactorPolicyMRO",
    "TestService",
    "TestWorkspaceLevelRefactor",
    "patterns",
    "pytestmark",
    "test_class_nesting_refactor_single_file_end_to_end",
]


_LAZY_CACHE: dict[str, FlextTypes.ModuleExport] = {}


def __getattr__(name: str) -> FlextTypes.ModuleExport:
    """Lazy-load module attributes on first access (PEP 562).

    A local cache ``_LAZY_CACHE`` persists resolved objects across repeated
    accesses during process lifetime.

    Args:
        name: Attribute name requested by dir()/import.

    Returns:
        Lazy-loaded module export type.

    Raises:
        AttributeError: If attribute not registered.

    """
    if name in _LAZY_CACHE:
        return _LAZY_CACHE[name]

    value = lazy_getattr(name, _LAZY_IMPORTS, globals(), __name__)
    _LAZY_CACHE[name] = value
    return value


def __dir__() -> list[str]:
    """Return list of available attributes for dir() and autocomplete.

    Returns:
        List of public names from module exports.

    """
    return sorted(__all__)


cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)
