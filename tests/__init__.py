# AUTO-GENERATED FILE — Regenerate with: make gen
"""Tests package."""

from __future__ import annotations

import typing as _t

from flext_core.lazy import (
    build_lazy_import_map,
    install_lazy_exports,
    merge_lazy_imports,
)

if _t.TYPE_CHECKING:
    from flext_tests import d, e, h, r, s, td, tf, tk, tm, tv, x

    from tests._constants.domain import TestsFlextCoreConstantsDomain
    from tests._constants.errors import TestsFlextCoreConstantsErrors
    from tests._constants.fixtures import TestsFlextCoreConstantsFixtures
    from tests._constants.loggings import TestsFlextCoreConstantsLoggings
    from tests._constants.other import TestsFlextCoreConstantsOther
    from tests._constants.result import TestsFlextCoreConstantsResult
    from tests._constants.services import TestsFlextCoreConstantsServices
    from tests._constants.settings import TestsFlextCoreConstantsSettings
    from tests._constants.strings import TestsFlextCoreConstantsStrings
    from tests._models.mixins import TestsFlextCoreModelsMixins
    from tests.benchmark.test_container_memory import TestContainerMemory
    from tests.benchmark.test_container_performance import TestContainerPerformance
    from tests.benchmark.test_lazy_performance import TestLazyPerformance
    from tests.constants import TestsFlextCoreConstants, c
    from tests.integration.patterns.test_advanced_patterns import TestAdvancedPatterns
    from tests.integration.patterns.test_architectural_patterns import (
        TestArchitecturalPatterns,
    )
    from tests.integration.patterns.test_patterns_commands import (
        TestsFlextCorePatternsCommands,
    )
    from tests.integration.patterns.test_patterns_logging import TestPatternsLogging
    from tests.integration.patterns.test_patterns_testing import TestPatternsTesting
    from tests.integration.test_architecture import TestAutomatedArchitecture
    from tests.integration.test_documented_patterns import TestDocumentedPatterns
    from tests.integration.test_examples_execution import TestExamplesExecution
    from tests.integration.test_integration import TestLibraryIntegration
    from tests.integration.test_migration_validation import TestMigrationValidation
    from tests.integration.test_service import TestsFlextCoreServiceIntegration
    from tests.integration.test_service_result_property import TestServiceResultProperty
    from tests.integration.test_settings_integration import (
        TestFlextSettingsSingletonIntegration,
    )
    from tests.integration.test_system import TestCompleteFlextSystemIntegration
    from tests.models import TestsFlextCoreModels, m
    from tests.protocols import TestsFlextCoreProtocols, p
    from tests.typings import T, T_co, T_contra, TestsFlextCoreTypes, t
    from tests.utilities import TestsFlextCoreUtilities, u
_LAZY_IMPORTS = merge_lazy_imports(
    (
        "._constants",
        "._models",
        ".benchmark",
        ".integration",
        ".unit",
    ),
    build_lazy_import_map(
        {
            "._constants.domain": ("TestsFlextCoreConstantsDomain",),
            "._constants.errors": ("TestsFlextCoreConstantsErrors",),
            "._constants.fixtures": ("TestsFlextCoreConstantsFixtures",),
            "._constants.loggings": ("TestsFlextCoreConstantsLoggings",),
            "._constants.other": ("TestsFlextCoreConstantsOther",),
            "._constants.result": ("TestsFlextCoreConstantsResult",),
            "._constants.services": ("TestsFlextCoreConstantsServices",),
            "._constants.settings": ("TestsFlextCoreConstantsSettings",),
            "._constants.strings": ("TestsFlextCoreConstantsStrings",),
            "._models.mixins": ("TestsFlextCoreModelsMixins",),
            ".benchmark.test_container_memory": ("TestContainerMemory",),
            ".benchmark.test_container_performance": ("TestContainerPerformance",),
            ".benchmark.test_lazy_performance": ("TestLazyPerformance",),
            ".constants": (
                "TestsFlextCoreConstants",
                "c",
            ),
            ".integration.patterns.test_advanced_patterns": ("TestAdvancedPatterns",),
            ".integration.patterns.test_architectural_patterns": (
                "TestArchitecturalPatterns",
            ),
            ".integration.patterns.test_patterns_commands": (
                "TestsFlextCorePatternsCommands",
            ),
            ".integration.patterns.test_patterns_logging": ("TestPatternsLogging",),
            ".integration.patterns.test_patterns_testing": ("TestPatternsTesting",),
            ".integration.test_architecture": ("TestAutomatedArchitecture",),
            ".integration.test_documented_patterns": ("TestDocumentedPatterns",),
            ".integration.test_examples_execution": ("TestExamplesExecution",),
            ".integration.test_integration": ("TestLibraryIntegration",),
            ".integration.test_migration_validation": ("TestMigrationValidation",),
            ".integration.test_service": ("TestsFlextCoreServiceIntegration",),
            ".integration.test_service_result_property": ("TestServiceResultProperty",),
            ".integration.test_settings_integration": (
                "TestFlextSettingsSingletonIntegration",
            ),
            ".integration.test_system": ("TestCompleteFlextSystemIntegration",),
            ".models": (
                "TestsFlextCoreModels",
                "m",
            ),
            ".protocols": (
                "TestsFlextCoreProtocols",
                "p",
            ),
            ".typings": (
                "T",
                "T_co",
                "T_contra",
                "TestsFlextCoreTypes",
                "t",
            ),
            ".utilities": (
                "TestsFlextCoreUtilities",
                "u",
            ),
            "flext_tests": (
                "d",
                "e",
                "h",
                "r",
                "s",
                "td",
                "tf",
                "tk",
                "tm",
                "tv",
                "x",
            ),
        },
    ),
    exclude_names=(
        "cleanup_submodule_namespace",
        "install_lazy_exports",
        "lazy_getattr",
        "logger",
        "merge_lazy_imports",
        "output",
        "output_reporting",
    ),
    module_name=__name__,
)


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS)

__all__: list[str] = [
    "T",
    "T_co",
    "T_contra",
    "TestAdvancedPatterns",
    "TestArchitecturalPatterns",
    "TestAutomatedArchitecture",
    "TestCompleteFlextSystemIntegration",
    "TestContainerMemory",
    "TestContainerPerformance",
    "TestDocumentedPatterns",
    "TestExamplesExecution",
    "TestFlextSettingsSingletonIntegration",
    "TestLazyPerformance",
    "TestLibraryIntegration",
    "TestMigrationValidation",
    "TestPatternsLogging",
    "TestPatternsTesting",
    "TestServiceResultProperty",
    "TestsFlextCoreConstants",
    "TestsFlextCoreConstantsDomain",
    "TestsFlextCoreConstantsErrors",
    "TestsFlextCoreConstantsFixtures",
    "TestsFlextCoreConstantsLoggings",
    "TestsFlextCoreConstantsOther",
    "TestsFlextCoreConstantsResult",
    "TestsFlextCoreConstantsServices",
    "TestsFlextCoreConstantsSettings",
    "TestsFlextCoreConstantsStrings",
    "TestsFlextCoreModels",
    "TestsFlextCoreModelsMixins",
    "TestsFlextCorePatternsCommands",
    "TestsFlextCoreProtocols",
    "TestsFlextCoreServiceIntegration",
    "TestsFlextCoreTypes",
    "TestsFlextCoreUtilities",
    "c",
    "d",
    "e",
    "h",
    "m",
    "p",
    "r",
    "s",
    "t",
    "td",
    "tf",
    "tk",
    "tm",
    "tv",
    "u",
    "x",
]
