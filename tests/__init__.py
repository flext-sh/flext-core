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

    from tests._constants.domain import TestsFlextConstantsDomain
    from tests._constants.errors import TestsFlextConstantsErrors
    from tests._constants.fixtures import TestsFlextConstantsFixtures
    from tests._constants.loggings import TestsFlextConstantsLoggings
    from tests._constants.other import TestsFlextConstantsOther
    from tests._constants.result import TestsFlextConstantsResult
    from tests._constants.services import TestsFlextConstantsServices
    from tests._constants.settings import TestsFlextConstantsSettings
    from tests._models.mixins import TestsFlextModelsMixins
    from tests.benchmark.test_container_memory import TestsFlextContainerMemory
    from tests.benchmark.test_container_performance import (
        TestsFlextContainerPerformance,
    )
    from tests.benchmark.test_lazy_performance import TestsFlextLazyPerformance
    from tests.constants import TestsFlextConstants, c
    from tests.integration.patterns.test_advanced_patterns import (
        TestsFlextAdvancedPatterns,
    )
    from tests.integration.patterns.test_architectural_patterns import (
        TestsFlextArchitecturalPatterns,
    )
    from tests.integration.patterns.test_patterns_commands import (
        TestsFlextPatternsCommands,
    )
    from tests.integration.patterns.test_patterns_logging import (
        TestsFlextPatternsLogging,
    )
    from tests.integration.patterns.test_patterns_testing import (
        TestsFlextPatternsTesting,
    )
    from tests.integration.test_architecture import TestsFlextAutomatedArchitecture
    from tests.integration.test_documented_patterns import TestsFlextDocumentedPatterns
    from tests.integration.test_examples_execution import TestsFlextExamplesExecution
    from tests.integration.test_integration import TestsFlextLibraryIntegration
    from tests.integration.test_migration_validation import (
        TestsFlextMigrationValidation,
    )
    from tests.integration.test_service import TestsFlextServiceIntegration
    from tests.integration.test_service_result_property import (
        TestsFlextServiceResultProperty,
    )
    from tests.integration.test_settings_integration import (
        TestsFlextSettingsIntegration,
    )
    from tests.integration.test_system import TestsFlextSystemIntegration
    from tests.models import TestsFlextModels, m
    from tests.protocols import TestsFlextProtocols, p
    from tests.typings import TestsFlextTypes, t
    from tests.unit._enforcement_integration_fixtures.bad_module import (
        TestsFlextBadAccessors,
        TestsFlextBadAnyField,
        TestsFlextBadBareCollection,
        TestsFlextBadConstants,
        TestsFlextBadFrozen,
        TestsFlextBadInlineUnion,
        TestsFlextBadMissingDesc,
        TestsFlextBadMutableDefault,
        TestsFlextBadWorkerSettings,
    )
    from tests.unit._enforcement_integration_fixtures.clean_module import (
        TestsFlextCleanConstants,
        TestsFlextCleanModels,
        TestsFlextCleanProtocols,
        TestsFlextCleanServiceBase,
    )
    from tests.unit._models.test_base import TestsFlextModelsBase
    from tests.unit._models.test_cqrs import TestsFlextModelsCQRS
    from tests.unit._models.test_enforcement_sources import (
        TestsFlextModelsEnforcementSources,
    )
    from tests.unit._models.test_entity import TestsFlextModelsEntity
    from tests.unit._models.test_exception_params import TestsFlextModelsExceptionParams
    from tests.unit._utilities.test_guards import TestsFlextUtilitiesGuards
    from tests.unit._utilities.test_mapper import TestsFlextUtilitiesMapper
    from tests.unit.base import TestsFlextServiceBase
    from tests.unit.test_beartype_engine import TestsFlextBeartypeEngine
    from tests.unit.test_constants_new import TestsFlextConstantsNew
    from tests.unit.test_constants_project_metadata import (
        TestsFlextConstantsProjectMetadata,
    )
    from tests.unit.test_container import TestsFlextContainer
    from tests.unit.test_context import TestsFlextContext
    from tests.unit.test_coverage_loggings import TestsFlextCoverageLoggings
    from tests.unit.test_decorators import TestsFlextDecoratorsLegacy
    from tests.unit.test_decorators_discovery_full_coverage import (
        TestsFlextDecoratorsDiscovery,
    )
    from tests.unit.test_decorators_full_coverage import TestsFlextDecorators
    from tests.unit.test_deprecation_warnings import TestsFlextDeprecationWarnings
    from tests.unit.test_dispatcher import TestsFlextDispatcher
    from tests.unit.test_enforcement import TestsFlextEnforcement
    from tests.unit.test_enforcement_apt_hooks import TestsFlextEnforcementAptHooks
    from tests.unit.test_enforcement_catalog import TestsFlextEnforcementCatalog
    from tests.unit.test_enforcement_integration import TestsFlextEnforcementIntegration
    from tests.unit.test_enum_utilities_coverage_100 import TestsFlextEnumUtilities
    from tests.unit.test_exceptions import (
        TestsFlextCoverageExceptions,
        TestsFlextExceptions,
    )
    from tests.unit.test_handler_decorator_discovery import (
        TestsFlextHandlerDecoratorDiscovery,
    )
    from tests.unit.test_handlers import TestsFlextFlextHandlers
    from tests.unit.test_lazy_exports import TestsFlextLazy
    from tests.unit.test_loggings_full_coverage import TestsFlextLoggings
    from tests.unit.test_mixins import TestsFlextMixins
    from tests.unit.test_models import TestsFlextModelsUnit
    from tests.unit.test_models_base_full_coverage import (
        TestsFlextModelsBaseFullCoverage,
    )
    from tests.unit.test_models_container import TestsFlextModelsContainer
    from tests.unit.test_models_cqrs_full_coverage import TestsFlextModelsCqrs
    from tests.unit.test_models_project_metadata import TestsFlextModelsProjectMetadata
    from tests.unit.test_project_metadata_facade_access import (
        TestsFlextFacadeFlatSsotAccess,
    )
    from tests.unit.test_registry import TestsFlextRegistry
    from tests.unit.test_result import (
        TestsFlextResult,
        TestsFlextResultExceptionCarrying,
    )
    from tests.unit.test_runtime import TestsFlextRuntime
    from tests.unit.test_service import TestsFlextService
    from tests.unit.test_service_bootstrap import TestsFlextServiceBootstrap
    from tests.unit.test_service_coverage_100 import TestsFlextService100Coverage
    from tests.unit.test_settings import TestsFlextSettings
    from tests.unit.test_typings_new import TestsFlextTypesUnit
    from tests.unit.test_utilities import TestsFlextUtilitiesSmoke
    from tests.unit.test_utilities_collection_coverage_100 import (
        TestsFlextUtilitiesCollection,
    )
    from tests.unit.test_utilities_coverage import TestsFlextUtilitiesCoverage
    from tests.unit.test_utilities_domain import TestsFlextUtilitiesDomain
    from tests.unit.test_utilities_generators_full_coverage import (
        TestsFlextUtilitiesGenerators,
    )
    from tests.unit.test_utilities_project_metadata import (
        TestsFlextUtilitiesProjectMetadata,
    )
    from tests.unit.test_utilities_pydantic_coverage_100 import (
        TestsFlextUtilitiesPydantic,
    )
    from tests.unit.test_utilities_reliability import TestsFlextUtilitiesReliability
    from tests.unit.test_utilities_runtime_violation_registry_coverage_100 import (
        TestsFlextRuntimeViolationRegistry,
    )
    from tests.unit.test_utilities_settings_coverage_100 import (
        TestsFlextUtilitiesSettings,
        TestsFlextUtilitiesSettingsEnvFile,
        TestsFlextUtilitiesSettingsRegisterFactory,
    )
    from tests.unit.test_utilities_text_full_coverage import TestsFlextUtilitiesText
    from tests.unit.test_utilities_type_guards_coverage_100 import (
        TestsFlextUtilitiesTypeGuards,
    )
    from tests.unit.test_version import TestsFlextVersion
    from tests.utilities import TestsFlextUtilities, u
_LAZY_IMPORTS = merge_lazy_imports(
    (
        "._constants",
        "._models",
        ".benchmark",
        ".fixtures",
        ".integration",
        ".unit",
    ),
    build_lazy_import_map(
        {
            "._constants.domain": ("TestsFlextConstantsDomain",),
            "._constants.errors": ("TestsFlextConstantsErrors",),
            "._constants.fixtures": ("TestsFlextConstantsFixtures",),
            "._constants.loggings": ("TestsFlextConstantsLoggings",),
            "._constants.other": ("TestsFlextConstantsOther",),
            "._constants.result": ("TestsFlextConstantsResult",),
            "._constants.services": ("TestsFlextConstantsServices",),
            "._constants.settings": ("TestsFlextConstantsSettings",),
            "._models.mixins": ("TestsFlextModelsMixins",),
            ".benchmark.test_container_memory": ("TestsFlextContainerMemory",),
            ".benchmark.test_container_performance": (
                "TestsFlextContainerPerformance",
            ),
            ".benchmark.test_lazy_performance": ("TestsFlextLazyPerformance",),
            ".constants": (
                "TestsFlextConstants",
                "c",
            ),
            ".integration.patterns.test_advanced_patterns": (
                "TestsFlextAdvancedPatterns",
            ),
            ".integration.patterns.test_architectural_patterns": (
                "TestsFlextArchitecturalPatterns",
            ),
            ".integration.patterns.test_patterns_commands": (
                "TestsFlextPatternsCommands",
            ),
            ".integration.patterns.test_patterns_logging": (
                "TestsFlextPatternsLogging",
            ),
            ".integration.patterns.test_patterns_testing": (
                "TestsFlextPatternsTesting",
            ),
            ".integration.test_architecture": ("TestsFlextAutomatedArchitecture",),
            ".integration.test_documented_patterns": ("TestsFlextDocumentedPatterns",),
            ".integration.test_examples_execution": ("TestsFlextExamplesExecution",),
            ".integration.test_integration": ("TestsFlextLibraryIntegration",),
            ".integration.test_migration_validation": (
                "TestsFlextMigrationValidation",
            ),
            ".integration.test_service": ("TestsFlextServiceIntegration",),
            ".integration.test_service_result_property": (
                "TestsFlextServiceResultProperty",
            ),
            ".integration.test_settings_integration": (
                "TestsFlextSettingsIntegration",
            ),
            ".integration.test_system": ("TestsFlextSystemIntegration",),
            ".models": (
                "TestsFlextModels",
                "m",
            ),
            ".protocols": (
                "TestsFlextProtocols",
                "p",
            ),
            ".typings": (
                "TestsFlextTypes",
                "t",
            ),
            ".unit._enforcement_integration_fixtures.bad_module": (
                "TestsFlextBadAccessors",
                "TestsFlextBadAnyField",
                "TestsFlextBadBareCollection",
                "TestsFlextBadConstants",
                "TestsFlextBadFrozen",
                "TestsFlextBadInlineUnion",
                "TestsFlextBadMissingDesc",
                "TestsFlextBadMutableDefault",
                "TestsFlextBadWorkerSettings",
            ),
            ".unit._enforcement_integration_fixtures.clean_module": (
                "TestsFlextCleanConstants",
                "TestsFlextCleanModels",
                "TestsFlextCleanProtocols",
                "TestsFlextCleanServiceBase",
            ),
            ".unit._models.test_base": ("TestsFlextModelsBase",),
            ".unit._models.test_cqrs": ("TestsFlextModelsCQRS",),
            ".unit._models.test_enforcement_sources": (
                "TestsFlextModelsEnforcementSources",
            ),
            ".unit._models.test_entity": ("TestsFlextModelsEntity",),
            ".unit._models.test_exception_params": ("TestsFlextModelsExceptionParams",),
            ".unit._utilities.test_guards": ("TestsFlextUtilitiesGuards",),
            ".unit._utilities.test_mapper": ("TestsFlextUtilitiesMapper",),
            ".unit.base": ("TestsFlextServiceBase",),
            ".unit.test_beartype_engine": ("TestsFlextBeartypeEngine",),
            ".unit.test_constants_new": ("TestsFlextConstantsNew",),
            ".unit.test_constants_project_metadata": (
                "TestsFlextConstantsProjectMetadata",
            ),
            ".unit.test_container": ("TestsFlextContainer",),
            ".unit.test_context": ("TestsFlextContext",),
            ".unit.test_coverage_loggings": ("TestsFlextCoverageLoggings",),
            ".unit.test_decorators": ("TestsFlextDecoratorsLegacy",),
            ".unit.test_decorators_discovery_full_coverage": (
                "TestsFlextDecoratorsDiscovery",
            ),
            ".unit.test_decorators_full_coverage": ("TestsFlextDecorators",),
            ".unit.test_deprecation_warnings": ("TestsFlextDeprecationWarnings",),
            ".unit.test_dispatcher": ("TestsFlextDispatcher",),
            ".unit.test_enforcement": ("TestsFlextEnforcement",),
            ".unit.test_enforcement_apt_hooks": ("TestsFlextEnforcementAptHooks",),
            ".unit.test_enforcement_catalog": ("TestsFlextEnforcementCatalog",),
            ".unit.test_enforcement_integration": ("TestsFlextEnforcementIntegration",),
            ".unit.test_enum_utilities_coverage_100": ("TestsFlextEnumUtilities",),
            ".unit.test_exceptions": (
                "TestsFlextCoverageExceptions",
                "TestsFlextExceptions",
            ),
            ".unit.test_handler_decorator_discovery": (
                "TestsFlextHandlerDecoratorDiscovery",
            ),
            ".unit.test_handlers": ("TestsFlextFlextHandlers",),
            ".unit.test_lazy_exports": ("TestsFlextLazy",),
            ".unit.test_loggings_full_coverage": ("TestsFlextLoggings",),
            ".unit.test_mixins": ("TestsFlextMixins",),
            ".unit.test_models": ("TestsFlextModelsUnit",),
            ".unit.test_models_base_full_coverage": (
                "TestsFlextModelsBaseFullCoverage",
            ),
            ".unit.test_models_container": ("TestsFlextModelsContainer",),
            ".unit.test_models_cqrs_full_coverage": ("TestsFlextModelsCqrs",),
            ".unit.test_models_project_metadata": ("TestsFlextModelsProjectMetadata",),
            ".unit.test_project_metadata_facade_access": (
                "TestsFlextFacadeFlatSsotAccess",
            ),
            ".unit.test_registry": ("TestsFlextRegistry",),
            ".unit.test_result": (
                "TestsFlextResult",
                "TestsFlextResultExceptionCarrying",
            ),
            ".unit.test_runtime": ("TestsFlextRuntime",),
            ".unit.test_service": ("TestsFlextService",),
            ".unit.test_service_bootstrap": ("TestsFlextServiceBootstrap",),
            ".unit.test_service_coverage_100": ("TestsFlextService100Coverage",),
            ".unit.test_settings": ("TestsFlextSettings",),
            ".unit.test_typings_new": ("TestsFlextTypesUnit",),
            ".unit.test_utilities": ("TestsFlextUtilitiesSmoke",),
            ".unit.test_utilities_collection_coverage_100": (
                "TestsFlextUtilitiesCollection",
            ),
            ".unit.test_utilities_coverage": ("TestsFlextUtilitiesCoverage",),
            ".unit.test_utilities_domain": ("TestsFlextUtilitiesDomain",),
            ".unit.test_utilities_generators_full_coverage": (
                "TestsFlextUtilitiesGenerators",
            ),
            ".unit.test_utilities_project_metadata": (
                "TestsFlextUtilitiesProjectMetadata",
            ),
            ".unit.test_utilities_pydantic_coverage_100": (
                "TestsFlextUtilitiesPydantic",
            ),
            ".unit.test_utilities_reliability": ("TestsFlextUtilitiesReliability",),
            ".unit.test_utilities_runtime_violation_registry_coverage_100": (
                "TestsFlextRuntimeViolationRegistry",
            ),
            ".unit.test_utilities_settings_coverage_100": (
                "TestsFlextUtilitiesSettings",
                "TestsFlextUtilitiesSettingsEnvFile",
                "TestsFlextUtilitiesSettingsRegisterFactory",
            ),
            ".unit.test_utilities_text_full_coverage": ("TestsFlextUtilitiesText",),
            ".unit.test_utilities_type_guards_coverage_100": (
                "TestsFlextUtilitiesTypeGuards",
            ),
            ".unit.test_version": ("TestsFlextVersion",),
            ".utilities": (
                "TestsFlextUtilities",
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
        "pytest_addoption",
        "pytest_collect_file",
        "pytest_collection_modifyitems",
        "pytest_configure",
        "pytest_runtest_setup",
        "pytest_runtest_teardown",
        "pytest_sessionfinish",
        "pytest_sessionstart",
        "pytest_terminal_summary",
        "pytest_warning_recorded",
    ),
    module_name=__name__,
)


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS)

__all__: list[str] = [
    "TestsFlextAdvancedPatterns",
    "TestsFlextArchitecturalPatterns",
    "TestsFlextAutomatedArchitecture",
    "TestsFlextBadAccessors",
    "TestsFlextBadAnyField",
    "TestsFlextBadBareCollection",
    "TestsFlextBadConstants",
    "TestsFlextBadFrozen",
    "TestsFlextBadInlineUnion",
    "TestsFlextBadMissingDesc",
    "TestsFlextBadMutableDefault",
    "TestsFlextBadWorkerSettings",
    "TestsFlextBeartypeEngine",
    "TestsFlextCleanConstants",
    "TestsFlextCleanModels",
    "TestsFlextCleanProtocols",
    "TestsFlextCleanServiceBase",
    "TestsFlextConstants",
    "TestsFlextConstantsDomain",
    "TestsFlextConstantsErrors",
    "TestsFlextConstantsFixtures",
    "TestsFlextConstantsLoggings",
    "TestsFlextConstantsNew",
    "TestsFlextConstantsOther",
    "TestsFlextConstantsProjectMetadata",
    "TestsFlextConstantsResult",
    "TestsFlextConstantsServices",
    "TestsFlextConstantsSettings",
    "TestsFlextContainer",
    "TestsFlextContainerMemory",
    "TestsFlextContainerPerformance",
    "TestsFlextContext",
    "TestsFlextCoverageExceptions",
    "TestsFlextCoverageLoggings",
    "TestsFlextDecorators",
    "TestsFlextDecoratorsDiscovery",
    "TestsFlextDecoratorsLegacy",
    "TestsFlextDeprecationWarnings",
    "TestsFlextDispatcher",
    "TestsFlextDocumentedPatterns",
    "TestsFlextEnforcement",
    "TestsFlextEnforcementAptHooks",
    "TestsFlextEnforcementCatalog",
    "TestsFlextEnforcementIntegration",
    "TestsFlextEnumUtilities",
    "TestsFlextExamplesExecution",
    "TestsFlextExceptions",
    "TestsFlextFacadeFlatSsotAccess",
    "TestsFlextFlextHandlers",
    "TestsFlextHandlerDecoratorDiscovery",
    "TestsFlextLazy",
    "TestsFlextLazyPerformance",
    "TestsFlextLibraryIntegration",
    "TestsFlextLoggings",
    "TestsFlextMigrationValidation",
    "TestsFlextMixins",
    "TestsFlextModels",
    "TestsFlextModelsBase",
    "TestsFlextModelsBaseFullCoverage",
    "TestsFlextModelsCQRS",
    "TestsFlextModelsContainer",
    "TestsFlextModelsCqrs",
    "TestsFlextModelsEnforcementSources",
    "TestsFlextModelsEntity",
    "TestsFlextModelsExceptionParams",
    "TestsFlextModelsMixins",
    "TestsFlextModelsProjectMetadata",
    "TestsFlextModelsUnit",
    "TestsFlextPatternsCommands",
    "TestsFlextPatternsLogging",
    "TestsFlextPatternsTesting",
    "TestsFlextProtocols",
    "TestsFlextRegistry",
    "TestsFlextResult",
    "TestsFlextResultExceptionCarrying",
    "TestsFlextRuntime",
    "TestsFlextRuntimeViolationRegistry",
    "TestsFlextService",
    "TestsFlextService100Coverage",
    "TestsFlextServiceBase",
    "TestsFlextServiceBootstrap",
    "TestsFlextServiceIntegration",
    "TestsFlextServiceResultProperty",
    "TestsFlextSettings",
    "TestsFlextSettingsIntegration",
    "TestsFlextSystemIntegration",
    "TestsFlextTypes",
    "TestsFlextTypesUnit",
    "TestsFlextUtilities",
    "TestsFlextUtilitiesCollection",
    "TestsFlextUtilitiesCoverage",
    "TestsFlextUtilitiesDomain",
    "TestsFlextUtilitiesGenerators",
    "TestsFlextUtilitiesGuards",
    "TestsFlextUtilitiesMapper",
    "TestsFlextUtilitiesProjectMetadata",
    "TestsFlextUtilitiesPydantic",
    "TestsFlextUtilitiesReliability",
    "TestsFlextUtilitiesSettings",
    "TestsFlextUtilitiesSettingsEnvFile",
    "TestsFlextUtilitiesSettingsRegisterFactory",
    "TestsFlextUtilitiesSmoke",
    "TestsFlextUtilitiesText",
    "TestsFlextUtilitiesTypeGuards",
    "TestsFlextVersion",
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
