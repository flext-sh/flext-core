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
    from tests.benchmark.test_container_memory import TestsFlextCoreContainerMemory
    from tests.benchmark.test_container_performance import (
        TestsFlextCoreContainerPerformance,
    )
    from tests.benchmark.test_lazy_performance import TestsFlextCoreLazyPerformance
    from tests.constants import TestsFlextCoreConstants, c
    from tests.integration.patterns.test_advanced_patterns import (
        TestsFlextCoreAdvancedPatterns,
    )
    from tests.integration.patterns.test_architectural_patterns import (
        TestsFlextCoreArchitecturalPatterns,
    )
    from tests.integration.patterns.test_patterns_commands import (
        TestsFlextCorePatternsCommands,
    )
    from tests.integration.patterns.test_patterns_logging import (
        TestsFlextCorePatternsLogging,
    )
    from tests.integration.patterns.test_patterns_testing import (
        TestsFlextCorePatternsTesting,
    )
    from tests.integration.test_architecture import TestsFlextCoreAutomatedArchitecture
    from tests.integration.test_documented_patterns import (
        TestsFlextCoreDocumentedPatterns,
    )
    from tests.integration.test_examples_execution import (
        TestsFlextCoreExamplesExecution,
    )
    from tests.integration.test_integration import TestsFlextCoreLibraryIntegration
    from tests.integration.test_migration_validation import (
        TestsFlextCoreMigrationValidation,
    )
    from tests.integration.test_service import TestsFlextCoreServiceIntegration
    from tests.integration.test_service_result_property import (
        TestsFlextCoreServiceResultProperty,
    )
    from tests.integration.test_settings_integration import (
        TestsFlextCoreSettingsIntegration,
    )
    from tests.integration.test_system import TestsFlextCoreSystemIntegration
    from tests.models import TestsFlextCoreModels, m
    from tests.protocols import TestsFlextCoreProtocols, p
    from tests.typings import TestsFlextCoreTypes, t
    from tests.unit._models.test_base import TestsFlextCoreModelsBase
    from tests.unit._models.test_cqrs import TestsFlextCoreModelsCQRS
    from tests.unit._models.test_enforcement_sources import (
        TestsFlextCoreModelsEnforcementSources,
    )
    from tests.unit._models.test_entity import TestsFlextCoreModelsEntity
    from tests.unit._models.test_exception_params import (
        TestsFlextCoreModelsExceptionParams,
    )
    from tests.unit._utilities.test_guards import TestsFlextCoreUtilitiesGuards
    from tests.unit._utilities.test_mapper import TestsFlextCoreUtilitiesMapper
    from tests.unit.base import TestsFlextCoreServiceBase
    from tests.unit.test_beartype_engine import TestsFlextCoreBeartypeEngine
    from tests.unit.test_constants_new import TestsFlextCoreConstantsNew
    from tests.unit.test_constants_project_metadata import (
        TestsFlextCoreConstantsProjectMetadata,
    )
    from tests.unit.test_container import TestsFlextCoreContainer
    from tests.unit.test_context import TestsFlextCoreContext
    from tests.unit.test_coverage_exceptions import TestsFlextCoreCoverageExceptions
    from tests.unit.test_coverage_loggings import TestsFlextCoreCoverageLoggings
    from tests.unit.test_decorators import TestsFlextCoreDecoratorsLegacy
    from tests.unit.test_decorators_discovery_full_coverage import (
        TestsFlextCoreDecoratorsDiscovery,
    )
    from tests.unit.test_decorators_full_coverage import TestsFlextCoreDecorators
    from tests.unit.test_deprecation_warnings import TestsFlextCoreDeprecationWarnings
    from tests.unit.test_dispatcher_di import TestsFlextCoreDispatcherDI
    from tests.unit.test_dispatcher_minimal import TestsFlextCoreDispatcherMinimal
    from tests.unit.test_dispatcher_reliability import (
        TestsFlextCoreDispatcherReliability,
    )
    from tests.unit.test_enforcement import TestsFlextCoreEnforcement
    from tests.unit.test_enforcement_apt_hooks import TestsFlextCoreEnforcementAptHooks
    from tests.unit.test_enforcement_catalog import TestsFlextCoreEnforcementCatalog
    from tests.unit.test_enforcement_integration import (
        TestsFlextCoreEnforcementIntegration,
    )
    from tests.unit.test_enum_utilities_coverage_100 import TestsFlextCoreEnumUtilities
    from tests.unit.test_exceptions import TestsFlextCoreExceptions
    from tests.unit.test_handler_decorator_discovery import (
        TestsFlextCoreHandlerDecoratorDiscovery,
    )
    from tests.unit.test_handlers import TestsFlextCoreFlextHandlers
    from tests.unit.test_lazy_exports import TestsFlextCoreLazy
    from tests.unit.test_loggings_full_coverage import TestsFlextCoreLoggings
    from tests.unit.test_mixins import TestsFlextCoreMixins
    from tests.unit.test_models import TestsFlextCoreModelsUnit
    from tests.unit.test_models_base_full_coverage import (
        TestsFlextCoreModelsBaseFullCoverage,
    )
    from tests.unit.test_models_container import TestsFlextCoreModelsContainer
    from tests.unit.test_models_cqrs_full_coverage import TestsFlextCoreModelsCqrs
    from tests.unit.test_models_project_metadata import (
        TestsFlextCoreModelsProjectMetadata,
    )
    from tests.unit.test_project_metadata_facade_access import (
        TestsFlextCoreFacadeFlatSsotAccess,
    )
    from tests.unit.test_registry import TestsFlextCoreRegistry
    from tests.unit.test_registry_full_coverage import (
        TestsFlextCoreRegistryFullCoverage,
    )
    from tests.unit.test_result import TestsFlextCoreResult
    from tests.unit.test_result_exception_carrying import (
        TestsFlextCoreResultExceptionCarrying,
    )
    from tests.unit.test_runtime import TestsFlextCoreRuntime
    from tests.unit.test_service import (
        TestsFlextCoreService,
        TestsFlextCoreServiceUserData,
        TestsFlextCoreServiceUserService,
    )
    from tests.unit.test_service_bootstrap import TestsFlextCoreServiceBootstrap
    from tests.unit.test_service_coverage_100 import TestsFlextCoreService100Coverage
    from tests.unit.test_settings import TestsFlextCoreSettings
    from tests.unit.test_settings_coverage import TestsFlextCoreSettingsCoverage
    from tests.unit.test_typings_new import TestsFlextCoreTypesUnit
    from tests.unit.test_utilities import TestsFlextCoreUtilitiesSmoke
    from tests.unit.test_utilities_collection_coverage_100 import (
        TestsFlextCoreUtilitiesCollection,
    )
    from tests.unit.test_utilities_coverage import TestsFlextCoreUtilitiesCoverage
    from tests.unit.test_utilities_domain import TestsFlextCoreUtilitiesDomain
    from tests.unit.test_utilities_generators_full_coverage import (
        TestsFlextCoreUtilitiesGenerators,
    )
    from tests.unit.test_utilities_project_metadata import (
        TestsFlextCoreUtilitiesProjectMetadata,
    )
    from tests.unit.test_utilities_reliability import TestsFlextCoreUtilitiesReliability
    from tests.unit.test_utilities_settings_coverage_100 import (
        TestsFlextCoreUtilitiesSettings,
    )
    from tests.unit.test_utilities_text_full_coverage import TestsFlextCoreUtilitiesText
    from tests.unit.test_utilities_type_guards_coverage_100 import (
        TestsFlextCoreUtilitiesTypeGuards,
    )
    from tests.unit.test_version import TestsFlextCoreVersion
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
            ".benchmark.test_container_memory": ("TestsFlextCoreContainerMemory",),
            ".benchmark.test_container_performance": (
                "TestsFlextCoreContainerPerformance",
            ),
            ".benchmark.test_lazy_performance": ("TestsFlextCoreLazyPerformance",),
            ".constants": (
                "TestsFlextCoreConstants",
                "c",
            ),
            ".integration.patterns.test_advanced_patterns": (
                "TestsFlextCoreAdvancedPatterns",
            ),
            ".integration.patterns.test_architectural_patterns": (
                "TestsFlextCoreArchitecturalPatterns",
            ),
            ".integration.patterns.test_patterns_commands": (
                "TestsFlextCorePatternsCommands",
            ),
            ".integration.patterns.test_patterns_logging": (
                "TestsFlextCorePatternsLogging",
            ),
            ".integration.patterns.test_patterns_testing": (
                "TestsFlextCorePatternsTesting",
            ),
            ".integration.test_architecture": ("TestsFlextCoreAutomatedArchitecture",),
            ".integration.test_documented_patterns": (
                "TestsFlextCoreDocumentedPatterns",
            ),
            ".integration.test_examples_execution": (
                "TestsFlextCoreExamplesExecution",
            ),
            ".integration.test_integration": ("TestsFlextCoreLibraryIntegration",),
            ".integration.test_migration_validation": (
                "TestsFlextCoreMigrationValidation",
            ),
            ".integration.test_service": ("TestsFlextCoreServiceIntegration",),
            ".integration.test_service_result_property": (
                "TestsFlextCoreServiceResultProperty",
            ),
            ".integration.test_settings_integration": (
                "TestsFlextCoreSettingsIntegration",
            ),
            ".integration.test_system": ("TestsFlextCoreSystemIntegration",),
            ".models": (
                "TestsFlextCoreModels",
                "m",
            ),
            ".protocols": (
                "TestsFlextCoreProtocols",
                "p",
            ),
            ".typings": (
                "TestsFlextCoreTypes",
                "t",
            ),
            ".unit._models.test_base": ("TestsFlextCoreModelsBase",),
            ".unit._models.test_cqrs": ("TestsFlextCoreModelsCQRS",),
            ".unit._models.test_enforcement_sources": (
                "TestsFlextCoreModelsEnforcementSources",
            ),
            ".unit._models.test_entity": ("TestsFlextCoreModelsEntity",),
            ".unit._models.test_exception_params": (
                "TestsFlextCoreModelsExceptionParams",
            ),
            ".unit._utilities.test_guards": ("TestsFlextCoreUtilitiesGuards",),
            ".unit._utilities.test_mapper": ("TestsFlextCoreUtilitiesMapper",),
            ".unit.base": ("TestsFlextCoreServiceBase",),
            ".unit.test_beartype_engine": ("TestsFlextCoreBeartypeEngine",),
            ".unit.test_constants_new": ("TestsFlextCoreConstantsNew",),
            ".unit.test_constants_project_metadata": (
                "TestsFlextCoreConstantsProjectMetadata",
            ),
            ".unit.test_container": ("TestsFlextCoreContainer",),
            ".unit.test_context": ("TestsFlextCoreContext",),
            ".unit.test_coverage_exceptions": ("TestsFlextCoreCoverageExceptions",),
            ".unit.test_coverage_loggings": ("TestsFlextCoreCoverageLoggings",),
            ".unit.test_decorators": ("TestsFlextCoreDecoratorsLegacy",),
            ".unit.test_decorators_discovery_full_coverage": (
                "TestsFlextCoreDecoratorsDiscovery",
            ),
            ".unit.test_decorators_full_coverage": ("TestsFlextCoreDecorators",),
            ".unit.test_deprecation_warnings": ("TestsFlextCoreDeprecationWarnings",),
            ".unit.test_dispatcher_di": ("TestsFlextCoreDispatcherDI",),
            ".unit.test_dispatcher_minimal": ("TestsFlextCoreDispatcherMinimal",),
            ".unit.test_dispatcher_reliability": (
                "TestsFlextCoreDispatcherReliability",
            ),
            ".unit.test_enforcement": ("TestsFlextCoreEnforcement",),
            ".unit.test_enforcement_apt_hooks": ("TestsFlextCoreEnforcementAptHooks",),
            ".unit.test_enforcement_catalog": ("TestsFlextCoreEnforcementCatalog",),
            ".unit.test_enforcement_integration": (
                "TestsFlextCoreEnforcementIntegration",
            ),
            ".unit.test_enum_utilities_coverage_100": ("TestsFlextCoreEnumUtilities",),
            ".unit.test_exceptions": ("TestsFlextCoreExceptions",),
            ".unit.test_handler_decorator_discovery": (
                "TestsFlextCoreHandlerDecoratorDiscovery",
            ),
            ".unit.test_handlers": ("TestsFlextCoreFlextHandlers",),
            ".unit.test_lazy_exports": ("TestsFlextCoreLazy",),
            ".unit.test_loggings_full_coverage": ("TestsFlextCoreLoggings",),
            ".unit.test_mixins": ("TestsFlextCoreMixins",),
            ".unit.test_models": ("TestsFlextCoreModelsUnit",),
            ".unit.test_models_base_full_coverage": (
                "TestsFlextCoreModelsBaseFullCoverage",
            ),
            ".unit.test_models_container": ("TestsFlextCoreModelsContainer",),
            ".unit.test_models_cqrs_full_coverage": ("TestsFlextCoreModelsCqrs",),
            ".unit.test_models_project_metadata": (
                "TestsFlextCoreModelsProjectMetadata",
            ),
            ".unit.test_project_metadata_facade_access": (
                "TestsFlextCoreFacadeFlatSsotAccess",
            ),
            ".unit.test_registry": ("TestsFlextCoreRegistry",),
            ".unit.test_registry_full_coverage": (
                "TestsFlextCoreRegistryFullCoverage",
            ),
            ".unit.test_result": ("TestsFlextCoreResult",),
            ".unit.test_result_exception_carrying": (
                "TestsFlextCoreResultExceptionCarrying",
            ),
            ".unit.test_runtime": ("TestsFlextCoreRuntime",),
            ".unit.test_service": (
                "TestsFlextCoreService",
                "TestsFlextCoreServiceUserData",
                "TestsFlextCoreServiceUserService",
            ),
            ".unit.test_service_bootstrap": ("TestsFlextCoreServiceBootstrap",),
            ".unit.test_service_coverage_100": ("TestsFlextCoreService100Coverage",),
            ".unit.test_settings": ("TestsFlextCoreSettings",),
            ".unit.test_settings_coverage": ("TestsFlextCoreSettingsCoverage",),
            ".unit.test_typings_new": ("TestsFlextCoreTypesUnit",),
            ".unit.test_utilities": ("TestsFlextCoreUtilitiesSmoke",),
            ".unit.test_utilities_collection_coverage_100": (
                "TestsFlextCoreUtilitiesCollection",
            ),
            ".unit.test_utilities_coverage": ("TestsFlextCoreUtilitiesCoverage",),
            ".unit.test_utilities_domain": ("TestsFlextCoreUtilitiesDomain",),
            ".unit.test_utilities_generators_full_coverage": (
                "TestsFlextCoreUtilitiesGenerators",
            ),
            ".unit.test_utilities_project_metadata": (
                "TestsFlextCoreUtilitiesProjectMetadata",
            ),
            ".unit.test_utilities_reliability": ("TestsFlextCoreUtilitiesReliability",),
            ".unit.test_utilities_settings_coverage_100": (
                "TestsFlextCoreUtilitiesSettings",
            ),
            ".unit.test_utilities_text_full_coverage": ("TestsFlextCoreUtilitiesText",),
            ".unit.test_utilities_type_guards_coverage_100": (
                "TestsFlextCoreUtilitiesTypeGuards",
            ),
            ".unit.test_version": ("TestsFlextCoreVersion",),
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
    "TestsFlextCoreAdvancedPatterns",
    "TestsFlextCoreArchitecturalPatterns",
    "TestsFlextCoreAutomatedArchitecture",
    "TestsFlextCoreBeartypeEngine",
    "TestsFlextCoreConstants",
    "TestsFlextCoreConstantsDomain",
    "TestsFlextCoreConstantsErrors",
    "TestsFlextCoreConstantsFixtures",
    "TestsFlextCoreConstantsLoggings",
    "TestsFlextCoreConstantsNew",
    "TestsFlextCoreConstantsOther",
    "TestsFlextCoreConstantsProjectMetadata",
    "TestsFlextCoreConstantsResult",
    "TestsFlextCoreConstantsServices",
    "TestsFlextCoreConstantsSettings",
    "TestsFlextCoreConstantsStrings",
    "TestsFlextCoreContainer",
    "TestsFlextCoreContainerMemory",
    "TestsFlextCoreContainerPerformance",
    "TestsFlextCoreContext",
    "TestsFlextCoreCoverageExceptions",
    "TestsFlextCoreCoverageLoggings",
    "TestsFlextCoreDecorators",
    "TestsFlextCoreDecoratorsDiscovery",
    "TestsFlextCoreDecoratorsLegacy",
    "TestsFlextCoreDeprecationWarnings",
    "TestsFlextCoreDispatcherDI",
    "TestsFlextCoreDispatcherMinimal",
    "TestsFlextCoreDispatcherReliability",
    "TestsFlextCoreDocumentedPatterns",
    "TestsFlextCoreEnforcement",
    "TestsFlextCoreEnforcementAptHooks",
    "TestsFlextCoreEnforcementCatalog",
    "TestsFlextCoreEnforcementIntegration",
    "TestsFlextCoreEnumUtilities",
    "TestsFlextCoreExamplesExecution",
    "TestsFlextCoreExceptions",
    "TestsFlextCoreFacadeFlatSsotAccess",
    "TestsFlextCoreFlextHandlers",
    "TestsFlextCoreHandlerDecoratorDiscovery",
    "TestsFlextCoreLazy",
    "TestsFlextCoreLazyPerformance",
    "TestsFlextCoreLibraryIntegration",
    "TestsFlextCoreLoggings",
    "TestsFlextCoreMigrationValidation",
    "TestsFlextCoreMixins",
    "TestsFlextCoreModels",
    "TestsFlextCoreModelsBase",
    "TestsFlextCoreModelsBaseFullCoverage",
    "TestsFlextCoreModelsCQRS",
    "TestsFlextCoreModelsContainer",
    "TestsFlextCoreModelsCqrs",
    "TestsFlextCoreModelsEnforcementSources",
    "TestsFlextCoreModelsEntity",
    "TestsFlextCoreModelsExceptionParams",
    "TestsFlextCoreModelsMixins",
    "TestsFlextCoreModelsProjectMetadata",
    "TestsFlextCoreModelsUnit",
    "TestsFlextCorePatternsCommands",
    "TestsFlextCorePatternsLogging",
    "TestsFlextCorePatternsTesting",
    "TestsFlextCoreProtocols",
    "TestsFlextCoreRegistry",
    "TestsFlextCoreRegistryFullCoverage",
    "TestsFlextCoreResult",
    "TestsFlextCoreResultExceptionCarrying",
    "TestsFlextCoreRuntime",
    "TestsFlextCoreService",
    "TestsFlextCoreService100Coverage",
    "TestsFlextCoreServiceBase",
    "TestsFlextCoreServiceBootstrap",
    "TestsFlextCoreServiceIntegration",
    "TestsFlextCoreServiceResultProperty",
    "TestsFlextCoreServiceUserData",
    "TestsFlextCoreServiceUserService",
    "TestsFlextCoreSettings",
    "TestsFlextCoreSettingsCoverage",
    "TestsFlextCoreSettingsIntegration",
    "TestsFlextCoreSystemIntegration",
    "TestsFlextCoreTypes",
    "TestsFlextCoreTypesUnit",
    "TestsFlextCoreUtilities",
    "TestsFlextCoreUtilitiesCollection",
    "TestsFlextCoreUtilitiesCoverage",
    "TestsFlextCoreUtilitiesDomain",
    "TestsFlextCoreUtilitiesGenerators",
    "TestsFlextCoreUtilitiesGuards",
    "TestsFlextCoreUtilitiesMapper",
    "TestsFlextCoreUtilitiesProjectMetadata",
    "TestsFlextCoreUtilitiesReliability",
    "TestsFlextCoreUtilitiesSettings",
    "TestsFlextCoreUtilitiesSmoke",
    "TestsFlextCoreUtilitiesText",
    "TestsFlextCoreUtilitiesTypeGuards",
    "TestsFlextCoreVersion",
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
