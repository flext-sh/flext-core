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
    from flext_tests import (
        d as d,
        e as e,
        h as h,
        r as r,
        td as td,
        tf as tf,
        tk as tk,
        tv as tv,
        x as x,
    )

    from tests._constants.domain import (
        TestsFlextConstantsDomain as TestsFlextConstantsDomain,
    )
    from tests._constants.errors import (
        TestsFlextConstantsErrors as TestsFlextConstantsErrors,
    )
    from tests._constants.fixtures import (
        TestsFlextConstantsFixtures as TestsFlextConstantsFixtures,
    )
    from tests._constants.loggings import (
        TestsFlextConstantsLoggings as TestsFlextConstantsLoggings,
    )
    from tests._constants.other import (
        TestsFlextConstantsOther as TestsFlextConstantsOther,
    )
    from tests._constants.result import (
        TestsFlextConstantsResult as TestsFlextConstantsResult,
    )
    from tests._constants.services import (
        TestsFlextConstantsServices as TestsFlextConstantsServices,
    )
    from tests._constants.settings import (
        TestsFlextConstantsSettings as TestsFlextConstantsSettings,
    )
    from tests._models.mixins import TestsFlextModelsMixins as TestsFlextModelsMixins
    from tests.base import TestsFlextServiceBase as TestsFlextServiceBase, s as s
    from tests.benchmark.test_container_memory import (
        TestsFlextContainerMemory as TestsFlextContainerMemory,
    )
    from tests.benchmark.test_container_performance import (
        TestsFlextContainerPerformance as TestsFlextContainerPerformance,
    )
    from tests.benchmark.test_lazy_performance import (
        TestsFlextLazyPerformance as TestsFlextLazyPerformance,
    )
    from tests.constants import TestsFlextConstants as TestsFlextConstants, c as c
    from tests.fixtures.bad_module import (
        TestsFlextBadAccessors as TestsFlextBadAccessors,
        TestsFlextBadAnyField as TestsFlextBadAnyField,
        TestsFlextBadBareCollection as TestsFlextBadBareCollection,
        TestsFlextBadConstants as TestsFlextBadConstants,
        TestsFlextBadFrozen as TestsFlextBadFrozen,
        TestsFlextBadInlineUnion as TestsFlextBadInlineUnion,
        TestsFlextBadMissingDesc as TestsFlextBadMissingDesc,
        TestsFlextBadMutableDefault as TestsFlextBadMutableDefault,
        TestsFlextBadWorkerSettings as TestsFlextBadWorkerSettings,
    )
    from tests.fixtures.clean_module import (
        TestsFlextCleanConstants as TestsFlextCleanConstants,
        TestsFlextCleanModels as TestsFlextCleanModels,
        TestsFlextCleanProtocols as TestsFlextCleanProtocols,
        TestsFlextCleanServiceBase as TestsFlextCleanServiceBase,
    )
    from tests.integration.test_architecture import (
        TestsFlextAutomatedArchitecture as TestsFlextAutomatedArchitecture,
    )
    from tests.integration.test_documented_patterns import (
        TestsFlextDocumentedPatterns as TestsFlextDocumentedPatterns,
    )
    from tests.integration.test_examples_execution import (
        TestsFlextExamplesExecution as TestsFlextExamplesExecution,
    )
    from tests.integration.test_integration import (
        TestsFlextLibraryIntegration as TestsFlextLibraryIntegration,
    )
    from tests.integration.test_migration_validation import (
        TestsFlextMigrationValidation as TestsFlextMigrationValidation,
    )
    from tests.integration.test_service import (
        TestsFlextServiceIntegration as TestsFlextServiceIntegration,
    )
    from tests.integration.test_settings_integration import (
        TestsFlextSettingsIntegration as TestsFlextSettingsIntegration,
    )
    from tests.integration.test_system import (
        TestsFlextSystemIntegration as TestsFlextSystemIntegration,
    )
    from tests.models import TestsFlextModels as TestsFlextModels, m as m
    from tests.protocols import TestsFlextProtocols as TestsFlextProtocols, p as p
    from tests.typings import TestsFlextTypes as TestsFlextTypes, t as t
    from tests.unit._models.test_base import (
        TestsFlextModelsBase as TestsFlextModelsBase,
    )
    from tests.unit._models.test_cqrs import (
        TestsFlextModelsCQRS as TestsFlextModelsCQRS,
    )
    from tests.unit._models.test_enforcement_sources import (
        TestsFlextModelsEnforcementSources as TestsFlextModelsEnforcementSources,
    )
    from tests.unit._models.test_entity import (
        TestsFlextModelsEntity as TestsFlextModelsEntity,
    )
    from tests.unit._models.test_exception_params import (
        TestsFlextModelsExceptionParams as TestsFlextModelsExceptionParams,
    )
    from tests.unit._utilities.test_guards import (
        TestsFlextUtilitiesGuards as TestsFlextUtilitiesGuards,
    )
    from tests.unit._utilities.test_mapper import (
        TestsFlextUtilitiesMapper as TestsFlextUtilitiesMapper,
    )
    from tests.unit.test_beartype_engine import (
        TestsFlextBeartypeEngine as TestsFlextBeartypeEngine,
    )
    from tests.unit.test_constants_new import (
        TestsFlextConstantsNew as TestsFlextConstantsNew,
    )
    from tests.unit.test_constants_project_metadata import (
        TestsFlextConstantsProjectMetadata as TestsFlextConstantsProjectMetadata,
    )
    from tests.unit.test_container import TestsFlextContainer as TestsFlextContainer
    from tests.unit.test_context import TestsFlextContext as TestsFlextContext
    from tests.unit.test_coverage_loggings import (
        TestsFlextCoverageLoggings as TestsFlextCoverageLoggings,
    )
    from tests.unit.test_decorators import (
        TestsFlextDecoratorsLegacy as TestsFlextDecoratorsLegacy,
    )
    from tests.unit.test_decorators_discovery_full_coverage import (
        TestsFlextDecoratorsDiscovery as TestsFlextDecoratorsDiscovery,
    )
    from tests.unit.test_decorators_full_coverage import (
        TestsFlextDecorators as TestsFlextDecorators,
    )
    from tests.unit.test_deprecation_warnings import (
        TestsFlextDeprecationWarnings as TestsFlextDeprecationWarnings,
    )
    from tests.unit.test_dispatcher import TestsFlextDispatcher as TestsFlextDispatcher
    from tests.unit.test_enforcement import (
        TestsFlextEnforcement as TestsFlextEnforcement,
    )
    from tests.unit.test_enforcement_apt_hooks import (
        TestsFlextEnforcementAptHooks as TestsFlextEnforcementAptHooks,
    )
    from tests.unit.test_enforcement_catalog import (
        TestsFlextEnforcementCatalog as TestsFlextEnforcementCatalog,
    )
    from tests.unit.test_enforcement_integration import (
        TestsFlextEnforcementIntegration as TestsFlextEnforcementIntegration,
    )
    from tests.unit.test_enum_utilities_coverage_100 import (
        TestsFlextEnumUtilities as TestsFlextEnumUtilities,
    )
    from tests.unit.test_exceptions import (
        TestsFlextCoverageExceptions as TestsFlextCoverageExceptions,
        TestsFlextExceptions as TestsFlextExceptions,
    )
    from tests.unit.test_handler_decorator_discovery import (
        TestsFlextHandlerDecoratorDiscovery as TestsFlextHandlerDecoratorDiscovery,
    )
    from tests.unit.test_handlers import (
        TestsFlextFlextHandlers as TestsFlextFlextHandlers,
    )
    from tests.unit.test_lazy_exports import TestsFlextLazy as TestsFlextLazy
    from tests.unit.test_loggings_full_coverage import (
        TestsFlextLoggings as TestsFlextLoggings,
    )
    from tests.unit.test_mixins import TestsFlextMixins as TestsFlextMixins
    from tests.unit.test_models import TestsFlextModelsUnit as TestsFlextModelsUnit
    from tests.unit.test_models_base_full_coverage import (
        TestsFlextModelsBaseFullCoverage as TestsFlextModelsBaseFullCoverage,
    )
    from tests.unit.test_models_container import (
        TestsFlextModelsContainer as TestsFlextModelsContainer,
    )
    from tests.unit.test_models_cqrs_full_coverage import (
        TestsFlextModelsCqrs as TestsFlextModelsCqrs,
    )
    from tests.unit.test_models_project_metadata import (
        TestsFlextModelsProjectMetadata as TestsFlextModelsProjectMetadata,
    )
    from tests.unit.test_project_metadata_facade_access import (
        TestsFlextFacadeFlatSsotAccess as TestsFlextFacadeFlatSsotAccess,
    )
    from tests.unit.test_public_api_contract import (
        TestsFlextCorePublicApiContract as TestsFlextCorePublicApiContract,
    )
    from tests.unit.test_registry import TestsFlextRegistry as TestsFlextRegistry
    from tests.unit.test_result import (
        TestsFlextResult as TestsFlextResult,
        TestsFlextResultExceptionCarrying as TestsFlextResultExceptionCarrying,
    )
    from tests.unit.test_runtime import TestsFlextRuntime as TestsFlextRuntime
    from tests.unit.test_service import TestsFlextService as TestsFlextService
    from tests.unit.test_service_bootstrap import (
        TestsFlextServiceBootstrap as TestsFlextServiceBootstrap,
    )
    from tests.unit.test_settings import TestsFlextSettings as TestsFlextSettings
    from tests.unit.test_typings_new import TestsFlextTypesUnit as TestsFlextTypesUnit
    from tests.unit.test_utilities import (
        TestsFlextUtilitiesSmoke as TestsFlextUtilitiesSmoke,
    )
    from tests.unit.test_utilities_collection_coverage_100 import (
        TestsFlextUtilitiesCollection as TestsFlextUtilitiesCollection,
    )
    from tests.unit.test_utilities_coverage import (
        TestsFlextUtilitiesCoverage as TestsFlextUtilitiesCoverage,
    )
    from tests.unit.test_utilities_domain import (
        TestsFlextUtilitiesDomain as TestsFlextUtilitiesDomain,
    )
    from tests.unit.test_utilities_generators_full_coverage import (
        TestsFlextUtilitiesGenerators as TestsFlextUtilitiesGenerators,
    )
    from tests.unit.test_utilities_project_metadata import (
        TestsFlextUtilitiesProjectMetadata as TestsFlextUtilitiesProjectMetadata,
    )
    from tests.unit.test_utilities_pydantic_coverage_100 import (
        TestsFlextUtilitiesPydantic as TestsFlextUtilitiesPydantic,
    )
    from tests.unit.test_utilities_reliability import (
        TestsFlextUtilitiesReliability as TestsFlextUtilitiesReliability,
    )
    from tests.unit.test_utilities_runtime_violation_registry_coverage_100 import (
        TestsFlextRuntimeViolationRegistry as TestsFlextRuntimeViolationRegistry,
    )
    from tests.unit.test_utilities_settings_coverage_100 import (
        TestsFlextUtilitiesSettings as TestsFlextUtilitiesSettings,
        TestsFlextUtilitiesSettingsEnvFile as TestsFlextUtilitiesSettingsEnvFile,
        TestsFlextUtilitiesSettingsRegisterFactory as TestsFlextUtilitiesSettingsRegisterFactory,
    )
    from tests.unit.test_utilities_text_full_coverage import (
        TestsFlextUtilitiesText as TestsFlextUtilitiesText,
    )
    from tests.unit.test_utilities_type_guards_coverage_100 import (
        TestsFlextUtilitiesTypeGuards as TestsFlextUtilitiesTypeGuards,
    )
    from tests.unit.test_version import TestsFlextVersion as TestsFlextVersion
    from tests.utilities import TestsFlextUtilities as TestsFlextUtilities, u as u
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
            ".base": (
                "TestsFlextServiceBase",
                "s",
            ),
            ".benchmark.test_container_memory": ("TestsFlextContainerMemory",),
            ".benchmark.test_container_performance": (
                "TestsFlextContainerPerformance",
            ),
            ".benchmark.test_lazy_performance": ("TestsFlextLazyPerformance",),
            ".constants": (
                "TestsFlextConstants",
                "c",
            ),
            ".fixtures.bad_module": (
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
            ".fixtures.clean_module": (
                "TestsFlextCleanConstants",
                "TestsFlextCleanModels",
                "TestsFlextCleanProtocols",
                "TestsFlextCleanServiceBase",
            ),
            ".integration.test_architecture": ("TestsFlextAutomatedArchitecture",),
            ".integration.test_documented_patterns": ("TestsFlextDocumentedPatterns",),
            ".integration.test_examples_execution": ("TestsFlextExamplesExecution",),
            ".integration.test_integration": ("TestsFlextLibraryIntegration",),
            ".integration.test_migration_validation": (
                "TestsFlextMigrationValidation",
            ),
            ".integration.test_service": ("TestsFlextServiceIntegration",),
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
            ".unit._models.test_base": ("TestsFlextModelsBase",),
            ".unit._models.test_cqrs": ("TestsFlextModelsCQRS",),
            ".unit._models.test_enforcement_sources": (
                "TestsFlextModelsEnforcementSources",
            ),
            ".unit._models.test_entity": ("TestsFlextModelsEntity",),
            ".unit._models.test_exception_params": ("TestsFlextModelsExceptionParams",),
            ".unit._utilities.test_guards": ("TestsFlextUtilitiesGuards",),
            ".unit._utilities.test_mapper": ("TestsFlextUtilitiesMapper",),
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
            ".unit.test_public_api_contract": ("TestsFlextCorePublicApiContract",),
            ".unit.test_registry": ("TestsFlextRegistry",),
            ".unit.test_result": (
                "TestsFlextResult",
                "TestsFlextResultExceptionCarrying",
            ),
            ".unit.test_runtime": ("TestsFlextRuntime",),
            ".unit.test_service": ("TestsFlextService",),
            ".unit.test_service_bootstrap": ("TestsFlextServiceBootstrap",),
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
                "td",
                "tf",
                "tk",
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
    "TestsFlextConstantsNew",
    "TestsFlextConstantsProjectMetadata",
    "TestsFlextContainer",
    "TestsFlextContainerMemory",
    "TestsFlextContainerPerformance",
    "TestsFlextContext",
    "TestsFlextCorePublicApiContract",
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
    "TestsFlextModelsBaseFullCoverage",
    "TestsFlextModelsContainer",
    "TestsFlextModelsCqrs",
    "TestsFlextModelsProjectMetadata",
    "TestsFlextModelsUnit",
    "TestsFlextProtocols",
    "TestsFlextRegistry",
    "TestsFlextResult",
    "TestsFlextResultExceptionCarrying",
    "TestsFlextRuntime",
    "TestsFlextRuntimeViolationRegistry",
    "TestsFlextService",
    "TestsFlextServiceBase",
    "TestsFlextServiceBootstrap",
    "TestsFlextServiceIntegration",
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
    "tv",
    "u",
    "x",
]
