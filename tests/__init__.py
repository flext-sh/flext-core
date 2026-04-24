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
    from tests.typings import TestsFlextCoreTypes, t
    from tests.unit._models.test_base import TestModelsBase
    from tests.unit._models.test_cqrs import TestModelsCQRS
    from tests.unit._models.test_entity import TestModelsEntity
    from tests.unit._models.test_exception_params import TestFlextModelsExceptionParams
    from tests.unit._utilities.test_guards import TestUtilitiesGuards
    from tests.unit._utilities.test_mapper import TestUtilitiesMapper
    from tests.unit.base import TestsFlextCoreServiceBase
    from tests.unit.test_beartype_engine import (
        TestAliasContainsAny,
        TestBeartypeClawCompatibility,
        TestBeartypeConf,
        TestContainsAny,
        TestCountUnionMembers,
        TestFacadeAccessibility,
        TestForbiddenCollectionOrigin,
        TestMatchesStrNoneUnion,
    )
    from tests.unit.test_constants_new import TestFlextConstants
    from tests.unit.test_constants_project_metadata import (
        TestsFlextCoreConstantsProjectMetadata,
    )
    from tests.unit.test_container import TestFlextContainer
    from tests.unit.test_context import TestsFlextCoreContext
    from tests.unit.test_coverage_exceptions import TestCoverageExceptions
    from tests.unit.test_coverage_loggings import TestCoverageLoggings
    from tests.unit.test_decorators import TestFlextDecorators
    from tests.unit.test_decorators_discovery_full_coverage import (
        TestDecoratorsDiscoveryFullCoverage,
    )
    from tests.unit.test_decorators_full_coverage import TestDecoratorsFullCoverage
    from tests.unit.test_deprecation_warnings import TestDeprecationWarnings
    from tests.unit.test_dispatcher_di import TestDispatcherDI
    from tests.unit.test_dispatcher_minimal import TestDispatcherMinimal
    from tests.unit.test_dispatcher_reliability import TestDispatcherReliability
    from tests.unit.test_enforcement import (
        TestAccessorMethodBan,
        TestBaseModelCoverage,
        TestClassPrefixScope,
        TestConstantsLayerRules,
        TestDetailSubstitution,
        TestEnforcementMode,
        TestFalsePositiveSkips,
        TestFieldRules,
        TestHasNestedNamespaceViaMro,
        TestModelClassRules,
        TestNamespaceInheritance,
        TestProjectDiscovery,
        TestProjectPrefixOverrides,
        TestProtocolsLayerRules,
        TestReportApi,
        TestSettingsInheritance,
        TestTypesLayerRules,
        TestUtilitiesLayerRules,
    )
    from tests.unit.test_enforcement_catalog import TestsFlextCoreEnforcementCatalog
    from tests.unit.test_enforcement_integration import (
        TestBadModuleFiresExpectedRules,
        TestCleanModuleEmitsNothing,
    )
    from tests.unit.test_enum_utilities_coverage_100 import TestEnumUtilitiesCoverage
    from tests.unit.test_exceptions import TestExceptions
    from tests.unit.test_handler_decorator_discovery import (
        TestHandlerDecoratorDiscovery,
    )
    from tests.unit.test_handlers import TestsFlextCoreFlextHandlers
    from tests.unit.test_handlers_full_coverage import TestHandlersFullCoverage
    from tests.unit.test_lazy_exports import TestsFlextCoreLazy
    from tests.unit.test_loggings_full_coverage import TestsFlextCoreLoggings
    from tests.unit.test_mixins import TestFlextMixinsNestedClasses
    from tests.unit.test_models import TestsFlextCoreModelsUnit
    from tests.unit.test_models_base_full_coverage import (
        TestsFlextCoreModelsBaseFullCoverage,
    )
    from tests.unit.test_models_container import TestModelsContainer
    from tests.unit.test_models_cqrs_full_coverage import TestModelsCqrsFullCoverage
    from tests.unit.test_models_entity_full_coverage import TestModelsEntityFullCoverage
    from tests.unit.test_models_project_metadata import TestModelsProjectMetadata
    from tests.unit.test_project_metadata_facade_access import TestFacadeFlatSsotAccess
    from tests.unit.test_protocols_project_metadata import (
        TestProjectClassStemDeriverProtocol,
        TestProjectMetadataReaderProtocol,
        TestProjectTierFacadeNamerProtocol,
    )
    from tests.unit.test_registry import TestRegistry
    from tests.unit.test_registry_full_coverage import TestRegistryFullCoverage
    from tests.unit.test_result import Testr
    from tests.unit.test_result_additional import TestResultAdditional
    from tests.unit.test_result_exception_carrying import (
        TestsFlextCoreResultExceptionCarrying,
    )
    from tests.unit.test_result_full_coverage import TestResultFullCoverage
    from tests.unit.test_runtime import TestFlextRuntime
    from tests.unit.test_service import (
        TestService,
        TestsFlextCoreServiceUserData,
        TestsFlextCoreServiceUserService,
    )
    from tests.unit.test_service_additional import TestServiceAdditional
    from tests.unit.test_service_bootstrap import TestsFlextCoreServiceBootstrap
    from tests.unit.test_service_coverage_100 import TestService100Coverage
    from tests.unit.test_settings import TestFlextSettings
    from tests.unit.test_settings_coverage import TestFlextSettingsCoverage
    from tests.unit.test_typings_full_coverage import TestTypingsFullCoverage
    from tests.unit.test_typings_new import TestFlextTypes
    from tests.unit.test_utilities import TestUtilitiesSmoke
    from tests.unit.test_utilities_collection_coverage_100 import (
        TestUtilitiesCollectionCoverage,
    )
    from tests.unit.test_utilities_context_full_coverage import (
        TestUtilitiesContextFullCoverage,
    )
    from tests.unit.test_utilities_coverage import TestUtilitiesCoverage
    from tests.unit.test_utilities_domain import TestUtilitiesDomain
    from tests.unit.test_utilities_domain_full_coverage import (
        TestUtilitiesDomainFullCoverage,
    )
    from tests.unit.test_utilities_generators_full_coverage import (
        TestUtilitiesGeneratorsFullCoverage,
    )
    from tests.unit.test_utilities_parser_full_coverage import (
        TestUtilitiesParserFullCoverage,
    )
    from tests.unit.test_utilities_project_metadata import (
        TestsFlextCoreUtilitiesProjectMetadata,
    )
    from tests.unit.test_utilities_reliability import TestFlextUtilitiesReliability
    from tests.unit.test_utilities_settings_coverage_100 import (
        TestFlextUtilitiesSettings,
    )
    from tests.unit.test_utilities_settings_full_coverage import (
        TestUtilitiesSettingsFullCoverage,
    )
    from tests.unit.test_utilities_text_full_coverage import (
        TestUtilitiesTextFullCoverage,
    )
    from tests.unit.test_utilities_type_checker_coverage_100 import (
        TestsFlextCoreUtilitiesTypeChecker,
    )
    from tests.unit.test_utilities_type_guards_coverage_100 import (
        TestUtilitiesTypeGuardsCoverage100,
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
                "TestsFlextCoreTypes",
                "t",
            ),
            ".unit._models.test_base": ("TestModelsBase",),
            ".unit._models.test_cqrs": ("TestModelsCQRS",),
            ".unit._models.test_entity": ("TestModelsEntity",),
            ".unit._models.test_exception_params": ("TestFlextModelsExceptionParams",),
            ".unit._utilities.test_guards": ("TestUtilitiesGuards",),
            ".unit._utilities.test_mapper": ("TestUtilitiesMapper",),
            ".unit.base": ("TestsFlextCoreServiceBase",),
            ".unit.test_beartype_engine": (
                "TestAliasContainsAny",
                "TestBeartypeClawCompatibility",
                "TestBeartypeConf",
                "TestContainsAny",
                "TestCountUnionMembers",
                "TestFacadeAccessibility",
                "TestForbiddenCollectionOrigin",
                "TestMatchesStrNoneUnion",
            ),
            ".unit.test_constants_new": ("TestFlextConstants",),
            ".unit.test_constants_project_metadata": (
                "TestsFlextCoreConstantsProjectMetadata",
            ),
            ".unit.test_container": ("TestFlextContainer",),
            ".unit.test_context": ("TestsFlextCoreContext",),
            ".unit.test_coverage_exceptions": ("TestCoverageExceptions",),
            ".unit.test_coverage_loggings": ("TestCoverageLoggings",),
            ".unit.test_decorators": ("TestFlextDecorators",),
            ".unit.test_decorators_discovery_full_coverage": (
                "TestDecoratorsDiscoveryFullCoverage",
            ),
            ".unit.test_decorators_full_coverage": ("TestDecoratorsFullCoverage",),
            ".unit.test_deprecation_warnings": ("TestDeprecationWarnings",),
            ".unit.test_dispatcher_di": ("TestDispatcherDI",),
            ".unit.test_dispatcher_minimal": ("TestDispatcherMinimal",),
            ".unit.test_dispatcher_reliability": ("TestDispatcherReliability",),
            ".unit.test_enforcement": (
                "TestAccessorMethodBan",
                "TestBaseModelCoverage",
                "TestClassPrefixScope",
                "TestConstantsLayerRules",
                "TestDetailSubstitution",
                "TestEnforcementMode",
                "TestFalsePositiveSkips",
                "TestFieldRules",
                "TestHasNestedNamespaceViaMro",
                "TestModelClassRules",
                "TestNamespaceInheritance",
                "TestProjectDiscovery",
                "TestProjectPrefixOverrides",
                "TestProtocolsLayerRules",
                "TestReportApi",
                "TestSettingsInheritance",
                "TestTypesLayerRules",
                "TestUtilitiesLayerRules",
            ),
            ".unit.test_enforcement_catalog": ("TestsFlextCoreEnforcementCatalog",),
            ".unit.test_enforcement_integration": (
                "TestBadModuleFiresExpectedRules",
                "TestCleanModuleEmitsNothing",
            ),
            ".unit.test_enum_utilities_coverage_100": ("TestEnumUtilitiesCoverage",),
            ".unit.test_exceptions": ("TestExceptions",),
            ".unit.test_handler_decorator_discovery": (
                "TestHandlerDecoratorDiscovery",
            ),
            ".unit.test_handlers": ("TestsFlextCoreFlextHandlers",),
            ".unit.test_handlers_full_coverage": ("TestHandlersFullCoverage",),
            ".unit.test_lazy_exports": ("TestsFlextCoreLazy",),
            ".unit.test_loggings_full_coverage": ("TestsFlextCoreLoggings",),
            ".unit.test_mixins": ("TestFlextMixinsNestedClasses",),
            ".unit.test_models": ("TestsFlextCoreModelsUnit",),
            ".unit.test_models_base_full_coverage": (
                "TestsFlextCoreModelsBaseFullCoverage",
            ),
            ".unit.test_models_container": ("TestModelsContainer",),
            ".unit.test_models_cqrs_full_coverage": ("TestModelsCqrsFullCoverage",),
            ".unit.test_models_entity_full_coverage": ("TestModelsEntityFullCoverage",),
            ".unit.test_models_project_metadata": ("TestModelsProjectMetadata",),
            ".unit.test_project_metadata_facade_access": ("TestFacadeFlatSsotAccess",),
            ".unit.test_protocols_project_metadata": (
                "TestProjectClassStemDeriverProtocol",
                "TestProjectMetadataReaderProtocol",
                "TestProjectTierFacadeNamerProtocol",
            ),
            ".unit.test_registry": ("TestRegistry",),
            ".unit.test_registry_full_coverage": ("TestRegistryFullCoverage",),
            ".unit.test_result": ("Testr",),
            ".unit.test_result_additional": ("TestResultAdditional",),
            ".unit.test_result_exception_carrying": (
                "TestsFlextCoreResultExceptionCarrying",
            ),
            ".unit.test_result_full_coverage": ("TestResultFullCoverage",),
            ".unit.test_runtime": ("TestFlextRuntime",),
            ".unit.test_service": (
                "TestService",
                "TestsFlextCoreServiceUserData",
                "TestsFlextCoreServiceUserService",
            ),
            ".unit.test_service_additional": ("TestServiceAdditional",),
            ".unit.test_service_bootstrap": ("TestsFlextCoreServiceBootstrap",),
            ".unit.test_service_coverage_100": ("TestService100Coverage",),
            ".unit.test_settings": ("TestFlextSettings",),
            ".unit.test_settings_coverage": ("TestFlextSettingsCoverage",),
            ".unit.test_typings_full_coverage": ("TestTypingsFullCoverage",),
            ".unit.test_typings_new": ("TestFlextTypes",),
            ".unit.test_utilities": ("TestUtilitiesSmoke",),
            ".unit.test_utilities_collection_coverage_100": (
                "TestUtilitiesCollectionCoverage",
            ),
            ".unit.test_utilities_context_full_coverage": (
                "TestUtilitiesContextFullCoverage",
            ),
            ".unit.test_utilities_coverage": ("TestUtilitiesCoverage",),
            ".unit.test_utilities_domain": ("TestUtilitiesDomain",),
            ".unit.test_utilities_domain_full_coverage": (
                "TestUtilitiesDomainFullCoverage",
            ),
            ".unit.test_utilities_generators_full_coverage": (
                "TestUtilitiesGeneratorsFullCoverage",
            ),
            ".unit.test_utilities_parser_full_coverage": (
                "TestUtilitiesParserFullCoverage",
            ),
            ".unit.test_utilities_project_metadata": (
                "TestsFlextCoreUtilitiesProjectMetadata",
            ),
            ".unit.test_utilities_reliability": ("TestFlextUtilitiesReliability",),
            ".unit.test_utilities_settings_coverage_100": (
                "TestFlextUtilitiesSettings",
            ),
            ".unit.test_utilities_settings_full_coverage": (
                "TestUtilitiesSettingsFullCoverage",
            ),
            ".unit.test_utilities_text_full_coverage": (
                "TestUtilitiesTextFullCoverage",
            ),
            ".unit.test_utilities_type_checker_coverage_100": (
                "TestsFlextCoreUtilitiesTypeChecker",
            ),
            ".unit.test_utilities_type_guards_coverage_100": (
                "TestUtilitiesTypeGuardsCoverage100",
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
    ),
    module_name=__name__,
)


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS)

__all__: list[str] = [
    "TestAccessorMethodBan",
    "TestAdvancedPatterns",
    "TestAliasContainsAny",
    "TestArchitecturalPatterns",
    "TestAutomatedArchitecture",
    "TestBadModuleFiresExpectedRules",
    "TestBaseModelCoverage",
    "TestBeartypeClawCompatibility",
    "TestBeartypeConf",
    "TestClassPrefixScope",
    "TestCleanModuleEmitsNothing",
    "TestCompleteFlextSystemIntegration",
    "TestConstantsLayerRules",
    "TestContainerMemory",
    "TestContainerPerformance",
    "TestContainsAny",
    "TestCountUnionMembers",
    "TestCoverageExceptions",
    "TestCoverageLoggings",
    "TestDecoratorsDiscoveryFullCoverage",
    "TestDecoratorsFullCoverage",
    "TestDeprecationWarnings",
    "TestDetailSubstitution",
    "TestDispatcherDI",
    "TestDispatcherMinimal",
    "TestDispatcherReliability",
    "TestDocumentedPatterns",
    "TestEnforcementMode",
    "TestEnumUtilitiesCoverage",
    "TestExamplesExecution",
    "TestExceptions",
    "TestFacadeAccessibility",
    "TestFacadeFlatSsotAccess",
    "TestFalsePositiveSkips",
    "TestFieldRules",
    "TestFlextConstants",
    "TestFlextContainer",
    "TestFlextDecorators",
    "TestFlextMixinsNestedClasses",
    "TestFlextModelsExceptionParams",
    "TestFlextRuntime",
    "TestFlextSettings",
    "TestFlextSettingsCoverage",
    "TestFlextSettingsSingletonIntegration",
    "TestFlextTypes",
    "TestFlextUtilitiesReliability",
    "TestFlextUtilitiesSettings",
    "TestForbiddenCollectionOrigin",
    "TestHandlerDecoratorDiscovery",
    "TestHandlersFullCoverage",
    "TestHasNestedNamespaceViaMro",
    "TestLazyPerformance",
    "TestLibraryIntegration",
    "TestMatchesStrNoneUnion",
    "TestMigrationValidation",
    "TestModelClassRules",
    "TestModelsBase",
    "TestModelsCQRS",
    "TestModelsContainer",
    "TestModelsCqrsFullCoverage",
    "TestModelsEntity",
    "TestModelsEntityFullCoverage",
    "TestModelsProjectMetadata",
    "TestNamespaceInheritance",
    "TestPatternsLogging",
    "TestPatternsTesting",
    "TestProjectClassStemDeriverProtocol",
    "TestProjectDiscovery",
    "TestProjectMetadataReaderProtocol",
    "TestProjectPrefixOverrides",
    "TestProjectTierFacadeNamerProtocol",
    "TestProtocolsLayerRules",
    "TestRegistry",
    "TestRegistryFullCoverage",
    "TestReportApi",
    "TestResultAdditional",
    "TestResultFullCoverage",
    "TestService",
    "TestService100Coverage",
    "TestServiceAdditional",
    "TestServiceResultProperty",
    "TestSettingsInheritance",
    "TestTypesLayerRules",
    "TestTypingsFullCoverage",
    "TestUtilitiesCollectionCoverage",
    "TestUtilitiesContextFullCoverage",
    "TestUtilitiesCoverage",
    "TestUtilitiesDomain",
    "TestUtilitiesDomainFullCoverage",
    "TestUtilitiesGeneratorsFullCoverage",
    "TestUtilitiesGuards",
    "TestUtilitiesLayerRules",
    "TestUtilitiesMapper",
    "TestUtilitiesParserFullCoverage",
    "TestUtilitiesSettingsFullCoverage",
    "TestUtilitiesSmoke",
    "TestUtilitiesTextFullCoverage",
    "TestUtilitiesTypeGuardsCoverage100",
    "Testr",
    "TestsFlextCoreConstants",
    "TestsFlextCoreConstantsDomain",
    "TestsFlextCoreConstantsErrors",
    "TestsFlextCoreConstantsFixtures",
    "TestsFlextCoreConstantsLoggings",
    "TestsFlextCoreConstantsOther",
    "TestsFlextCoreConstantsProjectMetadata",
    "TestsFlextCoreConstantsResult",
    "TestsFlextCoreConstantsServices",
    "TestsFlextCoreConstantsSettings",
    "TestsFlextCoreConstantsStrings",
    "TestsFlextCoreContext",
    "TestsFlextCoreEnforcementCatalog",
    "TestsFlextCoreFlextHandlers",
    "TestsFlextCoreLazy",
    "TestsFlextCoreLoggings",
    "TestsFlextCoreModels",
    "TestsFlextCoreModelsBaseFullCoverage",
    "TestsFlextCoreModelsMixins",
    "TestsFlextCoreModelsUnit",
    "TestsFlextCorePatternsCommands",
    "TestsFlextCoreProtocols",
    "TestsFlextCoreResultExceptionCarrying",
    "TestsFlextCoreServiceBase",
    "TestsFlextCoreServiceBootstrap",
    "TestsFlextCoreServiceIntegration",
    "TestsFlextCoreServiceUserData",
    "TestsFlextCoreServiceUserService",
    "TestsFlextCoreTypes",
    "TestsFlextCoreUtilities",
    "TestsFlextCoreUtilitiesProjectMetadata",
    "TestsFlextCoreUtilitiesTypeChecker",
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
