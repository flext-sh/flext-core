# AUTO-GENERATED FILE — Regenerate with: make gen
"""Tests package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import (
    build_lazy_import_map,
    install_lazy_exports,
    merge_lazy_imports,
)

if TYPE_CHECKING:
    from flext_tests import d, e, h, r, td, tf, tk, tm, tv, x

    from tests._constants.domain import TestsFlextConstantsDomain
    from tests._constants.errors import TestsFlextConstantsErrors
    from tests._constants.fixtures import TestsFlextConstantsFixtures
    from tests._constants.loggings import TestsFlextConstantsLoggings
    from tests._constants.other import TestsFlextConstantsOther
    from tests._constants.result import TestsFlextConstantsResult
    from tests._constants.services import TestsFlextConstantsServices
    from tests._constants.settings import TestsFlextConstantsSettings
    from tests._models._mixins.container import TestsFlextModelsContainerMixin
    from tests._models._mixins.core import TestsFlextModelsCoreMixin
    from tests._models._mixins.core_errors import TestsFlextModelsCoreErrorsMixin
    from tests._models._mixins.core_public import TestsFlextModelsCorePublicMixin
    from tests._models._mixins.core_state import TestsFlextModelsCoreStateMixin
    from tests._models._mixins.domain import TestsFlextModelsDomainMixin
    from tests._models._mixins.fixture_payloads import (
        TestsFlextModelsFixturePayloadsMixin,
    )
    from tests._models._mixins.fixture_suite import TestsFlextModelsFixtureSuiteMixin
    from tests._models._mixins.fixtures import TestsFlextModelsFixtureDictsMixin
    from tests._models._mixins.guards_mapper import TestsFlextModelsGuardsMapperMixin
    from tests._models._mixins.service_case_core import (
        TestsFlextModelsServiceCaseCoreMixin,
    )
    from tests._models._mixins.service_case_reliability import (
        TestsFlextModelsServiceCaseReliabilityMixin,
    )
    from tests._models._mixins.service_case_validation import (
        TestsFlextModelsServiceCaseValidationMixin,
    )
    from tests._models._mixins.service_cases import TestsFlextModelsServiceCasesMixin
    from tests._models._mixins.test_data import TestsFlextModelsTestDataMixin
    from tests._models._mixins.test_data_identity import (
        TestsFlextModelsTestDataIdentityMixin,
    )
    from tests._models._mixins.test_data_values import (
        TestsFlextModelsTestDataValuesMixin,
    )
    from tests._models.mixins import TestsFlextModelsMixins
    from tests._utilities.case_factories import TestsFlextUtilitiesCaseFactoriesMixin
    from tests._utilities.case_generators import TestsFlextUtilitiesCaseGeneratorsMixin
    from tests._utilities.case_service_factories import (
        TestsFlextUtilitiesCaseServiceFactoriesMixin,
    )
    from tests._utilities.contracts import TestsFlextUtilitiesContractsMixin
    from tests._utilities.dispatch import TestsFlextUtilitiesDispatchMixin
    from tests._utilities.parser_reliability import (
        TestsFlextUtilitiesParserReliabilityMixin,
    )
    from tests._utilities.parser_scenarios import (
        TestsFlextUtilitiesParserScenariosMixin,
    )
    from tests._utilities.railway import TestsFlextUtilitiesRailwayMixin
    from tests._utilities.railway_cases import TestsFlextUtilitiesRailwayCasesMixin
    from tests._utilities.railway_pipelines import (
        TestsFlextUtilitiesRailwayPipelinesMixin,
    )
    from tests._utilities.railway_services import (
        TestsFlextUtilitiesRailwayServicesMixin,
    )
    from tests._utilities.reliability_scenarios import (
        TestsFlextUtilitiesReliabilityScenariosMixin,
    )
    from tests._utilities.service_factories import (
        TestsFlextUtilitiesServiceFactoriesMixin,
    )
    from tests._utilities.services import TestsFlextUtilitiesServicesMixin
    from tests._utilities.user_factories import TestsFlextUtilitiesUserFactoriesMixin
    from tests._utilities.validation_factories import (
        TestsFlextUtilitiesValidationFactoriesMixin,
    )
    from tests._utilities.validation_network import (
        TestsFlextUtilitiesValidationNetworkScenarios,
    )
    from tests._utilities.validation_numeric import (
        TestsFlextUtilitiesValidationNumericScenarios,
    )
    from tests._utilities.validation_pattern import (
        TestsFlextUtilitiesValidationPatternScenarios,
    )
    from tests._utilities.validation_scenarios import (
        TestsFlextUtilitiesValidationScenariosMixin,
    )
    from tests._utilities.validation_string import (
        TestsFlextUtilitiesValidationStringScenarios,
    )
    from tests._utilities.validation_uri import (
        TestsFlextUtilitiesValidationUriScenarios,
    )
    from tests.base import TestsFlextServiceBase, s
    from tests.benchmark.test_container_memory import TestsFlextContainerMemory
    from tests.benchmark.test_container_performance import (
        TestsFlextContainerPerformance,
    )
    from tests.benchmark.test_lazy_performance import TestsFlextLazyPerformance
    from tests.constants import TestsFlextConstants, c
    from tests.fixtures.clean_module import (
        TestsFlextCleanConstants,
        TestsFlextCleanModels,
        TestsFlextCleanProtocols,
        TestsFlextCleanServiceBase,
    )
    from tests.integration.migration_validation_cases import (
        TestsFlextFlextMigrationApplicationCase,
    )
    from tests.integration.service_fixtures import (
        TestsFlextFlextServiceFixtures,
        TestsFlextLifecycleService,
        TestsFlextNotificationService,
        TestsFlextServiceConfig,
        TestsFlextUserQueryService,
        TestsFlextUserServiceEntity,
    )
    from tests.integration.service_lifecycle_cases import (
        TestsFlextFlextServiceLifecycleCases,
    )
    from tests.integration.settings_integration_factories import (
        TestsFlextFlextSettingsFactories,
        TestsFlextSettingsConfigTestCase,
        TestsFlextSettingsConfigTestFactories,
        TestsFlextSettingsThreadSafetyTest,
    )
    from tests.integration.settings_integration_precedence import (
        TestsFlextFlextSettingsPrecedenceCase,
    )
    from tests.integration.system_integration_cases import (
        TestsFlextFlextSystemWorkflowCases,
    )
    from tests.integration.test_architecture import TestsFlextCoreArchitecture
    from tests.integration.test_documented_patterns import (
        TestsFlextCoreDocumentedPatterns,
    )
    from tests.integration.test_examples_execution import TestsFlextExamplesExecution
    from tests.integration.test_integration import TestsFlextCoreIntegration
    from tests.integration.test_migration_validation import (
        TestsFlextCoreMigrationValidation,
    )
    from tests.integration.test_service import TestsFlextCoreService
    from tests.integration.test_settings_integration import (
        TestsFlextSettingsIntegration,
    )
    from tests.integration.test_system import TestsFlextCoreSystem
    from tests.models import TestsFlextModels, m
    from tests.protocols import TestsFlextProtocols, p
    from tests.typings import TestsFlextTypes, t
    from tests.unit._models.test_base import Sample, SampleValue, TestsFlextCoreBase
    from tests.unit._models.test_cqrs import TestsFlextCoreCqrs
    from tests.unit._models.test_enforcement_sources import (
        TestsFlextCoreEnforcementSources,
    )
    from tests.unit._models.test_entity import TestsFlextCoreEntity
    from tests.unit._models.test_exception_params_core import (
        TestsFlextModelsExceptionParamsCore,
    )
    from tests.unit._models.test_exception_params_operations import (
        TestsFlextCoreExceptionParamsOperations,
    )
    from tests.unit._models.test_exception_params_resources import (
        TestsFlextModelsExceptionParamsResources,
    )
    from tests.unit._utilities.test_guards import TestsFlextCoreGuards
    from tests.unit._utilities.test_mapper import TestsFlextCoreMapper
    from tests.unit.test_beartype_engine import TestsFlextCoreBeartypeEngine
    from tests.unit.test_beartype_engine_annotations import (
        TestsFlextBeartypeEngineAnnotations,
    )
    from tests.unit.test_beartype_engine_claw_packages import (
        TestsFlextCoreBeartypeEngineClawPackages,
    )
    from tests.unit.test_beartype_engine_config import (
        TestsFlextCoreBeartypeEngineConfig,
    )
    from tests.unit.test_beartype_engine_import_hooks import (
        TestsFlextCoreBeartypeEngineImportHooks,
    )
    from tests.unit.test_beartype_engine_namespace_hooks import (
        TestsFlextBeartypeEngineNamespaceHooks,
    )
    from tests.unit.test_beartype_engine_runtime import (
        TestsFlextCoreBeartypeEngineRuntime,
    )
    from tests.unit.test_constants_new import TestsFlextConstantsNew
    from tests.unit.test_constants_project_metadata import (
        TestsFlextCoreConstantsProjectMetadata,
    )
    from tests.unit.test_container import TestsFlextCoreContainer
    from tests.unit.test_container_config import TestsFlextCoreContainerConfig
    from tests.unit.test_container_lifecycle import TestsFlextContainerLifecycle
    from tests.unit.test_container_properties import TestsFlextCoreContainerProperties
    from tests.unit.test_container_registration import (
        TestsFlextCoreContainerRegistration,
    )
    from tests.unit.test_container_resolution import TestsFlextContainerResolution
    from tests.unit.test_context import TestsFlextCoreContext
    from tests.unit.test_coverage_loggings import TestsFlextCoverageLoggings
    from tests.unit.test_decorators import TestsFlextCoreDecorators
    from tests.unit.test_decorators_combined import TestsFlextCoreDecoratorsCombined
    from tests.unit.test_decorators_discovery_full_coverage import (
        TestsFlextDecoratorsDiscovery,
    )
    from tests.unit.test_decorators_injection_logging import (
        TestsFlextCoreDecoratorsInjectionLogging,
    )
    from tests.unit.test_decorators_railway_retry import (
        TestsFlextCoreDecoratorsRailwayRetry,
    )
    from tests.unit.test_deprecation_warnings import TestsFlextCoreDeprecationWarnings
    from tests.unit.test_dispatcher import TestsFlextCoreDispatcher
    from tests.unit.test_enforcement import TestsFlextCoreEnforcement
    from tests.unit.test_enforcement_accessors import TestsFlextCoreEnforcementAccessors
    from tests.unit.test_enforcement_apt_hooks import TestsFlextEnforcementAptHooks
    from tests.unit.test_enforcement_catalog import TestsFlextEnforcementCatalog
    from tests.unit.test_enforcement_integration import TestsFlextEnforcementIntegration
    from tests.unit.test_enforcement_layers import TestsFlextCoreEnforcementLayers
    from tests.unit.test_enforcement_models import TestsFlextEnforcementModels
    from tests.unit.test_enforcement_namespace import TestsFlextCoreEnforcementNamespace
    from tests.unit.test_enforcement_namespace_part_01 import (
        TestsFlextCoreEnforcementNamespacePart01,
    )
    from tests.unit.test_enforcement_namespace_part_02 import (
        TestsFlextCoreEnforcementNamespacePart02,
    )
    from tests.unit.test_enforcement_plugin import TestsFlextCoreEnforcementPlugin
    from tests.unit.test_enforcement_reports import TestsFlextCoreEnforcementReports
    from tests.unit.test_enforcement_warning_visibility import (
        TestsFlextCoreEnforcementWarningVisibility,
    )
    from tests.unit.test_enum_utilities_coverage_100 import TestsFlextCoreEnumUtilities
    from tests.unit.test_exceptions import TestsFlextCoreExceptions
    from tests.unit.test_exceptions_base import TestsFlextCoreExceptionsBase
    from tests.unit.test_exceptions_public_metrics import (
        TestsFlextCoreExceptionsPublicMetrics,
    )
    from tests.unit.test_exceptions_structured_contracts import (
        TestsFlextCoreExceptionsStructuredContracts,
    )
    from tests.unit.test_exceptions_typed_metrics import (
        TestsFlextCoreExceptionsTypedMetrics,
    )
    from tests.unit.test_handler_decorator_discovery import (
        TestsFlextCoreHandlerDecoratorDiscovery,
    )
    from tests.unit.test_handler_decorator_edges import TestsFlextHandlerDecoratorEdges
    from tests.unit.test_handler_decorator_metadata import (
        TestsFlextHandlerDecoratorMetadata,
    )
    from tests.unit.test_handler_discovery_class import (
        TestsFlextCoreHandlerDiscoveryClass,
    )
    from tests.unit.test_handler_discovery_module import (
        TestsFlextHandlerDiscoveryModule,
    )
    from tests.unit.test_handlers_dispatch import TestsFlextHandlersDispatch
    from tests.unit.test_handlers_factory import TestsFlextCoreHandlersFactory
    from tests.unit.test_handlers_lifecycle import TestsFlextHandlersLifecycle
    from tests.unit.test_handlers_properties import TestsFlextCoreHandlersProperties
    from tests.unit.test_handlers_validation_context import (
        TestsFlextCoreHandlersValidationContext,
    )
    from tests.unit.test_lazy_exports import TestsFlextCoreLazyExports
    from tests.unit.test_lazy_exports_merge import TestsFlextCoreLazyExportsMerge
    from tests.unit.test_loggings_full_coverage import TestsFlextLoggings
    from tests.unit.test_mixins import TestsFlextMixins
    from tests.unit.test_models import TestsFlextCoreModels
    from tests.unit.test_models_base_full_coverage import (
        TestsFlextCoreModelsBaseFullCoverage,
    )
    from tests.unit.test_models_container import TestsFlextCoreModelsContainer
    from tests.unit.test_models_cqrs_full_coverage import TestsFlextCoreModelsCqrs
    from tests.unit.test_models_project_metadata import TestsFlextModelsProjectMetadata
    from tests.unit.test_project_metadata_facade_access import (
        TestsFlextFacadeFlatSsotAccess,
    )
    from tests.unit.test_public_api_contract import TestsFlextCorePublicApiContract
    from tests.unit.test_registry import TestsFlextCoreRegistry
    from tests.unit.test_result import TestsFlextCoreResult
    from tests.unit.test_result_callables_fold import TestsFlextCoreResultCallablesFold
    from tests.unit.test_result_chain_helpers import TestsFlextCoreResultChainHelpers
    from tests.unit.test_result_exception_failures import (
        TestsFlextCoreResultExceptionFailures,
    )
    from tests.unit.test_result_exception_mapping import (
        TestsFlextCoreResultExceptionMapping,
    )
    from tests.unit.test_result_exception_safe_callable import (
        TestsFlextCoreResultExceptionSafeCallable,
    )
    from tests.unit.test_result_exception_traverse_validation import (
        TestsFlextCoreResultExceptionTraverseValidation,
    )
    from tests.unit.test_result_laws import TestsFlextCoreResultLaws
    from tests.unit.test_result_operations import TestsFlextResultOperations
    from tests.unit.test_result_recent_behaviors import (
        TestsFlextCoreResultRecentBehaviors,
    )
    from tests.unit.test_result_transforms import TestsFlextResultTransforms
    from tests.unit.test_result_traverse_resource import (
        TestsFlextResultTraverseResource,
    )
    from tests.unit.test_runtime import TestsFlextCoreRuntime
    from tests.unit.test_service import TestsFlextService
    from tests.unit.test_service_bootstrap import TestsFlextCoreServiceBootstrap
    from tests.unit.test_settings import TestsFlextCoreSettings
    from tests.unit.test_settings_validation_alias import (
        TestsFlextCoreSettingsValidationAlias,
    )
    from tests.unit.test_typings_aliases import TestsFlextCoreTypingsAliases
    from tests.unit.test_typings_containers import TestsFlextCoreTypingsContainers
    from tests.unit.test_typings_new import TestsFlextCoreTypingsNew
    from tests.unit.test_typings_validation_numbers import (
        TestsFlextCoreTypingsValidationNumbers,
    )
    from tests.unit.test_typings_validation_scalars import (
        TestsFlextCoreTypingsValidationScalars,
    )
    from tests.unit.test_utilities import TestsFlextCoreUtilities
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
    from tests.unit.test_utilities_project_metadata_config import (
        TestsFlextCoreUtilitiesProjectMetadataConfig,
    )
    from tests.unit.test_utilities_project_metadata_read import (
        TestsFlextUtilitiesProjectMetadataRead,
    )
    from tests.unit.test_utilities_pydantic_coverage_100 import (
        TestsFlextUtilitiesPydantic,
    )
    from tests.unit.test_utilities_reliability import TestsFlextCoreUtilitiesReliability
    from tests.unit.test_utilities_runtime_violation_registry_coverage_100 import (
        TestsFlextCoreUtilitiesRuntimeViolationRegistry,
    )
    from tests.unit.test_utilities_settings_coverage_100 import (
        TestsFlextCoreUtilitiesSettings,
    )
    from tests.unit.test_utilities_text_full_coverage import TestsFlextUtilitiesText
    from tests.unit.test_utilities_type_guards_coverage_100 import (
        TestsFlextCoreUtilitiesTypeGuards,
    )
    from tests.unit.test_version import TestsFlextCoreVersion
    from tests.utilities import TestsFlextUtilities, u
_LAZY_IMPORTS = merge_lazy_imports(
    (
        "._constants",
        "._models",
        "._utilities",
        ".benchmark",
        ".fixtures",
        ".integration",
        ".unit",
    ),
    build_lazy_import_map(
        {
            "._constants": ("_constants",),
            "._constants.domain": ("TestsFlextConstantsDomain",),
            "._constants.errors": ("TestsFlextConstantsErrors",),
            "._constants.fixtures": ("TestsFlextConstantsFixtures",),
            "._constants.loggings": ("TestsFlextConstantsLoggings",),
            "._constants.other": ("TestsFlextConstantsOther",),
            "._constants.result": ("TestsFlextConstantsResult",),
            "._constants.services": ("TestsFlextConstantsServices",),
            "._constants.settings": ("TestsFlextConstantsSettings",),
            "._models": ("_models",),
            "._models._mixins.container": ("TestsFlextModelsContainerMixin",),
            "._models._mixins.core": ("TestsFlextModelsCoreMixin",),
            "._models._mixins.core_errors": ("TestsFlextModelsCoreErrorsMixin",),
            "._models._mixins.core_public": ("TestsFlextModelsCorePublicMixin",),
            "._models._mixins.core_state": ("TestsFlextModelsCoreStateMixin",),
            "._models._mixins.domain": ("TestsFlextModelsDomainMixin",),
            "._models._mixins.fixture_payloads": (
                "TestsFlextModelsFixturePayloadsMixin",
            ),
            "._models._mixins.fixture_suite": ("TestsFlextModelsFixtureSuiteMixin",),
            "._models._mixins.fixtures": ("TestsFlextModelsFixtureDictsMixin",),
            "._models._mixins.guards_mapper": ("TestsFlextModelsGuardsMapperMixin",),
            "._models._mixins.service_case_core": (
                "TestsFlextModelsServiceCaseCoreMixin",
            ),
            "._models._mixins.service_case_reliability": (
                "TestsFlextModelsServiceCaseReliabilityMixin",
            ),
            "._models._mixins.service_case_validation": (
                "TestsFlextModelsServiceCaseValidationMixin",
            ),
            "._models._mixins.service_cases": ("TestsFlextModelsServiceCasesMixin",),
            "._models._mixins.test_data": ("TestsFlextModelsTestDataMixin",),
            "._models._mixins.test_data_identity": (
                "TestsFlextModelsTestDataIdentityMixin",
            ),
            "._models._mixins.test_data_values": (
                "TestsFlextModelsTestDataValuesMixin",
            ),
            "._models.mixins": ("TestsFlextModelsMixins",),
            "._utilities": ("_utilities",),
            "._utilities.case_factories": ("TestsFlextUtilitiesCaseFactoriesMixin",),
            "._utilities.case_generators": ("TestsFlextUtilitiesCaseGeneratorsMixin",),
            "._utilities.case_service_factories": (
                "TestsFlextUtilitiesCaseServiceFactoriesMixin",
            ),
            "._utilities.contracts": ("TestsFlextUtilitiesContractsMixin",),
            "._utilities.dispatch": ("TestsFlextUtilitiesDispatchMixin",),
            "._utilities.parser_reliability": (
                "TestsFlextUtilitiesParserReliabilityMixin",
            ),
            "._utilities.parser_scenarios": (
                "TestsFlextUtilitiesParserScenariosMixin",
            ),
            "._utilities.railway": ("TestsFlextUtilitiesRailwayMixin",),
            "._utilities.railway_cases": ("TestsFlextUtilitiesRailwayCasesMixin",),
            "._utilities.railway_pipelines": (
                "TestsFlextUtilitiesRailwayPipelinesMixin",
            ),
            "._utilities.railway_services": (
                "TestsFlextUtilitiesRailwayServicesMixin",
            ),
            "._utilities.reliability_scenarios": (
                "TestsFlextUtilitiesReliabilityScenariosMixin",
            ),
            "._utilities.service_factories": (
                "TestsFlextUtilitiesServiceFactoriesMixin",
            ),
            "._utilities.services": ("TestsFlextUtilitiesServicesMixin",),
            "._utilities.user_factories": ("TestsFlextUtilitiesUserFactoriesMixin",),
            "._utilities.validation_factories": (
                "TestsFlextUtilitiesValidationFactoriesMixin",
            ),
            "._utilities.validation_network": (
                "TestsFlextUtilitiesValidationNetworkScenarios",
            ),
            "._utilities.validation_numeric": (
                "TestsFlextUtilitiesValidationNumericScenarios",
            ),
            "._utilities.validation_pattern": (
                "TestsFlextUtilitiesValidationPatternScenarios",
            ),
            "._utilities.validation_scenarios": (
                "TestsFlextUtilitiesValidationScenariosMixin",
            ),
            "._utilities.validation_string": (
                "TestsFlextUtilitiesValidationStringScenarios",
            ),
            "._utilities.validation_uri": (
                "TestsFlextUtilitiesValidationUriScenarios",
            ),
            ".base": (
                "TestsFlextServiceBase",
                "s",
            ),
            ".benchmark": ("benchmark",),
            ".benchmark.test_container_memory": ("TestsFlextContainerMemory",),
            ".benchmark.test_container_performance": (
                "TestsFlextContainerPerformance",
            ),
            ".benchmark.test_lazy_performance": ("TestsFlextLazyPerformance",),
            ".conftest": ("conftest",),
            ".constants": (
                "TestsFlextConstants",
                "c",
            ),
            ".fixtures": ("fixtures",),
            ".fixtures.clean_module": (
                "TestsFlextCleanConstants",
                "TestsFlextCleanModels",
                "TestsFlextCleanProtocols",
                "TestsFlextCleanServiceBase",
            ),
            ".integration": ("integration",),
            ".integration.migration_validation_cases": (
                "TestsFlextFlextMigrationApplicationCase",
            ),
            ".integration.service_fixtures": (
                "TestsFlextFlextServiceFixtures",
                "TestsFlextLifecycleService",
                "TestsFlextNotificationService",
                "TestsFlextServiceConfig",
                "TestsFlextUserQueryService",
                "TestsFlextUserServiceEntity",
            ),
            ".integration.service_lifecycle_cases": (
                "TestsFlextFlextServiceLifecycleCases",
            ),
            ".integration.settings_integration_factories": (
                "TestsFlextFlextSettingsFactories",
                "TestsFlextSettingsConfigTestCase",
                "TestsFlextSettingsConfigTestFactories",
                "TestsFlextSettingsThreadSafetyTest",
            ),
            ".integration.settings_integration_precedence": (
                "TestsFlextFlextSettingsPrecedenceCase",
            ),
            ".integration.system_integration_cases": (
                "TestsFlextFlextSystemWorkflowCases",
            ),
            ".integration.test_architecture": ("TestsFlextCoreArchitecture",),
            ".integration.test_documented_patterns": (
                "TestsFlextCoreDocumentedPatterns",
            ),
            ".integration.test_examples_execution": ("TestsFlextExamplesExecution",),
            ".integration.test_integration": ("TestsFlextCoreIntegration",),
            ".integration.test_migration_validation": (
                "TestsFlextCoreMigrationValidation",
            ),
            ".integration.test_service": ("TestsFlextCoreService",),
            ".integration.test_settings_integration": (
                "TestsFlextSettingsIntegration",
            ),
            ".integration.test_system": ("TestsFlextCoreSystem",),
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
            ".unit": ("unit",),
            ".unit._models.test_base": (
                "Sample",
                "SampleValue",
                "TestsFlextCoreBase",
            ),
            ".unit._models.test_cqrs": ("TestsFlextCoreCqrs",),
            ".unit._models.test_enforcement_sources": (
                "TestsFlextCoreEnforcementSources",
            ),
            ".unit._models.test_entity": ("TestsFlextCoreEntity",),
            ".unit._models.test_exception_params_core": (
                "TestsFlextModelsExceptionParamsCore",
            ),
            ".unit._models.test_exception_params_operations": (
                "TestsFlextCoreExceptionParamsOperations",
            ),
            ".unit._models.test_exception_params_resources": (
                "TestsFlextModelsExceptionParamsResources",
            ),
            ".unit._utilities.test_guards": ("TestsFlextCoreGuards",),
            ".unit._utilities.test_mapper": ("TestsFlextCoreMapper",),
            ".unit.test_beartype_engine": ("TestsFlextCoreBeartypeEngine",),
            ".unit.test_beartype_engine_annotations": (
                "TestsFlextBeartypeEngineAnnotations",
            ),
            ".unit.test_beartype_engine_claw_packages": (
                "TestsFlextCoreBeartypeEngineClawPackages",
            ),
            ".unit.test_beartype_engine_config": (
                "TestsFlextCoreBeartypeEngineConfig",
            ),
            ".unit.test_beartype_engine_import_hooks": (
                "TestsFlextCoreBeartypeEngineImportHooks",
            ),
            ".unit.test_beartype_engine_namespace_hooks": (
                "TestsFlextBeartypeEngineNamespaceHooks",
            ),
            ".unit.test_beartype_engine_runtime": (
                "TestsFlextCoreBeartypeEngineRuntime",
            ),
            ".unit.test_constants_new": ("TestsFlextConstantsNew",),
            ".unit.test_constants_project_metadata": (
                "TestsFlextCoreConstantsProjectMetadata",
            ),
            ".unit.test_container": ("TestsFlextCoreContainer",),
            ".unit.test_container_config": ("TestsFlextCoreContainerConfig",),
            ".unit.test_container_lifecycle": ("TestsFlextContainerLifecycle",),
            ".unit.test_container_properties": ("TestsFlextCoreContainerProperties",),
            ".unit.test_container_registration": (
                "TestsFlextCoreContainerRegistration",
            ),
            ".unit.test_container_resolution": ("TestsFlextContainerResolution",),
            ".unit.test_context": ("TestsFlextCoreContext",),
            ".unit.test_coverage_loggings": ("TestsFlextCoverageLoggings",),
            ".unit.test_decorators": ("TestsFlextCoreDecorators",),
            ".unit.test_decorators_combined": ("TestsFlextCoreDecoratorsCombined",),
            ".unit.test_decorators_discovery_full_coverage": (
                "TestsFlextDecoratorsDiscovery",
            ),
            ".unit.test_decorators_injection_logging": (
                "TestsFlextCoreDecoratorsInjectionLogging",
            ),
            ".unit.test_decorators_railway_retry": (
                "TestsFlextCoreDecoratorsRailwayRetry",
            ),
            ".unit.test_deprecation_warnings": ("TestsFlextCoreDeprecationWarnings",),
            ".unit.test_dispatcher": ("TestsFlextCoreDispatcher",),
            ".unit.test_enforcement": ("TestsFlextCoreEnforcement",),
            ".unit.test_enforcement_accessors": ("TestsFlextCoreEnforcementAccessors",),
            ".unit.test_enforcement_apt_hooks": ("TestsFlextEnforcementAptHooks",),
            ".unit.test_enforcement_catalog": ("TestsFlextEnforcementCatalog",),
            ".unit.test_enforcement_integration": ("TestsFlextEnforcementIntegration",),
            ".unit.test_enforcement_layers": ("TestsFlextCoreEnforcementLayers",),
            ".unit.test_enforcement_models": ("TestsFlextEnforcementModels",),
            ".unit.test_enforcement_namespace": ("TestsFlextCoreEnforcementNamespace",),
            ".unit.test_enforcement_namespace_part_01": (
                "TestsFlextCoreEnforcementNamespacePart01",
            ),
            ".unit.test_enforcement_namespace_part_02": (
                "TestsFlextCoreEnforcementNamespacePart02",
            ),
            ".unit.test_enforcement_plugin": ("TestsFlextCoreEnforcementPlugin",),
            ".unit.test_enforcement_reports": ("TestsFlextCoreEnforcementReports",),
            ".unit.test_enforcement_warning_visibility": (
                "TestsFlextCoreEnforcementWarningVisibility",
            ),
            ".unit.test_enum_utilities_coverage_100": ("TestsFlextCoreEnumUtilities",),
            ".unit.test_exceptions": ("TestsFlextCoreExceptions",),
            ".unit.test_exceptions_base": ("TestsFlextCoreExceptionsBase",),
            ".unit.test_exceptions_public_metrics": (
                "TestsFlextCoreExceptionsPublicMetrics",
            ),
            ".unit.test_exceptions_structured_contracts": (
                "TestsFlextCoreExceptionsStructuredContracts",
            ),
            ".unit.test_exceptions_typed_metrics": (
                "TestsFlextCoreExceptionsTypedMetrics",
            ),
            ".unit.test_handler_decorator_discovery": (
                "TestsFlextCoreHandlerDecoratorDiscovery",
            ),
            ".unit.test_handler_decorator_edges": ("TestsFlextHandlerDecoratorEdges",),
            ".unit.test_handler_decorator_metadata": (
                "TestsFlextHandlerDecoratorMetadata",
            ),
            ".unit.test_handler_discovery_class": (
                "TestsFlextCoreHandlerDiscoveryClass",
            ),
            ".unit.test_handler_discovery_module": (
                "TestsFlextHandlerDiscoveryModule",
            ),
            ".unit.test_handlers_dispatch": ("TestsFlextHandlersDispatch",),
            ".unit.test_handlers_factory": ("TestsFlextCoreHandlersFactory",),
            ".unit.test_handlers_lifecycle": ("TestsFlextHandlersLifecycle",),
            ".unit.test_handlers_properties": ("TestsFlextCoreHandlersProperties",),
            ".unit.test_handlers_validation_context": (
                "TestsFlextCoreHandlersValidationContext",
            ),
            ".unit.test_lazy_exports": ("TestsFlextCoreLazyExports",),
            ".unit.test_lazy_exports_merge": ("TestsFlextCoreLazyExportsMerge",),
            ".unit.test_loggings_full_coverage": ("TestsFlextLoggings",),
            ".unit.test_mixins": ("TestsFlextMixins",),
            ".unit.test_models": ("TestsFlextCoreModels",),
            ".unit.test_models_base_full_coverage": (
                "TestsFlextCoreModelsBaseFullCoverage",
            ),
            ".unit.test_models_container": ("TestsFlextCoreModelsContainer",),
            ".unit.test_models_cqrs_full_coverage": ("TestsFlextCoreModelsCqrs",),
            ".unit.test_models_project_metadata": ("TestsFlextModelsProjectMetadata",),
            ".unit.test_project_metadata_facade_access": (
                "TestsFlextFacadeFlatSsotAccess",
            ),
            ".unit.test_public_api_contract": ("TestsFlextCorePublicApiContract",),
            ".unit.test_registry": ("TestsFlextCoreRegistry",),
            ".unit.test_result": ("TestsFlextCoreResult",),
            ".unit.test_result_callables_fold": ("TestsFlextCoreResultCallablesFold",),
            ".unit.test_result_chain_helpers": ("TestsFlextCoreResultChainHelpers",),
            ".unit.test_result_exception_failures": (
                "TestsFlextCoreResultExceptionFailures",
            ),
            ".unit.test_result_exception_mapping": (
                "TestsFlextCoreResultExceptionMapping",
            ),
            ".unit.test_result_exception_safe_callable": (
                "TestsFlextCoreResultExceptionSafeCallable",
            ),
            ".unit.test_result_exception_traverse_validation": (
                "TestsFlextCoreResultExceptionTraverseValidation",
            ),
            ".unit.test_result_laws": ("TestsFlextCoreResultLaws",),
            ".unit.test_result_operations": ("TestsFlextResultOperations",),
            ".unit.test_result_recent_behaviors": (
                "TestsFlextCoreResultRecentBehaviors",
            ),
            ".unit.test_result_transforms": ("TestsFlextResultTransforms",),
            ".unit.test_result_traverse_resource": (
                "TestsFlextResultTraverseResource",
            ),
            ".unit.test_runtime": ("TestsFlextCoreRuntime",),
            ".unit.test_service": ("TestsFlextService",),
            ".unit.test_service_bootstrap": ("TestsFlextCoreServiceBootstrap",),
            ".unit.test_settings": ("TestsFlextCoreSettings",),
            ".unit.test_settings_validation_alias": (
                "TestsFlextCoreSettingsValidationAlias",
            ),
            ".unit.test_typings_aliases": ("TestsFlextCoreTypingsAliases",),
            ".unit.test_typings_containers": ("TestsFlextCoreTypingsContainers",),
            ".unit.test_typings_new": ("TestsFlextCoreTypingsNew",),
            ".unit.test_typings_validation_numbers": (
                "TestsFlextCoreTypingsValidationNumbers",
            ),
            ".unit.test_typings_validation_scalars": (
                "TestsFlextCoreTypingsValidationScalars",
            ),
            ".unit.test_utilities": ("TestsFlextCoreUtilities",),
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
            ".unit.test_utilities_project_metadata_config": (
                "TestsFlextCoreUtilitiesProjectMetadataConfig",
            ),
            ".unit.test_utilities_project_metadata_read": (
                "TestsFlextUtilitiesProjectMetadataRead",
            ),
            ".unit.test_utilities_pydantic_coverage_100": (
                "TestsFlextUtilitiesPydantic",
            ),
            ".unit.test_utilities_reliability": ("TestsFlextCoreUtilitiesReliability",),
            ".unit.test_utilities_runtime_violation_registry_coverage_100": (
                "TestsFlextCoreUtilitiesRuntimeViolationRegistry",
            ),
            ".unit.test_utilities_settings_coverage_100": (
                "TestsFlextCoreUtilitiesSettings",
            ),
            ".unit.test_utilities_text_full_coverage": ("TestsFlextUtilitiesText",),
            ".unit.test_utilities_type_guards_coverage_100": (
                "TestsFlextCoreUtilitiesTypeGuards",
            ),
            ".unit.test_version": ("TestsFlextCoreVersion",),
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


install_lazy_exports(
    __name__,
    globals(),
    _LAZY_IMPORTS,
    publish_all=False,
)
