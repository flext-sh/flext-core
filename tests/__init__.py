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
    from flext_tests import (
        d as d,
        e as e,
        h as h,
        r as r,
        td as td,
        tf as tf,
        tk as tk,
        tm as tm,
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
    from tests._models._mixins.container import (
        TestsFlextModelsContainerMixin as TestsFlextModelsContainerMixin,
    )
    from tests._models._mixins.core import (
        TestsFlextModelsCoreMixin as TestsFlextModelsCoreMixin,
    )
    from tests._models._mixins.core_errors import (
        TestsFlextModelsCoreErrorsMixin as TestsFlextModelsCoreErrorsMixin,
    )
    from tests._models._mixins.core_public import (
        TestsFlextModelsCorePublicMixin as TestsFlextModelsCorePublicMixin,
    )
    from tests._models._mixins.core_state import (
        TestsFlextModelsCoreStateMixin as TestsFlextModelsCoreStateMixin,
    )
    from tests._models._mixins.domain import (
        TestsFlextModelsDomainMixin as TestsFlextModelsDomainMixin,
    )
    from tests._models._mixins.fixture_payloads import (
        TestsFlextModelsFixturePayloadsMixin as TestsFlextModelsFixturePayloadsMixin,
    )
    from tests._models._mixins.fixture_suite import (
        TestsFlextModelsFixtureSuiteMixin as TestsFlextModelsFixtureSuiteMixin,
    )
    from tests._models._mixins.fixtures import (
        TestsFlextModelsFixtureDictsMixin as TestsFlextModelsFixtureDictsMixin,
    )
    from tests._models._mixins.guards_mapper import (
        TestsFlextModelsGuardsMapperMixin as TestsFlextModelsGuardsMapperMixin,
    )
    from tests._models._mixins.service_case_core import (
        TestsFlextModelsServiceCaseCoreMixin as TestsFlextModelsServiceCaseCoreMixin,
    )
    from tests._models._mixins.service_case_reliability import (
        TestsFlextModelsServiceCaseReliabilityMixin as TestsFlextModelsServiceCaseReliabilityMixin,
    )
    from tests._models._mixins.service_case_validation import (
        TestsFlextModelsServiceCaseValidationMixin as TestsFlextModelsServiceCaseValidationMixin,
    )
    from tests._models._mixins.service_cases import (
        TestsFlextModelsServiceCasesMixin as TestsFlextModelsServiceCasesMixin,
    )
    from tests._models._mixins.test_data import (
        TestsFlextModelsTestDataMixin as TestsFlextModelsTestDataMixin,
    )
    from tests._models._mixins.test_data_identity import (
        TestsFlextModelsTestDataIdentityMixin as TestsFlextModelsTestDataIdentityMixin,
    )
    from tests._models._mixins.test_data_values import (
        TestsFlextModelsTestDataValuesMixin as TestsFlextModelsTestDataValuesMixin,
    )
    from tests._models.mixins import TestsFlextModelsMixins as TestsFlextModelsMixins
    from tests._utilities.case_factories import (
        TestsFlextUtilitiesCaseFactoriesMixin as TestsFlextUtilitiesCaseFactoriesMixin,
    )
    from tests._utilities.case_generators import (
        TestsFlextUtilitiesCaseGeneratorsMixin as TestsFlextUtilitiesCaseGeneratorsMixin,
    )
    from tests._utilities.case_service_factories import (
        TestsFlextUtilitiesCaseServiceFactoriesMixin as TestsFlextUtilitiesCaseServiceFactoriesMixin,
    )
    from tests._utilities.contracts import (
        TestsFlextUtilitiesContractsMixin as TestsFlextUtilitiesContractsMixin,
    )
    from tests._utilities.dispatch import (
        TestsFlextUtilitiesDispatchMixin as TestsFlextUtilitiesDispatchMixin,
    )
    from tests._utilities.parser_reliability import (
        TestsFlextUtilitiesParserReliabilityMixin as TestsFlextUtilitiesParserReliabilityMixin,
    )
    from tests._utilities.parser_scenarios import (
        TestsFlextUtilitiesParserScenariosMixin as TestsFlextUtilitiesParserScenariosMixin,
    )
    from tests._utilities.railway import (
        TestsFlextUtilitiesRailwayMixin as TestsFlextUtilitiesRailwayMixin,
    )
    from tests._utilities.railway_cases import (
        TestsFlextUtilitiesRailwayCasesMixin as TestsFlextUtilitiesRailwayCasesMixin,
    )
    from tests._utilities.railway_pipelines import (
        TestsFlextUtilitiesRailwayPipelinesMixin as TestsFlextUtilitiesRailwayPipelinesMixin,
    )
    from tests._utilities.railway_services import (
        TestsFlextUtilitiesRailwayServicesMixin as TestsFlextUtilitiesRailwayServicesMixin,
    )
    from tests._utilities.reliability_scenarios import (
        TestsFlextUtilitiesReliabilityScenariosMixin as TestsFlextUtilitiesReliabilityScenariosMixin,
    )
    from tests._utilities.service_factories import (
        TestsFlextUtilitiesServiceFactoriesMixin as TestsFlextUtilitiesServiceFactoriesMixin,
    )
    from tests._utilities.services import (
        TestsFlextUtilitiesServicesMixin as TestsFlextUtilitiesServicesMixin,
    )
    from tests._utilities.user_factories import (
        TestsFlextUtilitiesUserFactoriesMixin as TestsFlextUtilitiesUserFactoriesMixin,
    )
    from tests._utilities.validation_factories import (
        TestsFlextUtilitiesValidationFactoriesMixin as TestsFlextUtilitiesValidationFactoriesMixin,
    )
    from tests._utilities.validation_network import (
        TestsFlextUtilitiesValidationNetworkScenarios as TestsFlextUtilitiesValidationNetworkScenarios,
    )
    from tests._utilities.validation_numeric import (
        TestsFlextUtilitiesValidationNumericScenarios as TestsFlextUtilitiesValidationNumericScenarios,
    )
    from tests._utilities.validation_pattern import (
        TestsFlextUtilitiesValidationPatternScenarios as TestsFlextUtilitiesValidationPatternScenarios,
    )
    from tests._utilities.validation_scenarios import (
        TestsFlextUtilitiesValidationScenariosMixin as TestsFlextUtilitiesValidationScenariosMixin,
    )
    from tests._utilities.validation_string import (
        TestsFlextUtilitiesValidationStringScenarios as TestsFlextUtilitiesValidationStringScenarios,
    )
    from tests._utilities.validation_uri import (
        TestsFlextUtilitiesValidationUriScenarios as TestsFlextUtilitiesValidationUriScenarios,
    )
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
    from tests.fixtures.clean_module import (
        TestsFlextCleanConstants as TestsFlextCleanConstants,
        TestsFlextCleanModels as TestsFlextCleanModels,
        TestsFlextCleanProtocols as TestsFlextCleanProtocols,
        TestsFlextCleanServiceBase as TestsFlextCleanServiceBase,
    )
    from tests.integration.migration_validation_cases import (
        TestsFlextFlextMigrationApplicationCase as TestsFlextFlextMigrationApplicationCase,
    )
    from tests.integration.service_fixtures import (
        TestsFlextFlextServiceFixtures as TestsFlextFlextServiceFixtures,
        TestsFlextLifecycleService as TestsFlextLifecycleService,
        TestsFlextNotificationService as TestsFlextNotificationService,
        TestsFlextServiceConfig as TestsFlextServiceConfig,
        TestsFlextUserQueryService as TestsFlextUserQueryService,
        TestsFlextUserServiceEntity as TestsFlextUserServiceEntity,
    )
    from tests.integration.service_lifecycle_cases import (
        TestsFlextFlextServiceLifecycleCases as TestsFlextFlextServiceLifecycleCases,
    )
    from tests.integration.settings_integration_factories import (
        TestsFlextFlextSettingsFactories as TestsFlextFlextSettingsFactories,
        TestsFlextSettingsConfigTestCase as TestsFlextSettingsConfigTestCase,
        TestsFlextSettingsConfigTestFactories as TestsFlextSettingsConfigTestFactories,
        TestsFlextSettingsThreadSafetyTest as TestsFlextSettingsThreadSafetyTest,
    )
    from tests.integration.settings_integration_precedence import (
        TestsFlextFlextSettingsPrecedenceCase as TestsFlextFlextSettingsPrecedenceCase,
    )
    from tests.integration.system_integration_cases import (
        TestsFlextFlextSystemWorkflowCases as TestsFlextFlextSystemWorkflowCases,
    )
    from tests.integration.test_architecture import (
        TestsFlextCoreArchitecture as TestsFlextCoreArchitecture,
    )
    from tests.integration.test_documented_patterns import (
        TestsFlextCoreDocumentedPatterns as TestsFlextCoreDocumentedPatterns,
    )
    from tests.integration.test_examples_execution import (
        TestsFlextExamplesExecution as TestsFlextExamplesExecution,
    )
    from tests.integration.test_integration import (
        TestsFlextCoreIntegration as TestsFlextCoreIntegration,
    )
    from tests.integration.test_migration_validation import (
        TestsFlextCoreMigrationValidation as TestsFlextCoreMigrationValidation,
    )
    from tests.integration.test_service import (
        TestsFlextCoreService as TestsFlextCoreService,
    )
    from tests.integration.test_settings_integration import (
        TestsFlextSettingsIntegration as TestsFlextSettingsIntegration,
    )
    from tests.integration.test_system import (
        TestsFlextCoreSystem as TestsFlextCoreSystem,
    )
    from tests.models import TestsFlextModels as TestsFlextModels, m as m
    from tests.protocols import TestsFlextProtocols as TestsFlextProtocols, p as p
    from tests.typings import TestsFlextTypes as TestsFlextTypes, t as t
    from tests.unit._models.test_base import (
        Sample as Sample,
        SampleValue as SampleValue,
        TestsFlextCoreBase as TestsFlextCoreBase,
    )
    from tests.unit._models.test_cqrs import TestsFlextCoreCqrs as TestsFlextCoreCqrs
    from tests.unit._models.test_enforcement_sources import (
        TestsFlextCoreEnforcementSources as TestsFlextCoreEnforcementSources,
    )
    from tests.unit._models.test_entity import (
        TestsFlextCoreEntity as TestsFlextCoreEntity,
    )
    from tests.unit._models.test_exception_params_core import (
        TestsFlextModelsExceptionParamsCore as TestsFlextModelsExceptionParamsCore,
    )
    from tests.unit._models.test_exception_params_operations import (
        TestsFlextCoreExceptionParamsOperations as TestsFlextCoreExceptionParamsOperations,
    )
    from tests.unit._models.test_exception_params_resources import (
        TestsFlextModelsExceptionParamsResources as TestsFlextModelsExceptionParamsResources,
    )
    from tests.unit._utilities.test_guards import (
        TestsFlextCoreGuards as TestsFlextCoreGuards,
    )
    from tests.unit._utilities.test_mapper import (
        TestsFlextCoreMapper as TestsFlextCoreMapper,
    )
    from tests.unit.test_beartype_engine import (
        TestsFlextCoreBeartypeEngine as TestsFlextCoreBeartypeEngine,
    )
    from tests.unit.test_beartype_engine_annotations import (
        TestsFlextBeartypeEngineAnnotations as TestsFlextBeartypeEngineAnnotations,
    )
    from tests.unit.test_beartype_engine_claw_packages import (
        TestsFlextCoreBeartypeEngineClawPackages as TestsFlextCoreBeartypeEngineClawPackages,
    )
    from tests.unit.test_beartype_engine_config import (
        TestsFlextCoreBeartypeEngineConfig as TestsFlextCoreBeartypeEngineConfig,
    )
    from tests.unit.test_beartype_engine_import_hooks import (
        TestsFlextCoreBeartypeEngineImportHooks as TestsFlextCoreBeartypeEngineImportHooks,
    )
    from tests.unit.test_beartype_engine_namespace_hooks import (
        TestsFlextBeartypeEngineNamespaceHooks as TestsFlextBeartypeEngineNamespaceHooks,
    )
    from tests.unit.test_beartype_engine_runtime import (
        TestsFlextCoreBeartypeEngineRuntime as TestsFlextCoreBeartypeEngineRuntime,
    )
    from tests.unit.test_config_runtime import (
        TestsFlextCoreConfigSettingsCanonical as TestsFlextCoreConfigSettingsCanonical,
    )
    from tests.unit.test_constants_new import (
        TestsFlextConstantsNew as TestsFlextConstantsNew,
    )
    from tests.unit.test_constants_project_metadata import (
        TestsFlextCoreConstantsProjectMetadata as TestsFlextCoreConstantsProjectMetadata,
    )
    from tests.unit.test_container import (
        TestsFlextCoreContainer as TestsFlextCoreContainer,
    )
    from tests.unit.test_container_config import (
        TestsFlextCoreContainerConfig as TestsFlextCoreContainerConfig,
    )
    from tests.unit.test_container_lifecycle import (
        TestsFlextContainerLifecycle as TestsFlextContainerLifecycle,
    )
    from tests.unit.test_container_properties import (
        TestsFlextCoreContainerProperties as TestsFlextCoreContainerProperties,
    )
    from tests.unit.test_container_registration import (
        TestsFlextCoreContainerRegistration as TestsFlextCoreContainerRegistration,
    )
    from tests.unit.test_container_resolution import (
        TestsFlextContainerResolution as TestsFlextContainerResolution,
    )
    from tests.unit.test_context import TestsFlextCoreContext as TestsFlextCoreContext
    from tests.unit.test_coverage_loggings import (
        TestsFlextCoverageLoggings as TestsFlextCoverageLoggings,
    )
    from tests.unit.test_decorators import (
        TestsFlextCoreDecorators as TestsFlextCoreDecorators,
    )
    from tests.unit.test_decorators_combined import (
        TestsFlextCoreDecoratorsCombined as TestsFlextCoreDecoratorsCombined,
    )
    from tests.unit.test_decorators_discovery_full_coverage import (
        TestsFlextDecoratorsDiscovery as TestsFlextDecoratorsDiscovery,
    )
    from tests.unit.test_decorators_injection_logging import (
        TestsFlextCoreDecoratorsInjectionLogging as TestsFlextCoreDecoratorsInjectionLogging,
    )
    from tests.unit.test_decorators_railway_retry import (
        TestsFlextCoreDecoratorsRailwayRetry as TestsFlextCoreDecoratorsRailwayRetry,
    )
    from tests.unit.test_deprecation_warnings import (
        TestsFlextCoreDeprecationWarnings as TestsFlextCoreDeprecationWarnings,
    )
    from tests.unit.test_dispatcher import (
        TestsFlextCoreDispatcher as TestsFlextCoreDispatcher,
    )
    from tests.unit.test_enforcement import (
        TestsFlextCoreEnforcement as TestsFlextCoreEnforcement,
    )
    from tests.unit.test_enforcement_accessors import (
        TestsFlextCoreEnforcementAccessors as TestsFlextCoreEnforcementAccessors,
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
    from tests.unit.test_enforcement_layers import (
        TestsFlextCoreEnforcementLayers as TestsFlextCoreEnforcementLayers,
    )
    from tests.unit.test_enforcement_models import (
        TestsFlextEnforcementModels as TestsFlextEnforcementModels,
    )
    from tests.unit.test_enforcement_namespace import (
        TestsFlextCoreEnforcementNamespace as TestsFlextCoreEnforcementNamespace,
    )
    from tests.unit.test_enforcement_namespace_part_01 import (
        TestsFlextCoreEnforcementNamespacePart01 as TestsFlextCoreEnforcementNamespacePart01,
    )
    from tests.unit.test_enforcement_namespace_part_02 import (
        TestsFlextCoreEnforcementNamespacePart02 as TestsFlextCoreEnforcementNamespacePart02,
    )
    from tests.unit.test_enforcement_reports import (
        TestsFlextCoreEnforcementReports as TestsFlextCoreEnforcementReports,
    )
    from tests.unit.test_enforcement_warning_visibility import (
        TestsFlextCoreEnforcementWarningVisibility as TestsFlextCoreEnforcementWarningVisibility,
    )
    from tests.unit.test_enum_utilities_coverage_100 import (
        TestsFlextCoreEnumUtilities as TestsFlextCoreEnumUtilities,
    )
    from tests.unit.test_exceptions import (
        TestsFlextCoreExceptions as TestsFlextCoreExceptions,
    )
    from tests.unit.test_exceptions_base import (
        TestsFlextCoreExceptionsBase as TestsFlextCoreExceptionsBase,
    )
    from tests.unit.test_exceptions_public_metrics import (
        TestsFlextCoreExceptionsPublicMetrics as TestsFlextCoreExceptionsPublicMetrics,
    )
    from tests.unit.test_exceptions_structured_contracts import (
        TestsFlextCoreExceptionsStructuredContracts as TestsFlextCoreExceptionsStructuredContracts,
    )
    from tests.unit.test_exceptions_typed_metrics import (
        TestsFlextCoreExceptionsTypedMetrics as TestsFlextCoreExceptionsTypedMetrics,
    )
    from tests.unit.test_handler_decorator_discovery import (
        TestsFlextCoreHandlerDecoratorDiscovery as TestsFlextCoreHandlerDecoratorDiscovery,
    )
    from tests.unit.test_handler_decorator_edges import (
        TestsFlextHandlerDecoratorEdges as TestsFlextHandlerDecoratorEdges,
    )
    from tests.unit.test_handler_decorator_metadata import (
        TestsFlextHandlerDecoratorMetadata as TestsFlextHandlerDecoratorMetadata,
    )
    from tests.unit.test_handler_discovery_class import (
        TestsFlextCoreHandlerDiscoveryClass as TestsFlextCoreHandlerDiscoveryClass,
    )
    from tests.unit.test_handler_discovery_module import (
        TestsFlextHandlerDiscoveryModule as TestsFlextHandlerDiscoveryModule,
    )
    from tests.unit.test_handlers_dispatch import (
        TestsFlextHandlersDispatch as TestsFlextHandlersDispatch,
    )
    from tests.unit.test_handlers_factory import (
        TestsFlextCoreHandlersFactory as TestsFlextCoreHandlersFactory,
    )
    from tests.unit.test_handlers_lifecycle import (
        TestsFlextHandlersLifecycle as TestsFlextHandlersLifecycle,
    )
    from tests.unit.test_handlers_properties import (
        TestsFlextCoreHandlersProperties as TestsFlextCoreHandlersProperties,
    )
    from tests.unit.test_handlers_validation_context import (
        TestsFlextCoreHandlersValidationContext as TestsFlextCoreHandlersValidationContext,
    )
    from tests.unit.test_lazy_exports import (
        TestsFlextCoreLazyExports as TestsFlextCoreLazyExports,
    )
    from tests.unit.test_lazy_exports_merge import (
        TestsFlextCoreLazyExportsMerge as TestsFlextCoreLazyExportsMerge,
    )
    from tests.unit.test_loggings_full_coverage import (
        TestsFlextLoggings as TestsFlextLoggings,
    )
    from tests.unit.test_mixins import TestsFlextMixins as TestsFlextMixins
    from tests.unit.test_models import TestsFlextCoreModels as TestsFlextCoreModels
    from tests.unit.test_models_base_full_coverage import (
        TestsFlextCoreModelsBaseFullCoverage as TestsFlextCoreModelsBaseFullCoverage,
    )
    from tests.unit.test_models_container import (
        TestsFlextCoreModelsContainer as TestsFlextCoreModelsContainer,
    )
    from tests.unit.test_models_cqrs_full_coverage import (
        TestsFlextCoreModelsCqrs as TestsFlextCoreModelsCqrs,
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
    from tests.unit.test_registry import (
        TestsFlextCoreRegistry as TestsFlextCoreRegistry,
    )
    from tests.unit.test_result import TestsFlextCoreResult as TestsFlextCoreResult
    from tests.unit.test_result_callables_fold import (
        TestsFlextCoreResultCallablesFold as TestsFlextCoreResultCallablesFold,
    )
    from tests.unit.test_result_chain_helpers import (
        TestsFlextCoreResultChainHelpers as TestsFlextCoreResultChainHelpers,
    )
    from tests.unit.test_result_exception_failures import (
        TestsFlextCoreResultExceptionFailures as TestsFlextCoreResultExceptionFailures,
    )
    from tests.unit.test_result_exception_mapping import (
        TestsFlextCoreResultExceptionMapping as TestsFlextCoreResultExceptionMapping,
    )
    from tests.unit.test_result_exception_safe_callable import (
        TestsFlextCoreResultExceptionSafeCallable as TestsFlextCoreResultExceptionSafeCallable,
    )
    from tests.unit.test_result_exception_traverse_validation import (
        TestsFlextCoreResultExceptionTraverseValidation as TestsFlextCoreResultExceptionTraverseValidation,
    )
    from tests.unit.test_result_laws import (
        TestsFlextCoreResultLaws as TestsFlextCoreResultLaws,
    )
    from tests.unit.test_result_operations import (
        TestsFlextResultOperations as TestsFlextResultOperations,
    )
    from tests.unit.test_result_recent_behaviors import (
        TestsFlextCoreResultRecentBehaviors as TestsFlextCoreResultRecentBehaviors,
    )
    from tests.unit.test_result_transforms import (
        TestsFlextResultTransforms as TestsFlextResultTransforms,
    )
    from tests.unit.test_result_traverse_resource import (
        TestsFlextResultTraverseResource as TestsFlextResultTraverseResource,
    )
    from tests.unit.test_runtime import TestsFlextCoreRuntime as TestsFlextCoreRuntime
    from tests.unit.test_service import TestsFlextService as TestsFlextService
    from tests.unit.test_service_bootstrap import (
        TestsFlextCoreServiceBootstrap as TestsFlextCoreServiceBootstrap,
    )
    from tests.unit.test_settings import (
        TestsFlextCoreSettings as TestsFlextCoreSettings,
    )
    from tests.unit.test_settings_validation_alias import (
        TestsFlextCoreSettingsValidationAlias as TestsFlextCoreSettingsValidationAlias,
    )
    from tests.unit.test_typings_aliases import (
        TestsFlextCoreTypingsAliases as TestsFlextCoreTypingsAliases,
    )
    from tests.unit.test_typings_containers import (
        TestsFlextCoreTypingsContainers as TestsFlextCoreTypingsContainers,
    )
    from tests.unit.test_typings_new import (
        TestsFlextCoreTypingsNew as TestsFlextCoreTypingsNew,
    )
    from tests.unit.test_typings_validation_numbers import (
        TestsFlextCoreTypingsValidationNumbers as TestsFlextCoreTypingsValidationNumbers,
    )
    from tests.unit.test_typings_validation_scalars import (
        TestsFlextCoreTypingsValidationScalars as TestsFlextCoreTypingsValidationScalars,
    )
    from tests.unit.test_utilities import (
        TestsFlextCoreUtilities as TestsFlextCoreUtilities,
    )
    from tests.unit.test_utilities_collection_coverage_100 import (
        TestsFlextCoreUtilitiesCollection as TestsFlextCoreUtilitiesCollection,
    )
    from tests.unit.test_utilities_config import (
        TestsFlextCoreUtilitiesConfig as TestsFlextCoreUtilitiesConfig,
    )
    from tests.unit.test_utilities_coverage import (
        TestsFlextCoreUtilitiesCoverage as TestsFlextCoreUtilitiesCoverage,
    )
    from tests.unit.test_utilities_domain import (
        TestsFlextCoreUtilitiesDomain as TestsFlextCoreUtilitiesDomain,
    )
    from tests.unit.test_utilities_generators_full_coverage import (
        TestsFlextCoreUtilitiesGenerators as TestsFlextCoreUtilitiesGenerators,
    )
    from tests.unit.test_utilities_project_metadata import (
        TestsFlextCoreUtilitiesProjectMetadata as TestsFlextCoreUtilitiesProjectMetadata,
    )
    from tests.unit.test_utilities_project_metadata_read import (
        TestsFlextUtilitiesProjectMetadataRead as TestsFlextUtilitiesProjectMetadataRead,
    )
    from tests.unit.test_utilities_pydantic_coverage_100 import (
        TestsFlextUtilitiesPydantic as TestsFlextUtilitiesPydantic,
    )
    from tests.unit.test_utilities_reliability import (
        TestsFlextCoreUtilitiesReliability as TestsFlextCoreUtilitiesReliability,
    )
    from tests.unit.test_utilities_runtime_violation_registry_coverage_100 import (
        TestsFlextCoreUtilitiesRuntimeViolationRegistry as TestsFlextCoreUtilitiesRuntimeViolationRegistry,
    )
    from tests.unit.test_utilities_settings_coverage_100 import (
        TestsFlextCoreUtilitiesSettings as TestsFlextCoreUtilitiesSettings,
    )
    from tests.unit.test_utilities_text_full_coverage import (
        TestsFlextUtilitiesText as TestsFlextUtilitiesText,
    )
    from tests.unit.test_utilities_type_guards_coverage_100 import (
        TestsFlextCoreUtilitiesTypeGuards as TestsFlextCoreUtilitiesTypeGuards,
    )
    from tests.unit.test_version import TestsFlextCoreVersion as TestsFlextCoreVersion
    from tests.utilities import TestsFlextUtilities as TestsFlextUtilities, u as u
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
    build_lazy_import_map({
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
        "._models._mixins.fixture_payloads": ("TestsFlextModelsFixturePayloadsMixin",),
        "._models._mixins.fixture_suite": ("TestsFlextModelsFixtureSuiteMixin",),
        "._models._mixins.fixtures": ("TestsFlextModelsFixtureDictsMixin",),
        "._models._mixins.guards_mapper": ("TestsFlextModelsGuardsMapperMixin",),
        "._models._mixins.service_case_core": ("TestsFlextModelsServiceCaseCoreMixin",),
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
        "._models._mixins.test_data_values": ("TestsFlextModelsTestDataValuesMixin",),
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
        "._utilities.parser_scenarios": ("TestsFlextUtilitiesParserScenariosMixin",),
        "._utilities.railway": ("TestsFlextUtilitiesRailwayMixin",),
        "._utilities.railway_cases": ("TestsFlextUtilitiesRailwayCasesMixin",),
        "._utilities.railway_pipelines": ("TestsFlextUtilitiesRailwayPipelinesMixin",),
        "._utilities.railway_services": ("TestsFlextUtilitiesRailwayServicesMixin",),
        "._utilities.reliability_scenarios": (
            "TestsFlextUtilitiesReliabilityScenariosMixin",
        ),
        "._utilities.service_factories": ("TestsFlextUtilitiesServiceFactoriesMixin",),
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
        "._utilities.validation_uri": ("TestsFlextUtilitiesValidationUriScenarios",),
        ".base": ("TestsFlextServiceBase", "s"),
        ".benchmark": ("benchmark",),
        ".benchmark.test_container_memory": ("TestsFlextContainerMemory",),
        ".benchmark.test_container_performance": ("TestsFlextContainerPerformance",),
        ".benchmark.test_lazy_performance": ("TestsFlextLazyPerformance",),
        ".conftest": ("conftest",),
        ".constants": ("TestsFlextConstants", "c"),
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
        ".integration.test_documented_patterns": ("TestsFlextCoreDocumentedPatterns",),
        ".integration.test_examples_execution": ("TestsFlextExamplesExecution",),
        ".integration.test_integration": ("TestsFlextCoreIntegration",),
        ".integration.test_migration_validation": (
            "TestsFlextCoreMigrationValidation",
        ),
        ".integration.test_service": ("TestsFlextCoreService",),
        ".integration.test_settings_integration": ("TestsFlextSettingsIntegration",),
        ".integration.test_system": ("TestsFlextCoreSystem",),
        ".models": ("TestsFlextModels", "m"),
        ".protocols": ("TestsFlextProtocols", "p"),
        ".typings": ("TestsFlextTypes", "t"),
        ".unit": ("unit",),
        ".unit._models.test_base": ("Sample", "SampleValue", "TestsFlextCoreBase"),
        ".unit._models.test_cqrs": ("TestsFlextCoreCqrs",),
        ".unit._models.test_enforcement_sources": ("TestsFlextCoreEnforcementSources",),
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
        ".unit.test_beartype_engine_config": ("TestsFlextCoreBeartypeEngineConfig",),
        ".unit.test_beartype_engine_import_hooks": (
            "TestsFlextCoreBeartypeEngineImportHooks",
        ),
        ".unit.test_beartype_engine_namespace_hooks": (
            "TestsFlextBeartypeEngineNamespaceHooks",
        ),
        ".unit.test_beartype_engine_runtime": ("TestsFlextCoreBeartypeEngineRuntime",),
        ".unit.test_config_runtime": ("TestsFlextCoreConfigSettingsCanonical",),
        ".unit.test_constants_new": ("TestsFlextConstantsNew",),
        ".unit.test_constants_project_metadata": (
            "TestsFlextCoreConstantsProjectMetadata",
        ),
        ".unit.test_container": ("TestsFlextCoreContainer",),
        ".unit.test_container_config": ("TestsFlextCoreContainerConfig",),
        ".unit.test_container_lifecycle": ("TestsFlextContainerLifecycle",),
        ".unit.test_container_properties": ("TestsFlextCoreContainerProperties",),
        ".unit.test_container_registration": ("TestsFlextCoreContainerRegistration",),
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
        ".unit.test_handler_discovery_class": ("TestsFlextCoreHandlerDiscoveryClass",),
        ".unit.test_handler_discovery_module": ("TestsFlextHandlerDiscoveryModule",),
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
        ".unit.test_result_recent_behaviors": ("TestsFlextCoreResultRecentBehaviors",),
        ".unit.test_result_transforms": ("TestsFlextResultTransforms",),
        ".unit.test_result_traverse_resource": ("TestsFlextResultTraverseResource",),
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
        ".unit.test_utilities_config": ("TestsFlextCoreUtilitiesConfig",),
        ".unit.test_utilities_coverage": ("TestsFlextCoreUtilitiesCoverage",),
        ".unit.test_utilities_domain": ("TestsFlextCoreUtilitiesDomain",),
        ".unit.test_utilities_generators_full_coverage": (
            "TestsFlextCoreUtilitiesGenerators",
        ),
        ".unit.test_utilities_project_metadata": (
            "TestsFlextCoreUtilitiesProjectMetadata",
        ),
        ".unit.test_utilities_project_metadata_read": (
            "TestsFlextUtilitiesProjectMetadataRead",
        ),
        ".unit.test_utilities_pydantic_coverage_100": ("TestsFlextUtilitiesPydantic",),
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
        ".utilities": ("TestsFlextUtilities", "u"),
        "flext_tests": ("d", "e", "h", "r", "td", "tf", "tk", "tm", "tv", "x"),
    }),
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


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, publish_all=False)
