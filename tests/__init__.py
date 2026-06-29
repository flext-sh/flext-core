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
    from tests.integration.migration_validation_cases import (
        FlextMigrationApplicationCase as FlextMigrationApplicationCase,
    )
    from tests.integration.service_fixtures import (
        FlextServiceFixtures as FlextServiceFixtures,
        LifecycleService as LifecycleService,
        NotificationService as NotificationService,
        ServiceConfig as ServiceConfig,
        UserQueryService as UserQueryService,
        UserServiceEntity as UserServiceEntity,
    )
    from tests.integration.service_lifecycle_cases import (
        FlextServiceLifecycleCases as FlextServiceLifecycleCases,
    )
    from tests.integration.settings_integration_factories import (
        FlextSettingsFactories as FlextSettingsFactories,
        SettingsConfigTestCase as SettingsConfigTestCase,
        SettingsConfigTestFactories as SettingsConfigTestFactories,
        SettingsThreadSafetyTest as SettingsThreadSafetyTest,
    )
    from tests.integration.settings_integration_precedence import (
        FlextSettingsPrecedenceCase as FlextSettingsPrecedenceCase,
    )
    from tests.integration.system_integration_cases import (
        FlextSystemWorkflowCases as FlextSystemWorkflowCases,
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
    from tests.unit._models.test_exception_params_core import (
        TestsFlextModelsExceptionParamsCore as TestsFlextModelsExceptionParamsCore,
    )
    from tests.unit._models.test_exception_params_operations import (
        TestsFlextModelsExceptionParamsOperations as TestsFlextModelsExceptionParamsOperations,
    )
    from tests.unit._models.test_exception_params_resources import (
        TestsFlextModelsExceptionParamsResources as TestsFlextModelsExceptionParamsResources,
    )
    from tests.unit._utilities.test_guards import (
        TestsFlextUtilitiesGuards as TestsFlextUtilitiesGuards,
    )
    from tests.unit._utilities.test_mapper import (
        TestsFlextUtilitiesMapper as TestsFlextUtilitiesMapper,
    )
    from tests.unit.test_beartype_engine_annotations import (
        TestsFlextBeartypeEngineAnnotations as TestsFlextBeartypeEngineAnnotations,
    )
    from tests.unit.test_beartype_engine_claw_packages import (
        TestsFlextBeartypeEngineClawPackages as TestsFlextBeartypeEngineClawPackages,
    )
    from tests.unit.test_beartype_engine_config import (
        TestsFlextBeartypeEngineConfig as TestsFlextBeartypeEngineConfig,
    )
    from tests.unit.test_beartype_engine_import_hooks import (
        TestsFlextBeartypeEngineImportHooks as TestsFlextBeartypeEngineImportHooks,
    )
    from tests.unit.test_beartype_engine_namespace_hooks import (
        TestsFlextBeartypeEngineNamespaceHooks as TestsFlextBeartypeEngineNamespaceHooks,
    )
    from tests.unit.test_beartype_engine_runtime import (
        TestsFlextBeartypeEngineRuntime as TestsFlextBeartypeEngineRuntime,
    )
    from tests.unit.test_constants_new import (
        TestsFlextConstantsNew as TestsFlextConstantsNew,
    )
    from tests.unit.test_constants_project_metadata import (
        TestsFlextConstantsProjectMetadata as TestsFlextConstantsProjectMetadata,
    )
    from tests.unit.test_container_config import (
        TestsFlextContainerConfig as TestsFlextContainerConfig,
    )
    from tests.unit.test_container_lifecycle import (
        TestsFlextContainerLifecycle as TestsFlextContainerLifecycle,
    )
    from tests.unit.test_container_properties import (
        TestsFlextContainerProperties as TestsFlextContainerProperties,
    )
    from tests.unit.test_container_registration import (
        TestsFlextContainerRegistration as TestsFlextContainerRegistration,
    )
    from tests.unit.test_container_resolution import (
        TestsFlextContainerResolution as TestsFlextContainerResolution,
    )
    from tests.unit.test_context import TestsFlextContext as TestsFlextContext
    from tests.unit.test_coverage_loggings import (
        TestsFlextCoverageLoggings as TestsFlextCoverageLoggings,
    )
    from tests.unit.test_decorators_combined import (
        TestsFlextDecoratorsCombined as TestsFlextDecoratorsCombined,
    )
    from tests.unit.test_decorators_discovery_full_coverage import (
        TestsFlextDecoratorsDiscovery as TestsFlextDecoratorsDiscovery,
    )
    from tests.unit.test_decorators_full_coverage import (
        TestsFlextDecorators as TestsFlextDecorators,
    )
    from tests.unit.test_decorators_injection_logging import (
        TestsFlextDecoratorsInjectionLogging as TestsFlextDecoratorsInjectionLogging,
    )
    from tests.unit.test_decorators_railway_retry import (
        TestsFlextDecoratorsRailwayRetry as TestsFlextDecoratorsRailwayRetry,
    )
    from tests.unit.test_deprecation_warnings import (
        TestsFlextDeprecationWarnings as TestsFlextDeprecationWarnings,
    )
    from tests.unit.test_dispatcher import TestsFlextDispatcher as TestsFlextDispatcher
    from tests.unit.test_enforcement_accessors import (
        TestsFlextEnforcementAccessors as TestsFlextEnforcementAccessors,
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
        TestsFlextEnforcementLayers as TestsFlextEnforcementLayers,
    )
    from tests.unit.test_enforcement_models import (
        TestsFlextEnforcementModels as TestsFlextEnforcementModels,
    )
    from tests.unit.test_enforcement_namespace import (
        TestsFlextEnforcementNamespace as TestsFlextEnforcementNamespace,
    )
    from tests.unit.test_enforcement_reports import (
        TestsFlextEnforcementReports as TestsFlextEnforcementReports,
    )
    from tests.unit.test_enum_utilities_coverage_100 import (
        TestsFlextEnumUtilities as TestsFlextEnumUtilities,
    )
    from tests.unit.test_exceptions_base import (
        TestsFlextExceptionsBase as TestsFlextExceptionsBase,
    )
    from tests.unit.test_exceptions_public_metrics import (
        TestsFlextCoverageExceptionMetrics as TestsFlextCoverageExceptionMetrics,
    )
    from tests.unit.test_exceptions_structured_contracts import (
        TestsFlextCoverageExceptionContracts as TestsFlextCoverageExceptionContracts,
    )
    from tests.unit.test_exceptions_typed_metrics import (
        TestsFlextExceptionsTypedMetrics as TestsFlextExceptionsTypedMetrics,
    )
    from tests.unit.test_handler_decorator_edges import (
        TestsFlextHandlerDecoratorEdges as TestsFlextHandlerDecoratorEdges,
    )
    from tests.unit.test_handler_decorator_metadata import (
        TestsFlextHandlerDecoratorMetadata as TestsFlextHandlerDecoratorMetadata,
    )
    from tests.unit.test_handler_discovery_class import (
        TestsFlextHandlerDiscoveryClass as TestsFlextHandlerDiscoveryClass,
    )
    from tests.unit.test_handler_discovery_module import (
        TestsFlextHandlerDiscoveryModule as TestsFlextHandlerDiscoveryModule,
    )
    from tests.unit.test_handlers_dispatch import (
        TestsFlextHandlersDispatch as TestsFlextHandlersDispatch,
    )
    from tests.unit.test_handlers_factory import (
        TestsFlextHandlersFactory as TestsFlextHandlersFactory,
    )
    from tests.unit.test_handlers_lifecycle import (
        TestsFlextHandlersLifecycle as TestsFlextHandlersLifecycle,
    )
    from tests.unit.test_handlers_properties import (
        TestsFlextHandlersProperties as TestsFlextHandlersProperties,
    )
    from tests.unit.test_handlers_validation_context import (
        TestsFlextHandlersValidationContext as TestsFlextHandlersValidationContext,
    )
    from tests.unit.test_lazy_exports import TestsFlextLazy as TestsFlextLazy
    from tests.unit.test_lazy_exports_merge import (
        TestsFlextLazyMerge as TestsFlextLazyMerge,
    )
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
    from tests.unit.test_result_callables_fold import (
        TestsFlextResultCallablesFold as TestsFlextResultCallablesFold,
    )
    from tests.unit.test_result_chain_helpers import (
        TestsFlextResultChainHelpers as TestsFlextResultChainHelpers,
    )
    from tests.unit.test_result_exception_failures import (
        TestsFlextResultExceptionFailures as TestsFlextResultExceptionFailures,
    )
    from tests.unit.test_result_exception_mapping import (
        TestsFlextResultExceptionMapping as TestsFlextResultExceptionMapping,
    )
    from tests.unit.test_result_exception_safe_callable import (
        TestsFlextResultExceptionSafeCallable as TestsFlextResultExceptionSafeCallable,
    )
    from tests.unit.test_result_exception_traverse_validation import (
        TestsFlextResultExceptionTraverseValidation as TestsFlextResultExceptionTraverseValidation,
    )
    from tests.unit.test_result_laws import TestsFlextResultLaws as TestsFlextResultLaws
    from tests.unit.test_result_operations import (
        TestsFlextResultOperations as TestsFlextResultOperations,
    )
    from tests.unit.test_result_recent_behaviors import (
        TestsFlextResultRecentBehaviors as TestsFlextResultRecentBehaviors,
    )
    from tests.unit.test_result_transforms import (
        TestsFlextResultTransforms as TestsFlextResultTransforms,
    )
    from tests.unit.test_result_traverse_resource import (
        TestsFlextResultTraverseResource as TestsFlextResultTraverseResource,
    )
    from tests.unit.test_runtime import TestsFlextRuntime as TestsFlextRuntime
    from tests.unit.test_service import TestsFlextService as TestsFlextService
    from tests.unit.test_service_bootstrap import (
        TestsFlextServiceBootstrap as TestsFlextServiceBootstrap,
    )
    from tests.unit.test_settings import TestsFlextSettings as TestsFlextSettings
    from tests.unit.test_settings_validation_alias import (
        TestUpdateGlobalWithValidationAlias as TestUpdateGlobalWithValidationAlias,
    )
    from tests.unit.test_typings_aliases import (
        TestsFlextTypesAliases as TestsFlextTypesAliases,
    )
    from tests.unit.test_typings_containers import (
        TestsFlextTypesContainers as TestsFlextTypesContainers,
    )
    from tests.unit.test_typings_validation_numbers import (
        TestsFlextTypesValidationNumbers as TestsFlextTypesValidationNumbers,
    )
    from tests.unit.test_typings_validation_scalars import (
        TestsFlextTypesValidationScalars as TestsFlextTypesValidationScalars,
    )
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
    from tests.unit.test_utilities_project_metadata_config import (
        TestsFlextUtilitiesProjectMetadataConfig as TestsFlextUtilitiesProjectMetadataConfig,
    )
    from tests.unit.test_utilities_project_metadata_read import (
        TestsFlextUtilitiesProjectMetadataRead as TestsFlextUtilitiesProjectMetadataRead,
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
            ".integration": ("integration",),
            ".integration.migration_validation_cases": (
                "FlextMigrationApplicationCase",
            ),
            ".integration.service_fixtures": (
                "FlextServiceFixtures",
                "LifecycleService",
                "NotificationService",
                "ServiceConfig",
                "UserQueryService",
                "UserServiceEntity",
            ),
            ".integration.service_lifecycle_cases": ("FlextServiceLifecycleCases",),
            ".integration.settings_integration_factories": (
                "FlextSettingsFactories",
                "SettingsConfigTestCase",
                "SettingsConfigTestFactories",
                "SettingsThreadSafetyTest",
            ),
            ".integration.settings_integration_precedence": (
                "FlextSettingsPrecedenceCase",
            ),
            ".integration.system_integration_cases": ("FlextSystemWorkflowCases",),
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
            ".unit": ("unit",),
            ".unit._models.test_base": ("TestsFlextModelsBase",),
            ".unit._models.test_cqrs": ("TestsFlextModelsCQRS",),
            ".unit._models.test_enforcement_sources": (
                "TestsFlextModelsEnforcementSources",
            ),
            ".unit._models.test_entity": ("TestsFlextModelsEntity",),
            ".unit._models.test_exception_params_core": (
                "TestsFlextModelsExceptionParamsCore",
            ),
            ".unit._models.test_exception_params_operations": (
                "TestsFlextModelsExceptionParamsOperations",
            ),
            ".unit._models.test_exception_params_resources": (
                "TestsFlextModelsExceptionParamsResources",
            ),
            ".unit._utilities.test_guards": ("TestsFlextUtilitiesGuards",),
            ".unit._utilities.test_mapper": ("TestsFlextUtilitiesMapper",),
            ".unit.test_beartype_engine_annotations": (
                "TestsFlextBeartypeEngineAnnotations",
            ),
            ".unit.test_beartype_engine_claw_packages": (
                "TestsFlextBeartypeEngineClawPackages",
            ),
            ".unit.test_beartype_engine_config": ("TestsFlextBeartypeEngineConfig",),
            ".unit.test_beartype_engine_import_hooks": (
                "TestsFlextBeartypeEngineImportHooks",
            ),
            ".unit.test_beartype_engine_namespace_hooks": (
                "TestsFlextBeartypeEngineNamespaceHooks",
            ),
            ".unit.test_beartype_engine_runtime": ("TestsFlextBeartypeEngineRuntime",),
            ".unit.test_constants_new": ("TestsFlextConstantsNew",),
            ".unit.test_constants_project_metadata": (
                "TestsFlextConstantsProjectMetadata",
            ),
            ".unit.test_container_config": ("TestsFlextContainerConfig",),
            ".unit.test_container_lifecycle": ("TestsFlextContainerLifecycle",),
            ".unit.test_container_properties": ("TestsFlextContainerProperties",),
            ".unit.test_container_registration": ("TestsFlextContainerRegistration",),
            ".unit.test_container_resolution": ("TestsFlextContainerResolution",),
            ".unit.test_context": ("TestsFlextContext",),
            ".unit.test_coverage_loggings": ("TestsFlextCoverageLoggings",),
            ".unit.test_decorators_combined": ("TestsFlextDecoratorsCombined",),
            ".unit.test_decorators_discovery_full_coverage": (
                "TestsFlextDecoratorsDiscovery",
            ),
            ".unit.test_decorators_full_coverage": ("TestsFlextDecorators",),
            ".unit.test_decorators_injection_logging": (
                "TestsFlextDecoratorsInjectionLogging",
            ),
            ".unit.test_decorators_railway_retry": (
                "TestsFlextDecoratorsRailwayRetry",
            ),
            ".unit.test_deprecation_warnings": ("TestsFlextDeprecationWarnings",),
            ".unit.test_dispatcher": ("TestsFlextDispatcher",),
            ".unit.test_enforcement_accessors": ("TestsFlextEnforcementAccessors",),
            ".unit.test_enforcement_apt_hooks": ("TestsFlextEnforcementAptHooks",),
            ".unit.test_enforcement_catalog": ("TestsFlextEnforcementCatalog",),
            ".unit.test_enforcement_integration": ("TestsFlextEnforcementIntegration",),
            ".unit.test_enforcement_layers": ("TestsFlextEnforcementLayers",),
            ".unit.test_enforcement_models": ("TestsFlextEnforcementModels",),
            ".unit.test_enforcement_namespace": ("TestsFlextEnforcementNamespace",),
            ".unit.test_enforcement_reports": ("TestsFlextEnforcementReports",),
            ".unit.test_enum_utilities_coverage_100": ("TestsFlextEnumUtilities",),
            ".unit.test_exceptions_base": ("TestsFlextExceptionsBase",),
            ".unit.test_exceptions_public_metrics": (
                "TestsFlextCoverageExceptionMetrics",
            ),
            ".unit.test_exceptions_structured_contracts": (
                "TestsFlextCoverageExceptionContracts",
            ),
            ".unit.test_exceptions_typed_metrics": (
                "TestsFlextExceptionsTypedMetrics",
            ),
            ".unit.test_handler_decorator_edges": ("TestsFlextHandlerDecoratorEdges",),
            ".unit.test_handler_decorator_metadata": (
                "TestsFlextHandlerDecoratorMetadata",
            ),
            ".unit.test_handler_discovery_class": ("TestsFlextHandlerDiscoveryClass",),
            ".unit.test_handler_discovery_module": (
                "TestsFlextHandlerDiscoveryModule",
            ),
            ".unit.test_handlers_dispatch": ("TestsFlextHandlersDispatch",),
            ".unit.test_handlers_factory": ("TestsFlextHandlersFactory",),
            ".unit.test_handlers_lifecycle": ("TestsFlextHandlersLifecycle",),
            ".unit.test_handlers_properties": ("TestsFlextHandlersProperties",),
            ".unit.test_handlers_validation_context": (
                "TestsFlextHandlersValidationContext",
            ),
            ".unit.test_lazy_exports": ("TestsFlextLazy",),
            ".unit.test_lazy_exports_merge": ("TestsFlextLazyMerge",),
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
            ".unit.test_result_callables_fold": ("TestsFlextResultCallablesFold",),
            ".unit.test_result_chain_helpers": ("TestsFlextResultChainHelpers",),
            ".unit.test_result_exception_failures": (
                "TestsFlextResultExceptionFailures",
            ),
            ".unit.test_result_exception_mapping": (
                "TestsFlextResultExceptionMapping",
            ),
            ".unit.test_result_exception_safe_callable": (
                "TestsFlextResultExceptionSafeCallable",
            ),
            ".unit.test_result_exception_traverse_validation": (
                "TestsFlextResultExceptionTraverseValidation",
            ),
            ".unit.test_result_laws": ("TestsFlextResultLaws",),
            ".unit.test_result_operations": ("TestsFlextResultOperations",),
            ".unit.test_result_recent_behaviors": ("TestsFlextResultRecentBehaviors",),
            ".unit.test_result_transforms": ("TestsFlextResultTransforms",),
            ".unit.test_result_traverse_resource": (
                "TestsFlextResultTraverseResource",
            ),
            ".unit.test_runtime": ("TestsFlextRuntime",),
            ".unit.test_service": ("TestsFlextService",),
            ".unit.test_service_bootstrap": ("TestsFlextServiceBootstrap",),
            ".unit.test_settings": ("TestsFlextSettings",),
            ".unit.test_settings_validation_alias": (
                "TestUpdateGlobalWithValidationAlias",
            ),
            ".unit.test_typings_aliases": ("TestsFlextTypesAliases",),
            ".unit.test_typings_containers": ("TestsFlextTypesContainers",),
            ".unit.test_typings_validation_numbers": (
                "TestsFlextTypesValidationNumbers",
            ),
            ".unit.test_typings_validation_scalars": (
                "TestsFlextTypesValidationScalars",
            ),
            ".unit.test_utilities": ("TestsFlextUtilitiesSmoke",),
            ".unit.test_utilities_collection_coverage_100": (
                "TestsFlextUtilitiesCollection",
            ),
            ".unit.test_utilities_coverage": ("TestsFlextUtilitiesCoverage",),
            ".unit.test_utilities_domain": ("TestsFlextUtilitiesDomain",),
            ".unit.test_utilities_generators_full_coverage": (
                "TestsFlextUtilitiesGenerators",
            ),
            ".unit.test_utilities_project_metadata_config": (
                "TestsFlextUtilitiesProjectMetadataConfig",
            ),
            ".unit.test_utilities_project_metadata_read": (
                "TestsFlextUtilitiesProjectMetadataRead",
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
