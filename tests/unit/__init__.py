# AUTO-GENERATED FILE — Regenerate with: make gen
"""Unit package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import (
    build_lazy_import_map,
    install_lazy_exports,
    merge_lazy_imports,
)

if TYPE_CHECKING:
    from flext_tests import (
        c as c,
        d as d,
        e as e,
        h as h,
        m as m,
        p as p,
        r as r,
        s as s,
        t as t,
        td as td,
        tf as tf,
        tk as tk,
        tm as tm,
        tv as tv,
        u as u,
        x as x,
    )

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
_LAZY_IMPORTS = merge_lazy_imports(
    (
        "._models",
        "._utilities",
    ),
    build_lazy_import_map(
        {
            "._models": ("_models",),
            "._models.test_base": ("TestsFlextModelsBase",),
            "._models.test_cqrs": ("TestsFlextModelsCQRS",),
            "._models.test_enforcement_sources": (
                "TestsFlextModelsEnforcementSources",
            ),
            "._models.test_entity": ("TestsFlextModelsEntity",),
            "._models.test_exception_params_core": (
                "TestsFlextModelsExceptionParamsCore",
            ),
            "._models.test_exception_params_operations": (
                "TestsFlextModelsExceptionParamsOperations",
            ),
            "._models.test_exception_params_resources": (
                "TestsFlextModelsExceptionParamsResources",
            ),
            "._utilities": ("_utilities",),
            "._utilities.test_guards": ("TestsFlextUtilitiesGuards",),
            "._utilities.test_mapper": ("TestsFlextUtilitiesMapper",),
            ".test_beartype_engine": ("test_beartype_engine",),
            ".test_beartype_engine_annotations": (
                "TestsFlextBeartypeEngineAnnotations",
            ),
            ".test_beartype_engine_claw_packages": (
                "TestsFlextBeartypeEngineClawPackages",
            ),
            ".test_beartype_engine_config": ("TestsFlextBeartypeEngineConfig",),
            ".test_beartype_engine_import_hooks": (
                "TestsFlextBeartypeEngineImportHooks",
            ),
            ".test_beartype_engine_namespace_hooks": (
                "TestsFlextBeartypeEngineNamespaceHooks",
            ),
            ".test_beartype_engine_runtime": ("TestsFlextBeartypeEngineRuntime",),
            ".test_constants_new": ("TestsFlextConstantsNew",),
            ".test_constants_project_metadata": ("TestsFlextConstantsProjectMetadata",),
            ".test_container": ("test_container",),
            ".test_container_config": ("TestsFlextContainerConfig",),
            ".test_container_lifecycle": ("TestsFlextContainerLifecycle",),
            ".test_container_properties": ("TestsFlextContainerProperties",),
            ".test_container_registration": ("TestsFlextContainerRegistration",),
            ".test_container_resolution": ("TestsFlextContainerResolution",),
            ".test_context": ("TestsFlextContext",),
            ".test_coverage_loggings": ("TestsFlextCoverageLoggings",),
            ".test_decorators": ("test_decorators",),
            ".test_decorators_combined": ("TestsFlextDecoratorsCombined",),
            ".test_decorators_discovery_full_coverage": (
                "TestsFlextDecoratorsDiscovery",
            ),
            ".test_decorators_full_coverage": ("TestsFlextDecorators",),
            ".test_decorators_injection_logging": (
                "TestsFlextDecoratorsInjectionLogging",
            ),
            ".test_decorators_railway_retry": ("TestsFlextDecoratorsRailwayRetry",),
            ".test_deprecation_warnings": ("TestsFlextDeprecationWarnings",),
            ".test_dispatcher": ("TestsFlextDispatcher",),
            ".test_enforcement": ("test_enforcement",),
            ".test_enforcement_accessors": ("TestsFlextEnforcementAccessors",),
            ".test_enforcement_apt_hooks": ("TestsFlextEnforcementAptHooks",),
            ".test_enforcement_catalog": ("TestsFlextEnforcementCatalog",),
            ".test_enforcement_integration": ("TestsFlextEnforcementIntegration",),
            ".test_enforcement_layers": ("TestsFlextEnforcementLayers",),
            ".test_enforcement_models": ("TestsFlextEnforcementModels",),
            ".test_enforcement_namespace": ("TestsFlextEnforcementNamespace",),
            ".test_enforcement_reports": ("TestsFlextEnforcementReports",),
            ".test_enum_utilities_coverage_100": ("TestsFlextEnumUtilities",),
            ".test_exceptions": ("test_exceptions",),
            ".test_exceptions_base": ("TestsFlextExceptionsBase",),
            ".test_exceptions_public_metrics": ("TestsFlextCoverageExceptionMetrics",),
            ".test_exceptions_structured_contracts": (
                "TestsFlextCoverageExceptionContracts",
            ),
            ".test_exceptions_typed_metrics": ("TestsFlextExceptionsTypedMetrics",),
            ".test_handler_decorator_discovery": ("test_handler_decorator_discovery",),
            ".test_handler_decorator_edges": ("TestsFlextHandlerDecoratorEdges",),
            ".test_handler_decorator_metadata": ("TestsFlextHandlerDecoratorMetadata",),
            ".test_handler_discovery_class": ("TestsFlextHandlerDiscoveryClass",),
            ".test_handler_discovery_module": ("TestsFlextHandlerDiscoveryModule",),
            ".test_handlers": ("test_handlers",),
            ".test_handlers_dispatch": ("TestsFlextHandlersDispatch",),
            ".test_handlers_factory": ("TestsFlextHandlersFactory",),
            ".test_handlers_lifecycle": ("TestsFlextHandlersLifecycle",),
            ".test_handlers_properties": ("TestsFlextHandlersProperties",),
            ".test_handlers_validation_context": (
                "TestsFlextHandlersValidationContext",
            ),
            ".test_lazy_exports": ("TestsFlextLazy",),
            ".test_lazy_exports_merge": ("TestsFlextLazyMerge",),
            ".test_loggings_full_coverage": ("TestsFlextLoggings",),
            ".test_mixins": ("TestsFlextMixins",),
            ".test_models": ("TestsFlextModelsUnit",),
            ".test_models_base_full_coverage": ("TestsFlextModelsBaseFullCoverage",),
            ".test_models_container": ("TestsFlextModelsContainer",),
            ".test_models_cqrs_full_coverage": ("TestsFlextModelsCqrs",),
            ".test_models_project_metadata": ("TestsFlextModelsProjectMetadata",),
            ".test_project_metadata_facade_access": ("TestsFlextFacadeFlatSsotAccess",),
            ".test_public_api_contract": ("TestsFlextCorePublicApiContract",),
            ".test_registry": ("TestsFlextRegistry",),
            ".test_result": ("test_result",),
            ".test_result_callables_fold": ("TestsFlextResultCallablesFold",),
            ".test_result_chain_helpers": ("TestsFlextResultChainHelpers",),
            ".test_result_exception_failures": ("TestsFlextResultExceptionFailures",),
            ".test_result_exception_mapping": ("TestsFlextResultExceptionMapping",),
            ".test_result_exception_safe_callable": (
                "TestsFlextResultExceptionSafeCallable",
            ),
            ".test_result_exception_traverse_validation": (
                "TestsFlextResultExceptionTraverseValidation",
            ),
            ".test_result_laws": ("TestsFlextResultLaws",),
            ".test_result_operations": ("TestsFlextResultOperations",),
            ".test_result_recent_behaviors": ("TestsFlextResultRecentBehaviors",),
            ".test_result_transforms": ("TestsFlextResultTransforms",),
            ".test_result_traverse_resource": ("TestsFlextResultTraverseResource",),
            ".test_runtime": ("TestsFlextRuntime",),
            ".test_service": ("TestsFlextService",),
            ".test_service_bootstrap": ("TestsFlextServiceBootstrap",),
            ".test_settings": ("TestsFlextSettings",),
            ".test_settings_validation_alias": ("TestUpdateGlobalWithValidationAlias",),
            ".test_typings_aliases": ("TestsFlextTypesAliases",),
            ".test_typings_containers": ("TestsFlextTypesContainers",),
            ".test_typings_new": ("test_typings_new",),
            ".test_typings_validation_numbers": ("TestsFlextTypesValidationNumbers",),
            ".test_typings_validation_scalars": ("TestsFlextTypesValidationScalars",),
            ".test_utilities": ("TestsFlextUtilitiesSmoke",),
            ".test_utilities_collection_coverage_100": (
                "TestsFlextUtilitiesCollection",
            ),
            ".test_utilities_coverage": ("TestsFlextUtilitiesCoverage",),
            ".test_utilities_domain": ("TestsFlextUtilitiesDomain",),
            ".test_utilities_generators_full_coverage": (
                "TestsFlextUtilitiesGenerators",
            ),
            ".test_utilities_project_metadata": ("test_utilities_project_metadata",),
            ".test_utilities_project_metadata_config": (
                "TestsFlextUtilitiesProjectMetadataConfig",
            ),
            ".test_utilities_project_metadata_read": (
                "TestsFlextUtilitiesProjectMetadataRead",
            ),
            ".test_utilities_pydantic_coverage_100": ("TestsFlextUtilitiesPydantic",),
            ".test_utilities_reliability": ("TestsFlextUtilitiesReliability",),
            ".test_utilities_runtime_violation_registry_coverage_100": (
                "TestsFlextRuntimeViolationRegistry",
            ),
            ".test_utilities_settings_coverage_100": (
                "TestsFlextUtilitiesSettings",
                "TestsFlextUtilitiesSettingsEnvFile",
                "TestsFlextUtilitiesSettingsRegisterFactory",
            ),
            ".test_utilities_text_full_coverage": ("TestsFlextUtilitiesText",),
            ".test_utilities_type_guards_coverage_100": (
                "TestsFlextUtilitiesTypeGuards",
            ),
            ".test_version": ("TestsFlextVersion",),
            "flext_tests": (
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
