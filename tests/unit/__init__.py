# AUTO-GENERATED FILE — Regenerate with: make gen
"""Tests.unit package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from types import MappingProxyType

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from . import _models as _models
    from . import _utilities as _utilities
    from flext_tests import c, d, e, h, m, p, r, s, t, td, tf, tk, tm, tv, u, x

    from .test_beartype_engine import TestsFlextCoreBeartypeEngine
    from .test_beartype_engine_annotations import TestsFlextBeartypeEngineAnnotations
    from .test_beartype_engine_claw_packages import (
        TestsFlextCoreBeartypeEngineClawPackages,
    )
    from .test_beartype_engine_config import TestsFlextCoreBeartypeEngineConfig
    from .test_beartype_engine_import_hooks import (
        TestsFlextCoreBeartypeEngineImportHooks,
    )
    from .test_beartype_engine_namespace_hooks import (
        TestsFlextBeartypeEngineNamespaceHooks,
    )
    from .test_beartype_engine_runtime import TestsFlextCoreBeartypeEngineRuntime
    from .test_config_runtime import TestsFlextCoreConfigSettingsCanonical
    from .test_config_user_preferences import (
        TestPackagedConfigWithUserPreferences,
        isolate_import_state,
    )
    from .test_constants_new import TestsFlextConstantsNew
    from .test_constants_project_metadata import TestsFlextCoreConstantsProjectMetadata
    from .test_container import TestsFlextCoreContainer
    from .test_container_config import TestsFlextCoreContainerConfig
    from .test_container_lifecycle import TestsFlextContainerLifecycle
    from .test_container_properties import TestsFlextCoreContainerProperties
    from .test_container_registration import TestsFlextCoreContainerRegistration
    from .test_container_resolution import TestsFlextContainerResolution
    from .test_context import TestsFlextCoreContext
    from .test_coverage_loggings import TestsFlextCoverageLoggings
    from .test_decorators import TestsFlextCoreDecorators
    from .test_decorators_combined import TestsFlextCoreDecoratorsCombined
    from .test_decorators_discovery_full_coverage import TestsFlextDecoratorsDiscovery
    from .test_decorators_injection_logging import (
        TestsFlextCoreDecoratorsInjectionLogging,
    )
    from .test_decorators_railway_retry import TestsFlextCoreDecoratorsRailwayRetry
    from .test_deprecation_warnings import TestsFlextCoreDeprecationWarnings
    from .test_dispatcher import TestsFlextCoreDispatcher
    from .test_enforcement import TestsFlextCoreEnforcement
    from .test_enforcement_accessors import TestsFlextCoreEnforcementAccessors
    from .test_enforcement_apt_hooks import TestsFlextEnforcementAptHooks
    from .test_enforcement_catalog import TestsFlextEnforcementCatalog
    from .test_enforcement_integration import TestsFlextEnforcementIntegration
    from .test_enforcement_layers import TestsFlextCoreEnforcementLayers
    from .test_enforcement_models import TestsFlextEnforcementModels
    from .test_enforcement_namespace import TestsFlextCoreEnforcementNamespace
    from .test_enforcement_namespace_part_01 import (
        TestsFlextCoreEnforcementNamespacePart01,
    )
    from .test_enforcement_namespace_part_02 import (
        TestsFlextCoreEnforcementNamespacePart02,
    )
    from .test_enforcement_reports import TestsFlextCoreEnforcementReports
    from .test_enforcement_warning_visibility import (
        TestsFlextCoreEnforcementWarningVisibility,
    )
    from .test_enum_utilities_coverage_100 import TestsFlextCoreEnumUtilities
    from .test_exceptions import TestsFlextCoreExceptions
    from .test_exceptions_base import TestsFlextCoreExceptionsBase
    from .test_exceptions_public_metrics import TestsFlextCoreExceptionsPublicMetrics
    from .test_exceptions_structured_contracts import (
        TestsFlextCoreExceptionsStructuredContracts,
    )
    from .test_exceptions_typed_metrics import TestsFlextCoreExceptionsTypedMetrics
    from .test_handler_decorator_discovery import (
        TestsFlextCoreHandlerDecoratorDiscovery,
    )
    from .test_handler_decorator_edges import TestsFlextHandlerDecoratorEdges
    from .test_handler_decorator_metadata import TestsFlextHandlerDecoratorMetadata
    from .test_handler_discovery_class import TestsFlextCoreHandlerDiscoveryClass
    from .test_handler_discovery_module import TestsFlextHandlerDiscoveryModule
    from .test_handlers_dispatch import TestsFlextHandlersDispatch
    from .test_handlers_factory import TestsFlextCoreHandlersFactory
    from .test_handlers_lifecycle import (
        HANDLER_TYPES,
        HandlerTypeScenario,
        TestsFlextHandlersLifecycle,
        VALIDATION_TYPES,
    )
    from .test_handlers_properties import TestsFlextCoreHandlersProperties
    from .test_handlers_validation_context import (
        TestsFlextCoreHandlersValidationContext,
    )
    from .test_lazy_exports import TestsFlextCoreLazyExports
    from .test_lazy_exports_merge import TestsFlextCoreLazyExportsMerge
    from .test_loggings_full_coverage import LOG_LEVELS, TestsFlextLoggings
    from .test_mixins import TestsFlextMixins
    from .test_models import TestsFlextCoreModels
    from .test_models_base_full_coverage import TestsFlextCoreModelsBaseFullCoverage
    from .test_models_container import TestsFlextCoreModelsContainer
    from .test_models_cqrs_full_coverage import TestsFlextCoreModelsCqrs
    from .test_models_project_metadata import TestsFlextModelsProjectMetadata
    from .test_project_metadata_facade_access import TestsFlextFacadeFlatSsotAccess
    from .test_public_api_contract import TestsFlextCorePublicApiContract
    from .test_registry import TestsFlextCoreRegistry
    from .test_result import TestsFlextCoreResult
    from .test_result_callables_fold import TestsFlextCoreResultCallablesFold
    from .test_result_chain_helpers import TestsFlextCoreResultChainHelpers
    from .test_result_exception_failures import TestsFlextCoreResultExceptionFailures
    from .test_result_exception_mapping import TestsFlextCoreResultExceptionMapping
    from .test_result_exception_safe_callable import (
        TestsFlextCoreResultExceptionSafeCallable,
    )
    from .test_result_exception_traverse_validation import (
        TestsFlextCoreResultExceptionTraverseValidation,
    )
    from .test_result_factory_dip import TestsFlextCoreResultFactoryDip
    from .test_result_laws import TestsFlextCoreResultLaws
    from .test_result_operations import TestsFlextResultOperations
    from .test_result_recent_behaviors import TestsFlextCoreResultRecentBehaviors
    from .test_result_transforms import TestsFlextResultTransforms
    from .test_result_traverse_resource import TestsFlextResultTraverseResource
    from .test_runtime import TestsFlextCoreRuntime
    from .test_service import TestsFlextService
    from .test_service_bootstrap import TestsFlextCoreServiceBootstrap
    from .test_service_registration_spec import TestsServiceRegistrationSpecOwner
    from .test_settings import TestsFlextCoreSettings, TestsFlextCoreSettingsWorkDir
    from .test_settings_validation_alias import TestsFlextCoreSettingsValidationAlias
    from .test_typings_aliases import LEGACY_GENERIC_NAMES, TestsFlextCoreTypingsAliases
    from .test_typings_containers import TestsFlextCoreTypingsContainers
    from .test_typings_new import TestsFlextCoreTypingsNew
    from .test_typings_validation_numbers import (
        TestsFlextCoreTypingsStrippedStr,
        TestsFlextCoreTypingsValidationNumbers,
    )
    from .test_typings_validation_scalars import TestsFlextCoreTypingsValidationScalars
    from .test_utilities import TestsFlextCoreUtilities
    from .test_utilities_collection_coverage_100 import (
        TestsFlextCoreUtilitiesCollection,
    )
    from .test_utilities_config import TestsFlextCoreUtilitiesConfig
    from .test_utilities_coverage import TestsFlextCoreUtilitiesCoverage
    from .test_utilities_domain import TestsFlextCoreUtilitiesDomain
    from .test_utilities_generators_full_coverage import (
        TestsFlextCoreUtilitiesGenerators,
    )
    from .test_utilities_project_metadata import TestsFlextCoreUtilitiesProjectMetadata
    from .test_utilities_project_metadata_read import (
        TestsFlextUtilitiesProjectMetadataRead,
    )
    from .test_utilities_pydantic_coverage_100 import TestsFlextUtilitiesPydantic
    from .test_utilities_reliability import TestsFlextCoreUtilitiesReliability
    from .test_utilities_runtime_violation_registry_coverage_100 import (
        TestsFlextCoreUtilitiesRuntimeViolationRegistry,
    )
    from .test_utilities_settings_coverage_100 import TestsFlextCoreUtilitiesSettings
    from .test_utilities_text_full_coverage import TestsFlextUtilitiesText
    from .test_utilities_type_guards_coverage_100 import (
        TestsFlextCoreUtilitiesTypeGuards,
    )
    from .test_version import TestsFlextCoreVersion
__all__: tuple[str, ...] = (
    "HANDLER_TYPES",
    "LEGACY_GENERIC_NAMES",
    "LOG_LEVELS",
    "VALIDATION_TYPES",
    "HandlerTypeScenario",
    "TestPackagedConfigWithUserPreferences",
    "TestsFlextBeartypeEngineAnnotations",
    "TestsFlextBeartypeEngineNamespaceHooks",
    "TestsFlextConstantsNew",
    "TestsFlextContainerLifecycle",
    "TestsFlextContainerResolution",
    "TestsFlextCoreBeartypeEngine",
    "TestsFlextCoreBeartypeEngineClawPackages",
    "TestsFlextCoreBeartypeEngineConfig",
    "TestsFlextCoreBeartypeEngineImportHooks",
    "TestsFlextCoreBeartypeEngineRuntime",
    "TestsFlextCoreConfigSettingsCanonical",
    "TestsFlextCoreConstantsProjectMetadata",
    "TestsFlextCoreContainer",
    "TestsFlextCoreContainerConfig",
    "TestsFlextCoreContainerProperties",
    "TestsFlextCoreContainerRegistration",
    "TestsFlextCoreContext",
    "TestsFlextCoreDecorators",
    "TestsFlextCoreDecoratorsCombined",
    "TestsFlextCoreDecoratorsInjectionLogging",
    "TestsFlextCoreDecoratorsRailwayRetry",
    "TestsFlextCoreDeprecationWarnings",
    "TestsFlextCoreDispatcher",
    "TestsFlextCoreEnforcement",
    "TestsFlextCoreEnforcementAccessors",
    "TestsFlextCoreEnforcementLayers",
    "TestsFlextCoreEnforcementNamespace",
    "TestsFlextCoreEnforcementNamespacePart01",
    "TestsFlextCoreEnforcementNamespacePart02",
    "TestsFlextCoreEnforcementReports",
    "TestsFlextCoreEnforcementWarningVisibility",
    "TestsFlextCoreEnumUtilities",
    "TestsFlextCoreExceptions",
    "TestsFlextCoreExceptionsBase",
    "TestsFlextCoreExceptionsPublicMetrics",
    "TestsFlextCoreExceptionsStructuredContracts",
    "TestsFlextCoreExceptionsTypedMetrics",
    "TestsFlextCoreHandlerDecoratorDiscovery",
    "TestsFlextCoreHandlerDiscoveryClass",
    "TestsFlextCoreHandlersFactory",
    "TestsFlextCoreHandlersProperties",
    "TestsFlextCoreHandlersValidationContext",
    "TestsFlextCoreLazyExports",
    "TestsFlextCoreLazyExportsMerge",
    "TestsFlextCoreModels",
    "TestsFlextCoreModelsBaseFullCoverage",
    "TestsFlextCoreModelsContainer",
    "TestsFlextCoreModelsCqrs",
    "TestsFlextCorePublicApiContract",
    "TestsFlextCoreRegistry",
    "TestsFlextCoreResult",
    "TestsFlextCoreResultCallablesFold",
    "TestsFlextCoreResultChainHelpers",
    "TestsFlextCoreResultExceptionFailures",
    "TestsFlextCoreResultExceptionMapping",
    "TestsFlextCoreResultExceptionSafeCallable",
    "TestsFlextCoreResultExceptionTraverseValidation",
    "TestsFlextCoreResultFactoryDip",
    "TestsFlextCoreResultLaws",
    "TestsFlextCoreResultRecentBehaviors",
    "TestsFlextCoreRuntime",
    "TestsFlextCoreServiceBootstrap",
    "TestsFlextCoreSettings",
    "TestsFlextCoreSettingsValidationAlias",
    "TestsFlextCoreSettingsWorkDir",
    "TestsFlextCoreTypingsAliases",
    "TestsFlextCoreTypingsContainers",
    "TestsFlextCoreTypingsNew",
    "TestsFlextCoreTypingsStrippedStr",
    "TestsFlextCoreTypingsValidationNumbers",
    "TestsFlextCoreTypingsValidationScalars",
    "TestsFlextCoreUtilities",
    "TestsFlextCoreUtilitiesCollection",
    "TestsFlextCoreUtilitiesConfig",
    "TestsFlextCoreUtilitiesCoverage",
    "TestsFlextCoreUtilitiesDomain",
    "TestsFlextCoreUtilitiesGenerators",
    "TestsFlextCoreUtilitiesProjectMetadata",
    "TestsFlextCoreUtilitiesReliability",
    "TestsFlextCoreUtilitiesRuntimeViolationRegistry",
    "TestsFlextCoreUtilitiesSettings",
    "TestsFlextCoreUtilitiesTypeGuards",
    "TestsFlextCoreVersion",
    "TestsFlextCoverageLoggings",
    "TestsFlextDecoratorsDiscovery",
    "TestsFlextEnforcementAptHooks",
    "TestsFlextEnforcementCatalog",
    "TestsFlextEnforcementIntegration",
    "TestsFlextEnforcementModels",
    "TestsFlextFacadeFlatSsotAccess",
    "TestsFlextHandlerDecoratorEdges",
    "TestsFlextHandlerDecoratorMetadata",
    "TestsFlextHandlerDiscoveryModule",
    "TestsFlextHandlersDispatch",
    "TestsFlextHandlersLifecycle",
    "TestsFlextLoggings",
    "TestsFlextMixins",
    "TestsFlextModelsProjectMetadata",
    "TestsFlextResultOperations",
    "TestsFlextResultTransforms",
    "TestsFlextResultTraverseResource",
    "TestsFlextService",
    "TestsFlextUtilitiesProjectMetadataRead",
    "TestsFlextUtilitiesPydantic",
    "TestsFlextUtilitiesText",
    "TestsServiceRegistrationSpecOwner",
    "_models",
    "_utilities",
    "c",
    "d",
    "e",
    "h",
    "isolate_import_state",
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
)

_LAZY_IMPORTS = MappingProxyType(
    build_lazy_import_map(
        MappingProxyType({
            "._models": ("_models",),
            "._utilities": ("_utilities",),
            ".test_beartype_engine": ("TestsFlextCoreBeartypeEngine",),
            ".test_beartype_engine_annotations": (
                "TestsFlextBeartypeEngineAnnotations",
            ),
            ".test_beartype_engine_claw_packages": (
                "TestsFlextCoreBeartypeEngineClawPackages",
            ),
            ".test_beartype_engine_config": ("TestsFlextCoreBeartypeEngineConfig",),
            ".test_beartype_engine_import_hooks": (
                "TestsFlextCoreBeartypeEngineImportHooks",
            ),
            ".test_beartype_engine_namespace_hooks": (
                "TestsFlextBeartypeEngineNamespaceHooks",
            ),
            ".test_beartype_engine_runtime": ("TestsFlextCoreBeartypeEngineRuntime",),
            ".test_config_runtime": ("TestsFlextCoreConfigSettingsCanonical",),
            ".test_config_user_preferences": (
                "TestPackagedConfigWithUserPreferences",
                "isolate_import_state",
            ),
            ".test_constants_new": ("TestsFlextConstantsNew",),
            ".test_constants_project_metadata": (
                "TestsFlextCoreConstantsProjectMetadata",
            ),
            ".test_container": ("TestsFlextCoreContainer",),
            ".test_container_config": ("TestsFlextCoreContainerConfig",),
            ".test_container_lifecycle": ("TestsFlextContainerLifecycle",),
            ".test_container_properties": ("TestsFlextCoreContainerProperties",),
            ".test_container_registration": ("TestsFlextCoreContainerRegistration",),
            ".test_container_resolution": ("TestsFlextContainerResolution",),
            ".test_context": ("TestsFlextCoreContext",),
            ".test_coverage_loggings": ("TestsFlextCoverageLoggings",),
            ".test_decorators": ("TestsFlextCoreDecorators",),
            ".test_decorators_combined": ("TestsFlextCoreDecoratorsCombined",),
            ".test_decorators_discovery_full_coverage": (
                "TestsFlextDecoratorsDiscovery",
            ),
            ".test_decorators_injection_logging": (
                "TestsFlextCoreDecoratorsInjectionLogging",
            ),
            ".test_decorators_railway_retry": ("TestsFlextCoreDecoratorsRailwayRetry",),
            ".test_deprecation_warnings": ("TestsFlextCoreDeprecationWarnings",),
            ".test_dispatcher": ("TestsFlextCoreDispatcher",),
            ".test_enforcement": ("TestsFlextCoreEnforcement",),
            ".test_enforcement_accessors": ("TestsFlextCoreEnforcementAccessors",),
            ".test_enforcement_apt_hooks": ("TestsFlextEnforcementAptHooks",),
            ".test_enforcement_catalog": ("TestsFlextEnforcementCatalog",),
            ".test_enforcement_integration": ("TestsFlextEnforcementIntegration",),
            ".test_enforcement_layers": ("TestsFlextCoreEnforcementLayers",),
            ".test_enforcement_models": ("TestsFlextEnforcementModels",),
            ".test_enforcement_namespace": ("TestsFlextCoreEnforcementNamespace",),
            ".test_enforcement_namespace_part_01": (
                "TestsFlextCoreEnforcementNamespacePart01",
            ),
            ".test_enforcement_namespace_part_02": (
                "TestsFlextCoreEnforcementNamespacePart02",
            ),
            ".test_enforcement_reports": ("TestsFlextCoreEnforcementReports",),
            ".test_enforcement_warning_visibility": (
                "TestsFlextCoreEnforcementWarningVisibility",
            ),
            ".test_enum_utilities_coverage_100": ("TestsFlextCoreEnumUtilities",),
            ".test_exceptions": ("TestsFlextCoreExceptions",),
            ".test_exceptions_base": ("TestsFlextCoreExceptionsBase",),
            ".test_exceptions_public_metrics": (
                "TestsFlextCoreExceptionsPublicMetrics",
            ),
            ".test_exceptions_structured_contracts": (
                "TestsFlextCoreExceptionsStructuredContracts",
            ),
            ".test_exceptions_typed_metrics": ("TestsFlextCoreExceptionsTypedMetrics",),
            ".test_handler_decorator_discovery": (
                "TestsFlextCoreHandlerDecoratorDiscovery",
            ),
            ".test_handler_decorator_edges": ("TestsFlextHandlerDecoratorEdges",),
            ".test_handler_decorator_metadata": ("TestsFlextHandlerDecoratorMetadata",),
            ".test_handler_discovery_class": ("TestsFlextCoreHandlerDiscoveryClass",),
            ".test_handler_discovery_module": ("TestsFlextHandlerDiscoveryModule",),
            ".test_handlers_dispatch": ("TestsFlextHandlersDispatch",),
            ".test_handlers_factory": ("TestsFlextCoreHandlersFactory",),
            ".test_handlers_lifecycle": (
                "HANDLER_TYPES",
                "HandlerTypeScenario",
                "TestsFlextHandlersLifecycle",
                "VALIDATION_TYPES",
            ),
            ".test_handlers_properties": ("TestsFlextCoreHandlersProperties",),
            ".test_handlers_validation_context": (
                "TestsFlextCoreHandlersValidationContext",
            ),
            ".test_lazy_exports": ("TestsFlextCoreLazyExports",),
            ".test_lazy_exports_merge": ("TestsFlextCoreLazyExportsMerge",),
            ".test_loggings_full_coverage": ("LOG_LEVELS", "TestsFlextLoggings"),
            ".test_mixins": ("TestsFlextMixins",),
            ".test_models": ("TestsFlextCoreModels",),
            ".test_models_base_full_coverage": (
                "TestsFlextCoreModelsBaseFullCoverage",
            ),
            ".test_models_container": ("TestsFlextCoreModelsContainer",),
            ".test_models_cqrs_full_coverage": ("TestsFlextCoreModelsCqrs",),
            ".test_models_project_metadata": ("TestsFlextModelsProjectMetadata",),
            ".test_project_metadata_facade_access": ("TestsFlextFacadeFlatSsotAccess",),
            ".test_public_api_contract": ("TestsFlextCorePublicApiContract",),
            ".test_registry": ("TestsFlextCoreRegistry",),
            ".test_result": ("TestsFlextCoreResult",),
            ".test_result_callables_fold": ("TestsFlextCoreResultCallablesFold",),
            ".test_result_chain_helpers": ("TestsFlextCoreResultChainHelpers",),
            ".test_result_exception_failures": (
                "TestsFlextCoreResultExceptionFailures",
            ),
            ".test_result_exception_mapping": ("TestsFlextCoreResultExceptionMapping",),
            ".test_result_exception_safe_callable": (
                "TestsFlextCoreResultExceptionSafeCallable",
            ),
            ".test_result_exception_traverse_validation": (
                "TestsFlextCoreResultExceptionTraverseValidation",
            ),
            ".test_result_factory_dip": ("TestsFlextCoreResultFactoryDip",),
            ".test_result_laws": ("TestsFlextCoreResultLaws",),
            ".test_result_operations": ("TestsFlextResultOperations",),
            ".test_result_recent_behaviors": ("TestsFlextCoreResultRecentBehaviors",),
            ".test_result_transforms": ("TestsFlextResultTransforms",),
            ".test_result_traverse_resource": ("TestsFlextResultTraverseResource",),
            ".test_runtime": ("TestsFlextCoreRuntime",),
            ".test_service": ("TestsFlextService",),
            ".test_service_bootstrap": ("TestsFlextCoreServiceBootstrap",),
            ".test_service_registration_spec": ("TestsServiceRegistrationSpecOwner",),
            ".test_settings": (
                "TestsFlextCoreSettings",
                "TestsFlextCoreSettingsWorkDir",
            ),
            ".test_settings_validation_alias": (
                "TestsFlextCoreSettingsValidationAlias",
            ),
            ".test_typings_aliases": (
                "LEGACY_GENERIC_NAMES",
                "TestsFlextCoreTypingsAliases",
            ),
            ".test_typings_containers": ("TestsFlextCoreTypingsContainers",),
            ".test_typings_new": ("TestsFlextCoreTypingsNew",),
            ".test_typings_validation_numbers": (
                "TestsFlextCoreTypingsStrippedStr",
                "TestsFlextCoreTypingsValidationNumbers",
            ),
            ".test_typings_validation_scalars": (
                "TestsFlextCoreTypingsValidationScalars",
            ),
            ".test_utilities": ("TestsFlextCoreUtilities",),
            ".test_utilities_collection_coverage_100": (
                "TestsFlextCoreUtilitiesCollection",
            ),
            ".test_utilities_config": ("TestsFlextCoreUtilitiesConfig",),
            ".test_utilities_coverage": ("TestsFlextCoreUtilitiesCoverage",),
            ".test_utilities_domain": ("TestsFlextCoreUtilitiesDomain",),
            ".test_utilities_generators_full_coverage": (
                "TestsFlextCoreUtilitiesGenerators",
            ),
            ".test_utilities_project_metadata": (
                "TestsFlextCoreUtilitiesProjectMetadata",
            ),
            ".test_utilities_project_metadata_read": (
                "TestsFlextUtilitiesProjectMetadataRead",
            ),
            ".test_utilities_pydantic_coverage_100": ("TestsFlextUtilitiesPydantic",),
            ".test_utilities_reliability": ("TestsFlextCoreUtilitiesReliability",),
            ".test_utilities_runtime_violation_registry_coverage_100": (
                "TestsFlextCoreUtilitiesRuntimeViolationRegistry",
            ),
            ".test_utilities_settings_coverage_100": (
                "TestsFlextCoreUtilitiesSettings",
            ),
            ".test_utilities_text_full_coverage": ("TestsFlextUtilitiesText",),
            ".test_utilities_type_guards_coverage_100": (
                "TestsFlextCoreUtilitiesTypeGuards",
            ),
            ".test_version": ("TestsFlextCoreVersion",),
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
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    )
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
