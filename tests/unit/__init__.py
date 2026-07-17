# AUTO-GENERATED FILE — Regenerate with: make gen
"""Unit package."""

from __future__ import annotations

from flext_core.lazy import (
    build_lazy_import_map,
    install_lazy_exports,
    merge_lazy_imports,
)

_LAZY_IMPORTS = merge_lazy_imports(
    (
        "._models",
        "._utilities",
    ),
    build_lazy_import_map(
        {
            "._models": ("_models",),
            "._models.test_base": (
                "Sample",
                "SampleValue",
                "TestsFlextCoreBase",
            ),
            "._models.test_cqrs": ("TestsFlextCoreCqrs",),
            "._models.test_enforcement_sources": ("TestsFlextCoreEnforcementSources",),
            "._models.test_entity": ("TestsFlextCoreEntity",),
            "._models.test_exception_params_core": (
                "TestsFlextModelsExceptionParamsCore",
            ),
            "._models.test_exception_params_operations": (
                "TestsFlextCoreExceptionParamsOperations",
            ),
            "._models.test_exception_params_resources": (
                "TestsFlextModelsExceptionParamsResources",
            ),
            "._utilities": ("_utilities",),
            "._utilities.test_guards": ("TestsFlextCoreGuards",),
            "._utilities.test_mapper": ("TestsFlextCoreMapper",),
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
            ".test_enforcement_plugin": ("TestsFlextCoreEnforcementPlugin",),
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
            ".test_handlers": ("test_handlers",),
            ".test_handlers_dispatch": ("TestsFlextHandlersDispatch",),
            ".test_handlers_factory": ("TestsFlextCoreHandlersFactory",),
            ".test_handlers_lifecycle": ("TestsFlextHandlersLifecycle",),
            ".test_handlers_properties": ("TestsFlextCoreHandlersProperties",),
            ".test_handlers_validation_context": (
                "TestsFlextCoreHandlersValidationContext",
            ),
            ".test_lazy_exports": ("TestsFlextCoreLazyExports",),
            ".test_lazy_exports_merge": ("TestsFlextCoreLazyExportsMerge",),
            ".test_loggings_full_coverage": ("TestsFlextLoggings",),
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
            ".test_result_laws": ("TestsFlextCoreResultLaws",),
            ".test_result_operations": ("TestsFlextResultOperations",),
            ".test_result_recent_behaviors": ("TestsFlextCoreResultRecentBehaviors",),
            ".test_result_transforms": ("TestsFlextResultTransforms",),
            ".test_result_traverse_resource": ("TestsFlextResultTraverseResource",),
            ".test_runtime": ("TestsFlextCoreRuntime",),
            ".test_service": ("TestsFlextService",),
            ".test_service_bootstrap": ("TestsFlextCoreServiceBootstrap",),
            ".test_settings": ("TestsFlextCoreSettings",),
            ".test_settings_validation_alias": (
                "TestsFlextCoreSettingsValidationAlias",
            ),
            ".test_typings_aliases": ("TestsFlextCoreTypingsAliases",),
            ".test_typings_containers": ("TestsFlextCoreTypingsContainers",),
            ".test_typings_new": ("TestsFlextCoreTypingsNew",),
            ".test_typings_validation_numbers": (
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
