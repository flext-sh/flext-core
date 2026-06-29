"""Lazy export map part 02."""

from __future__ import annotations

from flext_core.lazy import build_lazy_import_map

TESTS_FLEXT_CORE_UNIT_LAZY_IMPORTS_PART_02 = build_lazy_import_map({
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
    ".test_exceptions_structured_contracts": ("TestsFlextCoverageExceptionContracts",),
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
    ".test_handlers_validation_context": ("TestsFlextHandlersValidationContext",),
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
})

__all__: list[str] = ["TESTS_FLEXT_CORE_UNIT_LAZY_IMPORTS_PART_02"]
