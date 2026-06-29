"""Lazy export map part 04."""

from __future__ import annotations

from flext_core.lazy import build_lazy_import_map

TESTS_FLEXT_CORE_LAZY_IMPORTS_PART_04 = build_lazy_import_map({
    ".unit.test_enforcement_models": ("TestsFlextEnforcementModels",),
    ".unit.test_enforcement_namespace": ("TestsFlextEnforcementNamespace",),
    ".unit.test_enforcement_reports": ("TestsFlextEnforcementReports",),
    ".unit.test_enum_utilities_coverage_100": ("TestsFlextEnumUtilities",),
    ".unit.test_exceptions_base": ("TestsFlextExceptionsBase",),
    ".unit.test_exceptions_public_metrics": ("TestsFlextCoverageExceptionMetrics",),
    ".unit.test_exceptions_structured_contracts": (
        "TestsFlextCoverageExceptionContracts",
    ),
    ".unit.test_exceptions_typed_metrics": ("TestsFlextExceptionsTypedMetrics",),
    ".unit.test_handler_decorator_edges": ("TestsFlextHandlerDecoratorEdges",),
    ".unit.test_handler_decorator_metadata": ("TestsFlextHandlerDecoratorMetadata",),
    ".unit.test_handler_discovery_class": ("TestsFlextHandlerDiscoveryClass",),
    ".unit.test_handler_discovery_module": ("TestsFlextHandlerDiscoveryModule",),
    ".unit.test_handlers_dispatch": ("TestsFlextHandlersDispatch",),
    ".unit.test_handlers_factory": ("TestsFlextHandlersFactory",),
    ".unit.test_handlers_lifecycle": ("TestsFlextHandlersLifecycle",),
    ".unit.test_handlers_properties": ("TestsFlextHandlersProperties",),
    ".unit.test_handlers_validation_context": ("TestsFlextHandlersValidationContext",),
    ".unit.test_lazy_exports": ("TestsFlextLazy",),
    ".unit.test_lazy_exports_merge": ("TestsFlextLazyMerge",),
    ".unit.test_loggings_full_coverage": ("TestsFlextLoggings",),
    ".unit.test_mixins": ("TestsFlextMixins",),
    ".unit.test_models": ("TestsFlextModelsUnit",),
    ".unit.test_models_base_full_coverage": ("TestsFlextModelsBaseFullCoverage",),
    ".unit.test_models_container": ("TestsFlextModelsContainer",),
    ".unit.test_models_cqrs_full_coverage": ("TestsFlextModelsCqrs",),
    ".unit.test_models_project_metadata": ("TestsFlextModelsProjectMetadata",),
    ".unit.test_project_metadata_facade_access": ("TestsFlextFacadeFlatSsotAccess",),
    ".unit.test_public_api_contract": ("TestsFlextCorePublicApiContract",),
    ".unit.test_registry": ("TestsFlextRegistry",),
    ".unit.test_result_callables_fold": ("TestsFlextResultCallablesFold",),
    ".unit.test_result_chain_helpers": ("TestsFlextResultChainHelpers",),
    ".unit.test_result_exception_failures": ("TestsFlextResultExceptionFailures",),
    ".unit.test_result_exception_mapping": ("TestsFlextResultExceptionMapping",),
    ".unit.test_result_exception_safe_callable": (
        "TestsFlextResultExceptionSafeCallable",
    ),
    ".unit.test_result_exception_traverse_validation": (
        "TestsFlextResultExceptionTraverseValidation",
    ),
    ".unit.test_result_laws": ("TestsFlextResultLaws",),
})

__all__: list[str] = ["TESTS_FLEXT_CORE_LAZY_IMPORTS_PART_04"]
