"""Lazy export map part 01."""

from __future__ import annotations

from flext_core.lazy import build_lazy_import_map

TESTS_FLEXT_CORE_UNIT_LAZY_IMPORTS_PART_01 = build_lazy_import_map({
    "._models.test_base": ("TestsFlextModelsBase",),
    "._models.test_cqrs": ("TestsFlextModelsCQRS",),
    "._models.test_enforcement_sources": ("TestsFlextModelsEnforcementSources",),
    "._models.test_entity": ("TestsFlextModelsEntity",),
    "._models.test_exception_params_core": ("TestsFlextModelsExceptionParamsCore",),
    "._models.test_exception_params_operations": (
        "TestsFlextModelsExceptionParamsOperations",
    ),
    "._models.test_exception_params_resources": (
        "TestsFlextModelsExceptionParamsResources",
    ),
    "._utilities.test_guards": ("TestsFlextUtilitiesGuards",),
    "._utilities.test_mapper": ("TestsFlextUtilitiesMapper",),
    ".test_beartype_engine": ("test_beartype_engine",),
    ".test_beartype_engine_annotations": ("TestsFlextBeartypeEngineAnnotations",),
    ".test_beartype_engine_claw_packages": ("TestsFlextBeartypeEngineClawPackages",),
    ".test_beartype_engine_config": ("TestsFlextBeartypeEngineConfig",),
    ".test_beartype_engine_import_hooks": ("TestsFlextBeartypeEngineImportHooks",),
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
    ".test_decorators_discovery_full_coverage": ("TestsFlextDecoratorsDiscovery",),
    ".test_decorators_full_coverage": ("TestsFlextDecorators",),
    ".test_decorators_injection_logging": ("TestsFlextDecoratorsInjectionLogging",),
    ".test_decorators_railway_retry": ("TestsFlextDecoratorsRailwayRetry",),
    ".test_deprecation_warnings": ("TestsFlextDeprecationWarnings",),
    ".test_dispatcher": ("TestsFlextDispatcher",),
    ".test_enforcement": ("test_enforcement",),
    ".test_enforcement_accessors": ("TestsFlextEnforcementAccessors",),
})

__all__: list[str] = ["TESTS_FLEXT_CORE_UNIT_LAZY_IMPORTS_PART_01"]
