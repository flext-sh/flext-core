"""Lazy export map part 03."""

from __future__ import annotations

from flext_core.lazy import build_lazy_import_map

TESTS_FLEXT_CORE_LAZY_IMPORTS_PART_03 = build_lazy_import_map({
    ".unit._models.test_base": ("TestsFlextModelsBase",),
    ".unit._models.test_cqrs": ("TestsFlextModelsCQRS",),
    ".unit._models.test_enforcement_sources": ("TestsFlextModelsEnforcementSources",),
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
    ".unit.test_beartype_engine_annotations": ("TestsFlextBeartypeEngineAnnotations",),
    ".unit.test_beartype_engine_claw_packages": (
        "TestsFlextBeartypeEngineClawPackages",
    ),
    ".unit.test_beartype_engine_config": ("TestsFlextBeartypeEngineConfig",),
    ".unit.test_beartype_engine_import_hooks": ("TestsFlextBeartypeEngineImportHooks",),
    ".unit.test_beartype_engine_namespace_hooks": (
        "TestsFlextBeartypeEngineNamespaceHooks",
    ),
    ".unit.test_beartype_engine_runtime": ("TestsFlextBeartypeEngineRuntime",),
    ".unit.test_constants_new": ("TestsFlextConstantsNew",),
    ".unit.test_constants_project_metadata": ("TestsFlextConstantsProjectMetadata",),
    ".unit.test_container_config": ("TestsFlextContainerConfig",),
    ".unit.test_container_lifecycle": ("TestsFlextContainerLifecycle",),
    ".unit.test_container_properties": ("TestsFlextContainerProperties",),
    ".unit.test_container_registration": ("TestsFlextContainerRegistration",),
    ".unit.test_container_resolution": ("TestsFlextContainerResolution",),
    ".unit.test_context": ("TestsFlextContext",),
    ".unit.test_coverage_loggings": ("TestsFlextCoverageLoggings",),
    ".unit.test_decorators_combined": ("TestsFlextDecoratorsCombined",),
    ".unit.test_decorators_discovery_full_coverage": ("TestsFlextDecoratorsDiscovery",),
    ".unit.test_decorators_full_coverage": ("TestsFlextDecorators",),
    ".unit.test_decorators_injection_logging": (
        "TestsFlextDecoratorsInjectionLogging",
    ),
    ".unit.test_decorators_railway_retry": ("TestsFlextDecoratorsRailwayRetry",),
    ".unit.test_deprecation_warnings": ("TestsFlextDeprecationWarnings",),
    ".unit.test_dispatcher": ("TestsFlextDispatcher",),
    ".unit.test_enforcement_accessors": ("TestsFlextEnforcementAccessors",),
    ".unit.test_enforcement_apt_hooks": ("TestsFlextEnforcementAptHooks",),
    ".unit.test_enforcement_catalog": ("TestsFlextEnforcementCatalog",),
    ".unit.test_enforcement_integration": ("TestsFlextEnforcementIntegration",),
    ".unit.test_enforcement_layers": ("TestsFlextEnforcementLayers",),
})

__all__: list[str] = ["TESTS_FLEXT_CORE_LAZY_IMPORTS_PART_03"]
