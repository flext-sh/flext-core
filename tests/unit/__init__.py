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
            "._models.test_base": ("TestsFlextCoreModelsBase",),
            "._models.test_cqrs": ("TestsFlextCoreModelsCQRS",),
            "._models.test_entity": ("TestsFlextCoreModelsEntity",),
            "._models.test_exception_params": ("TestsFlextCoreModelsExceptionParams",),
            "._utilities.test_guards": ("TestsFlextCoreUtilitiesGuards",),
            "._utilities.test_mapper": ("TestsFlextCoreUtilitiesMapper",),
            ".base": ("TestsFlextCoreServiceBase",),
            ".test_beartype_engine": ("TestsFlextCoreBeartypeEngine",),
            ".test_constants_new": ("TestsFlextCoreConstantsNew",),
            ".test_constants_project_metadata": (
                "TestsFlextCoreConstantsProjectMetadata",
            ),
            ".test_container": ("TestsFlextCoreContainer",),
            ".test_context": ("TestsFlextCoreContext",),
            ".test_coverage_exceptions": ("TestsFlextCoreCoverageExceptions",),
            ".test_coverage_loggings": ("TestsFlextCoreCoverageLoggings",),
            ".test_decorators": ("TestsFlextCoreDecoratorsLegacy",),
            ".test_decorators_discovery_full_coverage": (
                "TestsFlextCoreDecoratorsDiscovery",
            ),
            ".test_decorators_full_coverage": ("TestsFlextCoreDecorators",),
            ".test_deprecation_warnings": ("TestsFlextCoreDeprecationWarnings",),
            ".test_dispatcher_di": ("TestsFlextCoreDispatcherDI",),
            ".test_dispatcher_minimal": ("TestsFlextCoreDispatcherMinimal",),
            ".test_dispatcher_reliability": ("TestsFlextCoreDispatcherReliability",),
            ".test_enforcement": ("TestsFlextCoreEnforcement",),
            ".test_enforcement_catalog": ("TestsFlextCoreEnforcementCatalog",),
            ".test_enforcement_integration": ("TestsFlextCoreEnforcementIntegration",),
            ".test_enum_utilities_coverage_100": ("TestsFlextCoreEnumUtilities",),
            ".test_exceptions": ("TestsFlextCoreExceptions",),
            ".test_handler_decorator_discovery": (
                "TestsFlextCoreHandlerDecoratorDiscovery",
            ),
            ".test_handlers": ("TestsFlextCoreFlextHandlers",),
            ".test_lazy_exports": ("TestsFlextCoreLazy",),
            ".test_loggings_full_coverage": ("TestsFlextCoreLoggings",),
            ".test_mixins": ("TestsFlextCoreMixins",),
            ".test_models": ("TestsFlextCoreModelsUnit",),
            ".test_models_base_full_coverage": (
                "TestsFlextCoreModelsBaseFullCoverage",
            ),
            ".test_models_container": ("TestsFlextCoreModelsContainer",),
            ".test_models_cqrs_full_coverage": ("TestsFlextCoreModelsCqrs",),
            ".test_models_project_metadata": ("TestsFlextCoreModelsProjectMetadata",),
            ".test_project_metadata_facade_access": (
                "TestsFlextCoreFacadeFlatSsotAccess",
            ),
            ".test_protocols_project_metadata": (
                "TestsFlextCoreProtocolsProjectMetadata",
            ),
            ".test_registry": ("TestsFlextCoreRegistry",),
            ".test_registry_full_coverage": ("TestsFlextCoreRegistryFullCoverage",),
            ".test_result": ("TestsFlextCoreResult",),
            ".test_result_exception_carrying": (
                "TestsFlextCoreResultExceptionCarrying",
            ),
            ".test_runtime": ("TestsFlextCoreRuntime",),
            ".test_service": (
                "TestService",
                "TestsFlextCoreServiceUserData",
                "TestsFlextCoreServiceUserService",
            ),
            ".test_service_bootstrap": ("TestsFlextCoreServiceBootstrap",),
            ".test_service_coverage_100": ("TestsFlextCoreService100Coverage",),
            ".test_settings": ("TestsFlextCoreSettings",),
            ".test_settings_coverage": ("TestsFlextCoreSettingsCoverage",),
            ".test_typings_new": ("TestsFlextCoreTypesUnit",),
            ".test_utilities": ("TestsFlextCoreUtilitiesSmoke",),
            ".test_utilities_collection_coverage_100": (
                "TestsFlextCoreUtilitiesCollection",
            ),
            ".test_utilities_coverage": ("TestsFlextCoreUtilitiesCoverage",),
            ".test_utilities_domain": ("TestsFlextCoreUtilitiesDomain",),
            ".test_utilities_generators_full_coverage": (
                "TestsFlextCoreUtilitiesGenerators",
            ),
            ".test_utilities_project_metadata": (
                "TestsFlextCoreUtilitiesProjectMetadata",
            ),
            ".test_utilities_reliability": ("TestsFlextCoreUtilitiesReliability",),
            ".test_utilities_settings_coverage_100": (
                "TestsFlextCoreUtilitiesSettings",
            ),
            ".test_utilities_text_full_coverage": ("TestsFlextCoreUtilitiesText",),
            ".test_utilities_type_guards_coverage_100": (
                "TestsFlextCoreUtilitiesTypeGuards",
            ),
            ".test_version": ("TestsFlextCoreVersion",),
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
    ),
    module_name=__name__,
)


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, publish_all=False)
