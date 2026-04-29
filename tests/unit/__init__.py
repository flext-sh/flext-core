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
            "._models.test_base": ("TestsFlextModelsBase",),
            "._models.test_cqrs": ("TestsFlextModelsCQRS",),
            "._models.test_enforcement_sources": (
                "TestsFlextModelsEnforcementSources",
            ),
            "._models.test_entity": ("TestsFlextModelsEntity",),
            "._models.test_exception_params": ("TestsFlextModelsExceptionParams",),
            "._utilities.test_guards": ("TestsFlextUtilitiesGuards",),
            "._utilities.test_mapper": ("TestsFlextUtilitiesMapper",),
            ".base": ("TestsFlextServiceBase",),
            ".test_beartype_engine": ("TestsFlextBeartypeEngine",),
            ".test_constants_new": ("TestsFlextConstantsNew",),
            ".test_constants_project_metadata": ("TestsFlextConstantsProjectMetadata",),
            ".test_container": ("TestsFlextContainer",),
            ".test_context": ("TestsFlextContext",),
            ".test_coverage_exceptions": ("TestsFlextCoverageExceptions",),
            ".test_coverage_loggings": ("TestsFlextCoverageLoggings",),
            ".test_decorators": ("TestsFlextDecoratorsLegacy",),
            ".test_decorators_discovery_full_coverage": (
                "TestsFlextDecoratorsDiscovery",
            ),
            ".test_decorators_full_coverage": ("TestsFlextDecorators",),
            ".test_deprecation_warnings": ("TestsFlextDeprecationWarnings",),
            ".test_dispatcher_di": ("TestsFlextDispatcherDI",),
            ".test_dispatcher_minimal": ("TestsFlextDispatcherMinimal",),
            ".test_dispatcher_reliability": ("TestsFlextDispatcherReliability",),
            ".test_enforcement": ("TestsFlextEnforcement",),
            ".test_enforcement_apt_hooks": ("TestsFlextEnforcementAptHooks",),
            ".test_enforcement_catalog": ("TestsFlextEnforcementCatalog",),
            ".test_enforcement_integration": ("TestsFlextEnforcementIntegration",),
            ".test_enum_utilities_coverage_100": ("TestsFlextEnumUtilities",),
            ".test_exceptions": ("TestsFlextExceptions",),
            ".test_handler_decorator_discovery": (
                "TestsFlextHandlerDecoratorDiscovery",
            ),
            ".test_handlers": ("TestsFlextFlextHandlers",),
            ".test_lazy_exports": ("TestsFlextLazy",),
            ".test_loggings_full_coverage": ("TestsFlextLoggings",),
            ".test_mixins": ("TestsFlextMixins",),
            ".test_models": ("TestsFlextModelsUnit",),
            ".test_models_base_full_coverage": ("TestsFlextModelsBaseFullCoverage",),
            ".test_models_container": ("TestsFlextModelsContainer",),
            ".test_models_cqrs_full_coverage": ("TestsFlextModelsCqrs",),
            ".test_models_project_metadata": ("TestsFlextModelsProjectMetadata",),
            ".test_project_metadata_facade_access": ("TestsFlextFacadeFlatSsotAccess",),
            ".test_registry": ("TestsFlextRegistry",),
            ".test_registry_full_coverage": ("TestsFlextRegistryFullCoverage",),
            ".test_result": ("TestsFlextResult",),
            ".test_result_exception_carrying": ("TestsFlextResultExceptionCarrying",),
            ".test_runtime": ("TestsFlextRuntime",),
            ".test_service": (
                "TestsFlextService",
                "TestsFlextServiceUserData",
                "TestsFlextServiceUserService",
            ),
            ".test_service_bootstrap": ("TestsFlextServiceBootstrap",),
            ".test_service_coverage_100": ("TestsFlextService100Coverage",),
            ".test_settings": ("TestsFlextSettings",),
            ".test_settings_coverage": ("TestsFlextSettingsCoverage",),
            ".test_typings_new": ("TestsFlextTypesUnit",),
            ".test_utilities": ("TestsFlextUtilitiesSmoke",),
            ".test_utilities_collection_coverage_100": (
                "TestsFlextUtilitiesCollection",
            ),
            ".test_utilities_coverage": ("TestsFlextUtilitiesCoverage",),
            ".test_utilities_domain": ("TestsFlextUtilitiesDomain",),
            ".test_utilities_generators_full_coverage": (
                "TestsFlextUtilitiesGenerators",
            ),
            ".test_utilities_project_metadata": ("TestsFlextUtilitiesProjectMetadata",),
            ".test_utilities_reliability": ("TestsFlextUtilitiesReliability",),
            ".test_utilities_settings_coverage_100": ("TestsFlextUtilitiesSettings",),
            ".test_utilities_text_full_coverage": ("TestsFlextUtilitiesText",),
            ".test_utilities_type_guards_coverage_100": (
                "TestsFlextUtilitiesTypeGuards",
            ),
            ".test_version": ("TestsFlextVersion",),
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


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, publish_all=False)
