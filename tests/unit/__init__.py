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
            ".base": (
                "TestsFlextCoreServiceBase",
                "s",
            ),
            ".test_args_coverage_100": ("test_args_coverage_100",),
            ".test_beartype_engine": ("test_beartype_engine",),
            ".test_collection_utilities_coverage_100": (
                "test_collection_utilities_coverage_100",
            ),
            ".test_collections_coverage_100": ("test_collections_coverage_100",),
            ".test_constants_new": ("test_constants_new",),
            ".test_container": ("test_container",),
            ".test_container_full_coverage": ("test_container_full_coverage",),
            ".test_context": ("test_context",),
            ".test_context_coverage_100": ("test_context_coverage_100",),
            ".test_context_full_coverage": ("test_context_full_coverage",),
            ".test_coverage_context": ("test_coverage_context",),
            ".test_coverage_exceptions": ("test_coverage_exceptions",),
            ".test_coverage_loggings": ("test_coverage_loggings",),
            ".test_coverage_models": ("test_coverage_models",),
            ".test_coverage_utilities": ("test_coverage_utilities",),
            ".test_decorators": ("test_decorators",),
            ".test_decorators_discovery_full_coverage": (
                "test_decorators_discovery_full_coverage",
            ),
            ".test_decorators_full_coverage": ("test_decorators_full_coverage",),
            ".test_deprecation_warnings": ("test_deprecation_warnings",),
            ".test_di_incremental": ("test_di_incremental",),
            ".test_di_services_access": ("test_di_services_access",),
            ".test_dispatcher_di": ("test_dispatcher_di",),
            ".test_dispatcher_full_coverage": ("test_dispatcher_full_coverage",),
            ".test_dispatcher_minimal": ("test_dispatcher_minimal",),
            ".test_dispatcher_reliability": ("test_dispatcher_reliability",),
            ".test_dispatcher_timeout_coverage_100": (
                "test_dispatcher_timeout_coverage_100",
            ),
            ".test_enforcement": ("test_enforcement",),
            ".test_entity_coverage": ("test_entity_coverage",),
            ".test_enum_utilities_coverage_100": ("test_enum_utilities_coverage_100",),
            ".test_exceptions": ("test_exceptions",),
            ".test_handler_decorator_discovery": ("test_handler_decorator_discovery",),
            ".test_handlers": ("test_handlers",),
            ".test_handlers_full_coverage": ("test_handlers_full_coverage",),
            ".test_lazy_exports": ("test_lazy_exports",),
            ".test_loggings_error_paths_coverage": (
                "test_loggings_error_paths_coverage",
            ),
            ".test_loggings_full_coverage": ("test_loggings_full_coverage",),
            ".test_loggings_strict_returns": ("test_loggings_strict_returns",),
            ".test_mixins": ("test_mixins",),
            ".test_mixins_full_coverage": ("test_mixins_full_coverage",),
            ".test_models": ("test_models",),
            ".test_models_base_full_coverage": ("test_models_base_full_coverage",),
            ".test_models_container": ("test_models_container",),
            ".test_models_context_full_coverage": (
                "test_models_context_full_coverage",
            ),
            ".test_models_cqrs_full_coverage": ("test_models_cqrs_full_coverage",),
            ".test_models_entity_full_coverage": ("test_models_entity_full_coverage",),
            ".test_models_generic_full_coverage": (
                "test_models_generic_full_coverage",
            ),
            ".test_protocols_new": ("test_protocols_new",),
            ".test_registry": ("test_registry",),
            ".test_registry_full_coverage": ("test_registry_full_coverage",),
            ".test_result": ("test_result",),
            ".test_result_additional": ("test_result_additional",),
            ".test_result_coverage_100": ("test_result_coverage_100",),
            ".test_result_exception_carrying": ("test_result_exception_carrying",),
            ".test_result_full_coverage": ("test_result_full_coverage",),
            ".test_runtime": ("test_runtime",),
            ".test_runtime_coverage_100": ("test_runtime_coverage_100",),
            ".test_runtime_full_coverage": ("test_runtime_full_coverage",),
            ".test_service": ("test_service",),
            ".test_service_additional": ("test_service_additional",),
            ".test_service_bootstrap": ("test_service_bootstrap",),
            ".test_service_coverage_100": ("test_service_coverage_100",),
            ".test_settings": ("test_settings",),
            ".test_settings_coverage": ("test_settings_coverage",),
            ".test_typings_full_coverage": ("test_typings_full_coverage",),
            ".test_typings_new": ("test_typings_new",),
            ".test_utilities": ("test_utilities",),
            ".test_utilities_cache_coverage_100": (
                "test_utilities_cache_coverage_100",
            ),
            ".test_utilities_collection_coverage_100": (
                "test_utilities_collection_coverage_100",
            ),
            ".test_utilities_collection_full_coverage": (
                "test_utilities_collection_full_coverage",
            ),
            ".test_utilities_context_full_coverage": (
                "test_utilities_context_full_coverage",
            ),
            ".test_utilities_coverage": ("test_utilities_coverage",),
            ".test_utilities_data_mapper": ("test_utilities_data_mapper",),
            ".test_utilities_domain": ("test_utilities_domain",),
            ".test_utilities_domain_full_coverage": (
                "test_utilities_domain_full_coverage",
            ),
            ".test_utilities_generators_full_coverage": (
                "test_utilities_generators_full_coverage",
            ),
            ".test_utilities_guards_full_coverage": (
                "test_utilities_guards_full_coverage",
            ),
            ".test_utilities_mapper_coverage_100": (
                "test_utilities_mapper_coverage_100",
            ),
            ".test_utilities_mapper_full_coverage": (
                "test_utilities_mapper_full_coverage",
            ),
            ".test_utilities_parser_full_coverage": (
                "test_utilities_parser_full_coverage",
            ),
            ".test_utilities_reliability": ("test_utilities_reliability",),
            ".test_utilities_settings_coverage_100": (
                "test_utilities_settings_coverage_100",
            ),
            ".test_utilities_settings_full_coverage": (
                "test_utilities_settings_full_coverage",
            ),
            ".test_utilities_text_full_coverage": (
                "test_utilities_text_full_coverage",
            ),
            ".test_utilities_type_checker_coverage_100": (
                "test_utilities_type_checker_coverage_100",
            ),
            ".test_utilities_type_guards_coverage_100": (
                "test_utilities_type_guards_coverage_100",
            ),
            ".test_version": ("test_version",),
        },
    ),
    exclude_names=(
        "FlextDispatcher",
        "FlextLogger",
        "FlextRegistry",
        "FlextRuntime",
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
