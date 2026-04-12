# AUTO-GENERATED FILE — Regenerate with: make gen
"""Integration package."""

from __future__ import annotations

from flext_core.lazy import (
    build_lazy_import_map,
    install_lazy_exports,
    merge_lazy_imports,
)

_LAZY_IMPORTS = merge_lazy_imports(
    (".patterns",),
    build_lazy_import_map(
        {
            ".test_architecture": ("test_architecture",),
            ".test_documented_patterns": ("test_documented_patterns",),
            ".test_examples_execution": ("test_examples_execution",),
            ".test_integration": ("test_integration",),
            ".test_migration_validation": ("test_migration_validation",),
            ".test_service": ("test_service",),
            ".test_service_result_property": ("test_service_result_property",),
            ".test_settings_integration": ("test_settings_integration",),
            ".test_system": ("test_system",),
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
