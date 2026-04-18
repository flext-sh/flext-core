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
            ".patterns.test_advanced_patterns": ("TestAdvancedPatterns",),
            ".patterns.test_architectural_patterns": ("TestArchitecturalPatterns",),
            ".patterns.test_patterns_commands": ("TestPatternsCommands",),
            ".patterns.test_patterns_logging": ("TestPatternsLogging",),
            ".patterns.test_patterns_testing": ("TestPatternsTesting",),
            ".test_architecture": ("TestAutomatedArchitecture",),
            ".test_documented_patterns": ("TestDocumentedPatterns",),
            ".test_examples_execution": ("TestExamplesExecution",),
            ".test_integration": ("TestLibraryIntegration",),
            ".test_migration_validation": ("TestMigrationValidation",),
            ".test_service": ("TestServiceIntegration",),
            ".test_service_result_property": ("TestServiceResultProperty",),
            ".test_settings_integration": ("TestFlextSettingsSingletonIntegration",),
            ".test_system": ("TestCompleteFlextSystemIntegration",),
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
