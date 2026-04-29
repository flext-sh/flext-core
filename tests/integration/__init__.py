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
            ".patterns.test_advanced_patterns": ("TestsFlextAdvancedPatterns",),
            ".patterns.test_architectural_patterns": (
                "TestsFlextArchitecturalPatterns",
            ),
            ".patterns.test_patterns_commands": ("TestsFlextPatternsCommands",),
            ".patterns.test_patterns_logging": ("TestsFlextPatternsLogging",),
            ".patterns.test_patterns_testing": ("TestsFlextPatternsTesting",),
            ".test_architecture": ("TestsFlextAutomatedArchitecture",),
            ".test_documented_patterns": ("TestsFlextDocumentedPatterns",),
            ".test_examples_execution": ("TestsFlextExamplesExecution",),
            ".test_integration": ("TestsFlextLibraryIntegration",),
            ".test_migration_validation": ("TestsFlextMigrationValidation",),
            ".test_service": ("TestsFlextServiceIntegration",),
            ".test_service_result_property": ("TestsFlextServiceResultProperty",),
            ".test_settings_integration": ("TestsFlextSettingsIntegration",),
            ".test_system": ("TestsFlextSystemIntegration",),
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
