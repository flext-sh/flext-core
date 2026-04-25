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
            ".patterns.test_advanced_patterns": ("TestsFlextCoreAdvancedPatterns",),
            ".patterns.test_architectural_patterns": (
                "TestsFlextCoreArchitecturalPatterns",
            ),
            ".patterns.test_patterns_commands": ("TestsFlextCorePatternsCommands",),
            ".patterns.test_patterns_logging": ("TestsFlextCorePatternsLogging",),
            ".patterns.test_patterns_testing": ("TestsFlextCorePatternsTesting",),
            ".test_architecture": ("TestsFlextCoreAutomatedArchitecture",),
            ".test_documented_patterns": ("TestsFlextCoreDocumentedPatterns",),
            ".test_examples_execution": ("TestsFlextCoreExamplesExecution",),
            ".test_integration": ("TestsFlextCoreLibraryIntegration",),
            ".test_migration_validation": ("TestsFlextCoreMigrationValidation",),
            ".test_service": ("TestsFlextCoreServiceIntegration",),
            ".test_service_result_property": ("TestsFlextCoreServiceResultProperty",),
            ".test_settings_integration": ("TestsFlextCoreSettingsIntegration",),
            ".test_system": ("TestsFlextCoreSystemIntegration",),
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
