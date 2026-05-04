# AUTO-GENERATED FILE — Regenerate with: make gen
"""Integration package."""

from __future__ import annotations

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".test_architecture": ("TestsFlextAutomatedArchitecture",),
        ".test_documented_patterns": ("TestsFlextDocumentedPatterns",),
        ".test_examples_execution": ("TestsFlextExamplesExecution",),
        ".test_integration": ("TestsFlextLibraryIntegration",),
        ".test_migration_validation": ("TestsFlextMigrationValidation",),
        ".test_service": ("TestsFlextServiceIntegration",),
        ".test_service_result_property": ("test_service_result_property",),
        ".test_settings_integration": ("TestsFlextSettingsIntegration",),
        ".test_system": ("TestsFlextSystemIntegration",),
    },
)


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, publish_all=False)
