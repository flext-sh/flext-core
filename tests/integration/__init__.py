# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core package."""

from __future__ import annotations

import typing as _t

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if _t.TYPE_CHECKING:
    from flext_core.test_architecture import TestAutomatedArchitecture
    from flext_core.test_config_integration import TestFlextSettingsSingletonIntegration
    from flext_core.test_documented_patterns import TestDocumentedPatterns
    from flext_core.test_examples_execution import (
        REPO_ROOT,
        test_public_example_scripts_match_golden_files,
    )
    from flext_core.test_integration import TestLibraryIntegration, pytestmark
    from flext_core.test_migration_validation import TestMigrationValidation
    from flext_core.test_service import TestService
    from flext_core.test_service_result_property import TestServiceResultProperty
    from flext_core.test_system import TestCompleteFlextSystemIntegration
_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".test_architecture": ("TestAutomatedArchitecture",),
        ".test_config_integration": ("TestFlextSettingsSingletonIntegration",),
        ".test_documented_patterns": ("TestDocumentedPatterns",),
        ".test_examples_execution": (
            "REPO_ROOT",
            "test_public_example_scripts_match_golden_files",
        ),
        ".test_integration": (
            "TestLibraryIntegration",
            "pytestmark",
        ),
        ".test_migration_validation": ("TestMigrationValidation",),
        ".test_service": ("TestService",),
        ".test_service_result_property": ("TestServiceResultProperty",),
        ".test_system": ("TestCompleteFlextSystemIntegration",),
    },
)


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS)

__all__ = [
    "REPO_ROOT",
    "TestAutomatedArchitecture",
    "TestCompleteFlextSystemIntegration",
    "TestDocumentedPatterns",
    "TestFlextSettingsSingletonIntegration",
    "TestLibraryIntegration",
    "TestMigrationValidation",
    "TestService",
    "TestServiceResultProperty",
    "pytestmark",
    "test_public_example_scripts_match_golden_files",
]
