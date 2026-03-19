# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make codegen
#
"""Pattern implementation unit tests.

Tests for FLEXT Core design patterns:
- Command pattern and CQRS
- Handler patterns
- Validation patterns
- Logging patterns
- Field metadata patterns

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import cleanup_submodule_namespace, lazy_getattr

if TYPE_CHECKING:
    from flext_core.typings import FlextTypes

    from .test_advanced_patterns import TestAdvancedPatterns, TestFunction
    from .test_architectural_patterns import TestArchitecturalPatterns
    from .test_patterns_commands import TestPatternsCommands
    from .test_patterns_logging import EXPECTED_BULK_SIZE, TestPatternsLogging
    from .test_patterns_testing import TestPatternsTesting, pytestmark

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "EXPECTED_BULK_SIZE": (
        "tests.integration.patterns.test_patterns_logging",
        "EXPECTED_BULK_SIZE",
    ),
    "TestAdvancedPatterns": (
        "tests.integration.patterns.test_advanced_patterns",
        "TestAdvancedPatterns",
    ),
    "TestArchitecturalPatterns": (
        "tests.integration.patterns.test_architectural_patterns",
        "TestArchitecturalPatterns",
    ),
    "TestFunction": (
        "tests.integration.patterns.test_advanced_patterns",
        "TestFunction",
    ),
    "TestPatternsCommands": (
        "tests.integration.patterns.test_patterns_commands",
        "TestPatternsCommands",
    ),
    "TestPatternsLogging": (
        "tests.integration.patterns.test_patterns_logging",
        "TestPatternsLogging",
    ),
    "TestPatternsTesting": (
        "tests.integration.patterns.test_patterns_testing",
        "TestPatternsTesting",
    ),
    "pytestmark": ("tests.integration.patterns.test_patterns_testing", "pytestmark"),
}

__all__ = [
    "EXPECTED_BULK_SIZE",
    "TestAdvancedPatterns",
    "TestArchitecturalPatterns",
    "TestFunction",
    "TestPatternsCommands",
    "TestPatternsLogging",
    "TestPatternsTesting",
    "pytestmark",
]


def __getattr__(name: str) -> FlextTypes.ModuleExport:
    """Lazy-load module attributes on first access (PEP 562)."""
    return lazy_getattr(name, _LAZY_IMPORTS, globals(), __name__)


def __dir__() -> list[str]:
    """Return list of available attributes for dir() and autocomplete."""
    return sorted(__all__)


cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)
