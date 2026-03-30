# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make gen
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

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from tests.integration.patterns import (
        test_advanced_patterns as test_advanced_patterns,
        test_architectural_patterns as test_architectural_patterns,
        test_patterns_commands as test_patterns_commands,
        test_patterns_logging as test_patterns_logging,
        test_patterns_testing as test_patterns_testing,
    )
    from tests.integration.patterns.test_advanced_patterns import (
        TestAdvancedPatterns as TestAdvancedPatterns,
        TestFunction as TestFunction,
    )
    from tests.integration.patterns.test_architectural_patterns import (
        TestArchitecturalPatterns as TestArchitecturalPatterns,
    )
    from tests.integration.patterns.test_patterns_commands import (
        TestPatternsCommands as TestPatternsCommands,
    )
    from tests.integration.patterns.test_patterns_logging import (
        EXPECTED_BULK_SIZE as EXPECTED_BULK_SIZE,
        TestPatternsLogging as TestPatternsLogging,
    )
    from tests.integration.patterns.test_patterns_testing import (
        TestPatternsTesting as TestPatternsTesting,
        pytestmark as pytestmark,
    )

_LAZY_IMPORTS: Mapping[str, Sequence[str]] = {
    "EXPECTED_BULK_SIZE": [
        "tests.integration.patterns.test_patterns_logging",
        "EXPECTED_BULK_SIZE",
    ],
    "TestAdvancedPatterns": [
        "tests.integration.patterns.test_advanced_patterns",
        "TestAdvancedPatterns",
    ],
    "TestArchitecturalPatterns": [
        "tests.integration.patterns.test_architectural_patterns",
        "TestArchitecturalPatterns",
    ],
    "TestFunction": [
        "tests.integration.patterns.test_advanced_patterns",
        "TestFunction",
    ],
    "TestPatternsCommands": [
        "tests.integration.patterns.test_patterns_commands",
        "TestPatternsCommands",
    ],
    "TestPatternsLogging": [
        "tests.integration.patterns.test_patterns_logging",
        "TestPatternsLogging",
    ],
    "TestPatternsTesting": [
        "tests.integration.patterns.test_patterns_testing",
        "TestPatternsTesting",
    ],
    "pytestmark": ["tests.integration.patterns.test_patterns_testing", "pytestmark"],
    "test_advanced_patterns": ["tests.integration.patterns.test_advanced_patterns", ""],
    "test_architectural_patterns": [
        "tests.integration.patterns.test_architectural_patterns",
        "",
    ],
    "test_patterns_commands": ["tests.integration.patterns.test_patterns_commands", ""],
    "test_patterns_logging": ["tests.integration.patterns.test_patterns_logging", ""],
    "test_patterns_testing": ["tests.integration.patterns.test_patterns_testing", ""],
}

_EXPORTS: Sequence[str] = [
    "EXPECTED_BULK_SIZE",
    "TestAdvancedPatterns",
    "TestArchitecturalPatterns",
    "TestFunction",
    "TestPatternsCommands",
    "TestPatternsLogging",
    "TestPatternsTesting",
    "pytestmark",
    "test_advanced_patterns",
    "test_architectural_patterns",
    "test_patterns_commands",
    "test_patterns_logging",
    "test_patterns_testing",
]


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, _EXPORTS)
