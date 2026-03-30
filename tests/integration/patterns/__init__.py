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
        test_advanced_patterns,
        test_architectural_patterns,
        test_patterns_commands,
        test_patterns_logging,
        test_patterns_testing,
    )
    from tests.integration.patterns.test_advanced_patterns import *
    from tests.integration.patterns.test_architectural_patterns import *
    from tests.integration.patterns.test_patterns_commands import *
    from tests.integration.patterns.test_patterns_logging import *
    from tests.integration.patterns.test_patterns_testing import *

_LAZY_IMPORTS: Mapping[str, str | Sequence[str]] = {
    "EXPECTED_BULK_SIZE": "tests.integration.patterns.test_patterns_logging",
    "TestAdvancedPatterns": "tests.integration.patterns.test_advanced_patterns",
    "TestArchitecturalPatterns": "tests.integration.patterns.test_architectural_patterns",
    "TestFunction": "tests.integration.patterns.test_advanced_patterns",
    "TestPatternsCommands": "tests.integration.patterns.test_patterns_commands",
    "TestPatternsLogging": "tests.integration.patterns.test_patterns_logging",
    "TestPatternsTesting": "tests.integration.patterns.test_patterns_testing",
    "pytestmark": "tests.integration.patterns.test_patterns_testing",
    "test_advanced_patterns": "tests.integration.patterns.test_advanced_patterns",
    "test_architectural_patterns": "tests.integration.patterns.test_architectural_patterns",
    "test_patterns_commands": "tests.integration.patterns.test_patterns_commands",
    "test_patterns_logging": "tests.integration.patterns.test_patterns_logging",
    "test_patterns_testing": "tests.integration.patterns.test_patterns_testing",
}


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, sorted(_LAZY_IMPORTS))
