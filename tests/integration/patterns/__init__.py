# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make gen
#
"""Patterns package."""

from __future__ import annotations

import typing as _t

from flext_core.constants import FlextConstants as c
from flext_core.decorators import FlextDecorators as d
from flext_core.exceptions import FlextExceptions as e
from flext_core.handlers import FlextHandlers as h
from flext_core.lazy import install_lazy_exports
from flext_core.mixins import FlextMixins as x
from flext_core.models import FlextModels as m
from flext_core.protocols import FlextProtocols as p
from flext_core.result import FlextResult as r
from flext_core.service import FlextService as s
from flext_core.typings import FlextTypes as t
from flext_core.utilities import FlextUtilities as u
from tests.integration.patterns.test_advanced_patterns import (
    TestAdvancedPatterns,
    TestFunction,
)
from tests.integration.patterns.test_architectural_patterns import (
    TestArchitecturalPatterns,
)
from tests.integration.patterns.test_patterns_commands import TestPatternsCommands
from tests.integration.patterns.test_patterns_logging import (
    EXPECTED_BULK_SIZE,
    TestPatternsLogging,
)
from tests.integration.patterns.test_patterns_testing import (
    P,
    R,
    TestPatternsTesting,
    pytestmark,
)

if _t.TYPE_CHECKING:
    import tests.integration.patterns.test_advanced_patterns as _tests_integration_patterns_test_advanced_patterns

    test_advanced_patterns = _tests_integration_patterns_test_advanced_patterns
    import tests.integration.patterns.test_architectural_patterns as _tests_integration_patterns_test_architectural_patterns

    test_architectural_patterns = (
        _tests_integration_patterns_test_architectural_patterns
    )
    import tests.integration.patterns.test_patterns_commands as _tests_integration_patterns_test_patterns_commands

    test_patterns_commands = _tests_integration_patterns_test_patterns_commands
    import tests.integration.patterns.test_patterns_logging as _tests_integration_patterns_test_patterns_logging

    test_patterns_logging = _tests_integration_patterns_test_patterns_logging
    import tests.integration.patterns.test_patterns_testing as _tests_integration_patterns_test_patterns_testing

    test_patterns_testing = _tests_integration_patterns_test_patterns_testing

    _ = (
        EXPECTED_BULK_SIZE,
        P,
        R,
        TestAdvancedPatterns,
        TestArchitecturalPatterns,
        TestFunction,
        TestPatternsCommands,
        TestPatternsLogging,
        TestPatternsTesting,
        c,
        d,
        e,
        h,
        m,
        p,
        pytestmark,
        r,
        s,
        t,
        test_advanced_patterns,
        test_architectural_patterns,
        test_patterns_commands,
        test_patterns_logging,
        test_patterns_testing,
        u,
        x,
    )
_LAZY_IMPORTS = {
    "EXPECTED_BULK_SIZE": "tests.integration.patterns.test_patterns_logging",
    "P": "tests.integration.patterns.test_patterns_testing",
    "R": "tests.integration.patterns.test_patterns_testing",
    "TestAdvancedPatterns": "tests.integration.patterns.test_advanced_patterns",
    "TestArchitecturalPatterns": "tests.integration.patterns.test_architectural_patterns",
    "TestFunction": "tests.integration.patterns.test_advanced_patterns",
    "TestPatternsCommands": "tests.integration.patterns.test_patterns_commands",
    "TestPatternsLogging": "tests.integration.patterns.test_patterns_logging",
    "TestPatternsTesting": "tests.integration.patterns.test_patterns_testing",
    "c": ("flext_core.constants", "FlextConstants"),
    "d": ("flext_core.decorators", "FlextDecorators"),
    "e": ("flext_core.exceptions", "FlextExceptions"),
    "h": ("flext_core.handlers", "FlextHandlers"),
    "m": ("flext_core.models", "FlextModels"),
    "p": ("flext_core.protocols", "FlextProtocols"),
    "pytestmark": "tests.integration.patterns.test_patterns_testing",
    "r": ("flext_core.result", "FlextResult"),
    "s": ("flext_core.service", "FlextService"),
    "t": ("flext_core.typings", "FlextTypes"),
    "test_advanced_patterns": "tests.integration.patterns.test_advanced_patterns",
    "test_architectural_patterns": "tests.integration.patterns.test_architectural_patterns",
    "test_patterns_commands": "tests.integration.patterns.test_patterns_commands",
    "test_patterns_logging": "tests.integration.patterns.test_patterns_logging",
    "test_patterns_testing": "tests.integration.patterns.test_patterns_testing",
    "u": ("flext_core.utilities", "FlextUtilities"),
    "x": ("flext_core.mixins", "FlextMixins"),
}

__all__ = [
    "EXPECTED_BULK_SIZE",
    "P",
    "R",
    "TestAdvancedPatterns",
    "TestArchitecturalPatterns",
    "TestFunction",
    "TestPatternsCommands",
    "TestPatternsLogging",
    "TestPatternsTesting",
    "c",
    "d",
    "e",
    "h",
    "m",
    "p",
    "pytestmark",
    "r",
    "s",
    "t",
    "test_advanced_patterns",
    "test_architectural_patterns",
    "test_patterns_commands",
    "test_patterns_logging",
    "test_patterns_testing",
    "u",
    "x",
]


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS)
