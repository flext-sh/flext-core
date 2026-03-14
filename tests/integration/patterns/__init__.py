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
    from tests.integration.patterns.test_architectural_patterns import (
        TestEnterprisePatterns,
        TestEventDrivenPatterns,
    )
    from tests.integration.patterns.test_patterns_commands import (
        CreateUserCommand,
        CreateUserCommandHandler,
        FailingCommand,
        FailingCommandHandler,
        FlextCommandId,
        FlextCommandType,
        TestFlextCommand,
        TestFlextCommandHandler,
        TestFlextCommandResults,
        UpdateUserCommand,
        UpdateUserCommandHandler,
    )
    from tests.integration.patterns.test_patterns_logging import (
        EXPECTED_BULK_SIZE,
        TestFlextContext,
        TestFlextLogger,
        TestFlextLoggerIntegration,
        TestFlextLoggerUsage,
        TestFlextLogLevel,
        assert_result_success,
        make_result_logger,
    )
    from tests.integration.patterns.test_patterns_testing import (
        AssertionBuilder,
        FixtureBuilder,
        FlextTestBuilder,
        GivenWhenThenBuilder,
        MockScenario,
        ParameterizedTestBuilder,
        SuiteBuilder,
        TestAdvancedPatterns,
        TestComprehensiveIntegration,
        TestPerformanceAnalysis,
        TestPropertyBasedPatterns,
        TestRealWorldScenarios,
        arrange_act_assert,
        mark_test_pattern,
        pytestmark,
    )

# Lazy import mapping: export_name -> (module_path, attr_name)
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "AssertionBuilder": (
        "tests.integration.patterns.test_patterns_testing",
        "AssertionBuilder",
    ),
    "CreateUserCommand": (
        "tests.integration.patterns.test_patterns_commands",
        "CreateUserCommand",
    ),
    "CreateUserCommandHandler": (
        "tests.integration.patterns.test_patterns_commands",
        "CreateUserCommandHandler",
    ),
    "EXPECTED_BULK_SIZE": (
        "tests.integration.patterns.test_patterns_logging",
        "EXPECTED_BULK_SIZE",
    ),
    "FailingCommand": (
        "tests.integration.patterns.test_patterns_commands",
        "FailingCommand",
    ),
    "FailingCommandHandler": (
        "tests.integration.patterns.test_patterns_commands",
        "FailingCommandHandler",
    ),
    "FixtureBuilder": (
        "tests.integration.patterns.test_patterns_testing",
        "FixtureBuilder",
    ),
    "FlextCommandId": (
        "tests.integration.patterns.test_patterns_commands",
        "FlextCommandId",
    ),
    "FlextCommandType": (
        "tests.integration.patterns.test_patterns_commands",
        "FlextCommandType",
    ),
    "FlextTestBuilder": (
        "tests.integration.patterns.test_patterns_testing",
        "FlextTestBuilder",
    ),
    "GivenWhenThenBuilder": (
        "tests.integration.patterns.test_patterns_testing",
        "GivenWhenThenBuilder",
    ),
    "MockScenario": (
        "tests.integration.patterns.test_patterns_testing",
        "MockScenario",
    ),
    "ParameterizedTestBuilder": (
        "tests.integration.patterns.test_patterns_testing",
        "ParameterizedTestBuilder",
    ),
    "SuiteBuilder": (
        "tests.integration.patterns.test_patterns_testing",
        "SuiteBuilder",
    ),
    "TestAdvancedPatterns": (
        "tests.integration.patterns.test_patterns_testing",
        "TestAdvancedPatterns",
    ),
    "TestComprehensiveIntegration": (
        "tests.integration.patterns.test_patterns_testing",
        "TestComprehensiveIntegration",
    ),
    "TestEnterprisePatterns": (
        "tests.integration.patterns.test_architectural_patterns",
        "TestEnterprisePatterns",
    ),
    "TestEventDrivenPatterns": (
        "tests.integration.patterns.test_architectural_patterns",
        "TestEventDrivenPatterns",
    ),
    "TestFlextCommand": (
        "tests.integration.patterns.test_patterns_commands",
        "TestFlextCommand",
    ),
    "TestFlextCommandHandler": (
        "tests.integration.patterns.test_patterns_commands",
        "TestFlextCommandHandler",
    ),
    "TestFlextCommandResults": (
        "tests.integration.patterns.test_patterns_commands",
        "TestFlextCommandResults",
    ),
    "TestFlextContext": (
        "tests.integration.patterns.test_patterns_logging",
        "TestFlextContext",
    ),
    "TestFlextLogLevel": (
        "tests.integration.patterns.test_patterns_logging",
        "TestFlextLogLevel",
    ),
    "TestFlextLogger": (
        "tests.integration.patterns.test_patterns_logging",
        "TestFlextLogger",
    ),
    "TestFlextLoggerIntegration": (
        "tests.integration.patterns.test_patterns_logging",
        "TestFlextLoggerIntegration",
    ),
    "TestFlextLoggerUsage": (
        "tests.integration.patterns.test_patterns_logging",
        "TestFlextLoggerUsage",
    ),
    "TestPerformanceAnalysis": (
        "tests.integration.patterns.test_patterns_testing",
        "TestPerformanceAnalysis",
    ),
    "TestPropertyBasedPatterns": (
        "tests.integration.patterns.test_patterns_testing",
        "TestPropertyBasedPatterns",
    ),
    "TestRealWorldScenarios": (
        "tests.integration.patterns.test_patterns_testing",
        "TestRealWorldScenarios",
    ),
    "UpdateUserCommand": (
        "tests.integration.patterns.test_patterns_commands",
        "UpdateUserCommand",
    ),
    "UpdateUserCommandHandler": (
        "tests.integration.patterns.test_patterns_commands",
        "UpdateUserCommandHandler",
    ),
    "arrange_act_assert": (
        "tests.integration.patterns.test_patterns_testing",
        "arrange_act_assert",
    ),
    "assert_result_success": (
        "tests.integration.patterns.test_patterns_logging",
        "assert_result_success",
    ),
    "make_result_logger": (
        "tests.integration.patterns.test_patterns_logging",
        "make_result_logger",
    ),
    "mark_test_pattern": (
        "tests.integration.patterns.test_patterns_testing",
        "mark_test_pattern",
    ),
    "pytestmark": ("tests.integration.patterns.test_patterns_testing", "pytestmark"),
}

__all__ = [
    "EXPECTED_BULK_SIZE",
    "AssertionBuilder",
    "CreateUserCommand",
    "CreateUserCommandHandler",
    "FailingCommand",
    "FailingCommandHandler",
    "FixtureBuilder",
    "FlextCommandId",
    "FlextCommandType",
    "FlextTestBuilder",
    "GivenWhenThenBuilder",
    "MockScenario",
    "ParameterizedTestBuilder",
    "SuiteBuilder",
    "TestAdvancedPatterns",
    "TestComprehensiveIntegration",
    "TestEnterprisePatterns",
    "TestEventDrivenPatterns",
    "TestFlextCommand",
    "TestFlextCommandHandler",
    "TestFlextCommandResults",
    "TestFlextContext",
    "TestFlextLogLevel",
    "TestFlextLogger",
    "TestFlextLoggerIntegration",
    "TestFlextLoggerUsage",
    "TestPerformanceAnalysis",
    "TestPropertyBasedPatterns",
    "TestRealWorldScenarios",
    "UpdateUserCommand",
    "UpdateUserCommandHandler",
    "arrange_act_assert",
    "assert_result_success",
    "make_result_logger",
    "mark_test_pattern",
    "pytestmark",
]


def __getattr__(name: str) -> t.ModuleExport:
    """Lazy-load module attributes on first access (PEP 562)."""
    return lazy_getattr(name, _LAZY_IMPORTS, globals(), __name__)


def __dir__() -> list[str]:
    """Return list of available attributes for dir() and autocomplete."""
    return sorted(__all__)


cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)
