"""Final Phase 2 coverage push - targeted tests to reach 75% threshold.

Module: flext_core (coverage tests)
Scope: Strategic tests targeting remaining coverage gaps

This test file contains strategic tests targeting the remaining ~86 uncovered lines
needed to reach Phase 2 completion at 75% coverage.

Uses Python 3.13 patterns, FlextTestsUtilities, FlextConstants,
and aggressive parametrization for DRY testing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from flext_core import FlextResult, FlextSettings

from tests.test_utils import assertion_helpers


@dataclass(frozen=True, slots=True)
class ResultChainingScenario:
    """FlextResult chaining test scenario."""

    name: str
    initial_value: str
    operations: list[str]
    expected_success: bool
    expected_value: str | None = None


class CoverageScenarios:
    """Centralized coverage test scenarios using FlextConstants."""

    CHAINING_SCENARIOS: ClassVar[list[ResultChainingScenario]] = [
        ResultChainingScenario(
            "map_chaining",
            "hello",
            ["upper", "append_excl"],
            True,
            "HELLO!",
        ),
        ResultChainingScenario(
            "flat_map_chaining",
            "hi",
            ["double", "double"],
            True,
            "hihihihi",
        ),
        ResultChainingScenario(
            "error_propagation",
            "input",
            ["fail", "upper"],
            False,
            None,
        ),
    ]


class TestPhase2FinalCoveragePush:
    """Strategic tests targeting remaining coverage gaps using FlextTestsUtilities."""

    def test_flext_result_chaining_operations(self) -> None:
        """Test FlextResult chaining with multiple operations."""
        result = FlextResult[str].ok("hello").map(str.upper).map(lambda x: f"{x}!")
        _ = assertion_helpers.assert_flext_result_success(result)
        assert result.value == "HELLO!"

    def test_flext_result_flat_map_chaining(self) -> None:
        """Test FlextResult flat_map chaining."""

        def double_string(s: object) -> FlextResult[object]:
            """Double the string or fail."""
            if not isinstance(s, str):
                return FlextResult[object].fail("Not a string")
            if len(s) > 10:
                return FlextResult[object].fail("Too long")
            return FlextResult[object].ok(s + s)

        result = (
            FlextResult[str].ok("hi").flat_map(double_string).flat_map(double_string)
        )
        _ = assertion_helpers.assert_flext_result_success(result)
        assert result.value == "hihihihi"

    def test_flext_result_error_propagation(self) -> None:
        """Test FlextResult error propagation in chain."""

        def failing_op(s: object) -> FlextResult[object]:
            """Operation that fails."""
            return FlextResult[object].fail("operation failed")

        def upper_func(v: object) -> object:
            """Convert to uppercase."""
            if isinstance(v, str):
                return v.upper()
            return v

        result = FlextResult[str].ok("input").flat_map(failing_op).map(upper_func)
        _ = assertion_helpers.assert_flext_result_failure(result)
        assert result.error == "operation failed"

    def test_config_with_all_field_types(self) -> None:
        """Test config with all field types together."""
        config = FlextSettings(
            app_name="complete_test",
            version="2.0.0",
            debug=True,
            trace=False,
            max_retry_attempts=5,
            timeout_seconds=60.0,
        )
        assert config.app_name == "complete_test"
        assert config.max_retry_attempts == 5
        assert config.timeout_seconds == 60.0

    def test_config_json_serialization(self) -> None:
        """Test config JSON serialization and deserialization."""
        original = FlextSettings(app_name="json_test", version="1.0.0", debug=False)
        config_dict = original.model_dump()
        assert isinstance(config_dict, dict)
        assert config_dict["app_name"] == "json_test"
        # Exclude computed fields that have no setters
        config_dict_filtered = {
            k: v
            for k, v in config_dict.items()
            if k not in {"is_production", "effective_log_level"}
        }
        new_config = FlextSettings(**config_dict_filtered)
        assert new_config.app_name == original.app_name
        assert new_config.version == original.version

    def test_flext_result_unwrap_safe_operations(self) -> None:
        """Test FlextResult unwrap and value operations."""
        success_result = FlextResult[str].ok("success_value")
        assert success_result.value == "success_value"
        assert success_result.value == "success_value"
        assert success_result.value == "success_value"

    def test_flext_result_is_methods(self) -> None:
        """Test FlextResult boolean check methods."""
        success = FlextResult[str].ok("test")
        assert success.is_success is True
        assert success.is_failure is False
        failure: FlextResult[str] = FlextResult[str].fail("error")
        assert failure.is_success is False
        assert failure.is_failure is True

    def test_flext_result_error_access(self) -> None:
        """Test FlextResult error property access."""
        failure: FlextResult[str] = FlextResult[str].fail("test_error")
        assert failure.error == "test_error"

    def test_flext_result_lash_operations(self) -> None:
        """Test FlextResult lash recovery operations."""
        failure: FlextResult[str] = FlextResult[str].fail("original_error")

        def recover_func(error: str) -> FlextResult[str]:
            """Recovery function."""
            return FlextResult[str].ok("recovered_value")

        recovered = failure.lash(recover_func)
        assert recovered.is_success
        assert recovered.value == "recovered_value"

    def test_flext_result_alt_operations(self) -> None:
        """Test FlextResult alt error transformation."""
        failure: FlextResult[str] = FlextResult[str].fail("original_error")

        def transform_error(error: str) -> str:
            """Transform error message."""
            return f"Transformed: {error}"

        transformed = failure.alt(transform_error)
        assert transformed.is_failure
        assert transformed.error == "Transformed: original_error"


__all__ = ["TestPhase2FinalCoveragePush"]
