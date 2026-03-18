"""Final Phase 2 coverage push - targeted tests to reach 75% threshold.

Module: flext_core (coverage tests)
Scope: Strategic tests targeting remaining coverage gaps

This test file contains strategic tests targeting the remaining ~86 uncovered lines
needed to reach Phase 2 completion at 75% coverage.

Uses Python 3.13 patterns, u, FlextConstants,
and aggressive parametrization for DRY testing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Annotated, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from flext_core import FlextSettings, r

from ..test_utils import assertion_helpers


class TestPhase2CoverageFinal:
    """Strategic tests targeting remaining coverage gaps using u."""

    class ResultChainingScenario(BaseModel):
        """r chaining test scenario."""

        model_config = ConfigDict(frozen=True)

        name: Annotated[str, Field(description="Result chaining scenario name")]
        initial_value: Annotated[str, Field(description="Initial string value")]
        operations: Annotated[
            list[str], Field(description="Operation names applied in sequence")
        ]
        expected_success: Annotated[bool, Field(description="Expected success state")]
        expected_value: Annotated[
            str | None, Field(default=None, description="Expected resulting value")
        ] = None

    CHAINING_SCENARIOS: ClassVar[list[ResultChainingScenario]] = [
        ResultChainingScenario(
            name="map_chaining",
            initial_value="hello",
            operations=["upper", "append_excl"],
            expected_success=True,
            expected_value="HELLO!",
        ),
        ResultChainingScenario(
            name="flat_map_chaining",
            initial_value="hi",
            operations=["double", "double"],
            expected_success=True,
            expected_value="hihihihi",
        ),
        ResultChainingScenario(
            name="error_propagation",
            initial_value="input",
            operations=["fail", "upper"],
            expected_success=False,
            expected_value=None,
        ),
    ]

    def test_flext_result_chaining_operations(self) -> None:
        """Test r chaining with multiple operations."""
        result = r[str].ok("hello").map(str.upper).map(lambda x: f"{x}!")
        _ = assertion_helpers.assert_flext_result_success(result)
        assert result.value == "HELLO!"

    def test_flext_result_flat_map_chaining(self) -> None:
        """Test r flat_map chaining."""

        def double_string(s: str) -> r[str]:
            """Double the string or fail."""
            if len(s) > 10:
                return r[str].fail("Too long")
            return r[str].ok(s + s)

        result = r[str].ok("hi").flat_map(double_string).flat_map(double_string)
        _ = assertion_helpers.assert_flext_result_success(result)
        assert result.value == "hihihihi"

    def test_flext_result_error_propagation(self) -> None:
        """Test r error propagation in chain."""

        def failing_op(_s: str) -> r[str]:
            """Operation that fails."""
            return r[str].fail("operation failed")

        def upper_func(v: str) -> str:
            """Convert to uppercase."""
            return v.upper()

        result = r[str].ok("input").flat_map(failing_op).map(upper_func)
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
        assert abs(config.timeout_seconds - 60.0) < 1e-09

    def test_config_json_serialization(self) -> None:
        """Test config JSON serialization and deserialization."""
        original = FlextSettings(app_name="json_test", version="1.0.0", debug=False)
        config_dict = original.model_dump()
        assert isinstance(config_dict, dict)
        assert config_dict["app_name"] == "json_test"
        config_dict_filtered = {
            k: v
            for k, v in config_dict.items()
            if k not in {"is_production", "effective_log_level"}
        }
        new_config = FlextSettings(**config_dict_filtered)
        assert new_config.app_name == original.app_name
        assert new_config.version == original.version

    def test_flext_result_unwrap_safe_operations(self) -> None:
        """Test r unwrap and value operations."""
        success_result = r[str].ok("success_value")
        assert success_result.value == "success_value"
        assert success_result.value == "success_value"
        assert success_result.value == "success_value"

    def test_flext_result_is_methods(self) -> None:
        """Test r boolean check methods."""
        success = r[str].ok("test")
        assert success.is_success is True
        assert success.is_failure is False
        failure: r[str] = r[str].fail("error")
        assert failure.is_success is False
        assert failure.is_failure is True

    def test_flext_result_error_access(self) -> None:
        """Test r error property access."""
        failure: r[str] = r[str].fail("test_error")
        assert failure.error == "test_error"

    def test_flext_result_lash_operations(self) -> None:
        """Test r lash recovery operations."""
        failure: r[str] = r[str].fail("original_error")

        def recover_func(_error: str) -> r[str]:
            """Recovery function."""
            return r[str].ok("recovered_value")

        recovered = failure.lash(recover_func)
        assert recovered.is_success
        assert recovered.value == "recovered_value"

    def test_flext_result_alt_operations(self) -> None:
        """Test r alt error transformation."""
        failure: r[str] = r[str].fail("original_error")

        def transform_error(error: str) -> str:
            """Transform error message."""
            return f"Transformed: {error}"

        transformed = failure.map_error(transform_error)
        assert transformed.is_failure
        assert transformed.error == "Transformed: original_error"


__all__ = ["TestPhase2CoverageFinal"]
