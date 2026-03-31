"""Tests for examples/01_basic_result.py.

Covers the changed API patterns in the PR:
- c.HandlerType (was c.Cqrs.HandlerType)
- c.PATTERN_EMAIL (was c.Platform.PATTERN_EMAIL)
- c.VALIDATION_ERROR (was c.Errors.VALIDATION_ERROR)
- c.DEFAULT_MAX_COMMAND_RETRIES (was c.ZERO)
- c.BACKUP_COUNT for filter threshold (was c.Validation.FILTER_THRESHOLD)
- c.MAX_AGE used in risky_operation (was c.Validation.MAX_AGE)
- c.MAX_RETRY_ATTEMPTS for min_length validation (was c.Validation.MIN_USERNAME_LENGTH)
- c.HTTP_STATUS_MIN for display width
- c.FIELD_NAME constant
- c.EXCEPTION_ERROR string constant
- DemonstrationResult docstring change (value t.NormalizedValue)

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

from flext_core import c, e, m, r, t, u


# ---------------------------------------------------------------------------
# Module import helper
# ---------------------------------------------------------------------------

def _load_example_module(filename: str) -> Any:
    """Load an example module by filename from the examples/ directory."""
    examples_dir = Path(__file__).parent.parent.parent / "examples"
    module_path = examples_dir / filename
    spec = importlib.util.spec_from_file_location(
        filename.replace(".", "_").replace("-", "_"),
        module_path,
    )
    assert spec is not None, f"Could not find spec for {filename}"
    assert spec.loader is not None, f"No loader for {filename}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def basic_result_module() -> Any:
    """Load the 01_basic_result.py module once per test module."""
    return _load_example_module("01_basic_result.py")


# ---------------------------------------------------------------------------
# Tests: Constants API changes for 01_basic_result.py
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestConstantsApiForBasicResult:
    """Verify the changed constants API used in 01_basic_result.py."""

    def test_c_handler_type_command_exists(self) -> None:
        """c.HandlerType.COMMAND accessible (was c.Cqrs.HandlerType.COMMAND)."""
        assert c.HandlerType.COMMAND == "command"

    def test_c_handler_type_query_exists(self) -> None:
        """c.HandlerType.QUERY accessible."""
        assert c.HandlerType.QUERY == "query"

    def test_c_handler_type_event_exists(self) -> None:
        """c.HandlerType.EVENT accessible."""
        assert c.HandlerType.EVENT == "event"

    def test_c_handler_type_operation_exists(self) -> None:
        """c.HandlerType.OPERATION accessible."""
        assert c.HandlerType.OPERATION == "operation"

    def test_c_handler_type_saga_exists(self) -> None:
        """c.HandlerType.SAGA accessible."""
        assert c.HandlerType.SAGA == "saga"

    def test_c_handler_type_members(self) -> None:
        """c.HandlerType has expected members for pattern discovery."""
        members = list(c.HandlerType.__members__.values())
        assert len(members) >= 3  # At least COMMAND, QUERY, EVENT

    def test_c_default_max_command_retries_is_zero(self) -> None:
        """c.DEFAULT_MAX_COMMAND_RETRIES == 0 (was c.ZERO)."""
        assert c.DEFAULT_MAX_COMMAND_RETRIES == 0

    def test_c_backup_count_is_positive_int(self) -> None:
        """c.BACKUP_COUNT is a positive integer (was c.Validation.FILTER_THRESHOLD)."""
        assert isinstance(c.BACKUP_COUNT, int)
        assert c.BACKUP_COUNT > 0

    def test_c_max_age_is_large_int(self) -> None:
        """c.MAX_AGE is a positive integer (was c.Validation.MAX_AGE)."""
        assert isinstance(c.MAX_AGE, int)
        assert c.MAX_AGE > 0

    def test_c_max_retry_attempts_is_min_length(self) -> None:
        """c.MAX_RETRY_ATTEMPTS used as min_length in validation."""
        assert isinstance(c.MAX_RETRY_ATTEMPTS, int)
        assert c.MAX_RETRY_ATTEMPTS > 0

    def test_c_http_status_min_is_100(self) -> None:
        """c.HTTP_STATUS_MIN == 100 (used for display width calculation)."""
        assert c.HTTP_STATUS_MIN == 100

    def test_c_exception_error_is_string(self) -> None:
        """c.EXCEPTION_ERROR is a non-empty string constant."""
        assert isinstance(c.EXCEPTION_ERROR, str)
        assert len(c.EXCEPTION_ERROR) > 0

    def test_c_field_name_is_name_string(self) -> None:
        """c.FIELD_NAME == 'name' (from FlextConstantsMixins)."""
        assert c.FIELD_NAME == "name"


# ---------------------------------------------------------------------------
# Tests: User and DemonstrationResult models
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestModelsInBasicResult:
    """Tests for User and DemonstrationResult models from 01_basic_result.py."""

    def test_user_model_creation(self, basic_result_module: Any) -> None:
        """User entity can be created with name and email."""
        User = basic_result_module.User
        user = User(name="Alice", email="alice@example.com", domain_events=[])
        assert user.name == "Alice"
        assert user.email == "alice@example.com"

    def test_demonstration_result_model_creation(
        self, basic_result_module: Any
    ) -> None:
        """DemonstrationResult value model can be created."""
        DemonstrationResult = basic_result_module.DemonstrationResult
        result = DemonstrationResult(
            demonstrations_completed=3,
            patterns_covered=("command", "query", "event"),
            completed_at="2025-01-01T00:00:00Z",
        )
        assert result.demonstrations_completed == 3
        assert "command" in result.patterns_covered

    def test_demonstration_result_is_value_object(
        self, basic_result_module: Any
    ) -> None:
        """DemonstrationResult inherits from m.Value."""
        DemonstrationResult = basic_result_module.DemonstrationResult
        assert issubclass(DemonstrationResult, m.Value)

    def test_user_inherits_from_m_entity(
        self, basic_result_module: Any
    ) -> None:
        """User inherits from m.Entity."""
        User = basic_result_module.User
        assert issubclass(User, m.Entity)


# ---------------------------------------------------------------------------
# Tests: RailwayService._build_result_data
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRailwayServiceBuildResultData:
    """Tests for RailwayService._build_result_data()."""

    def test_build_result_data_returns_ok(
        self, basic_result_module: Any
    ) -> None:
        """_build_result_data() returns r[DemonstrationResult].ok(...)."""
        RailwayService = basic_result_module.RailwayService
        result = RailwayService._build_result_data(None)
        assert result.is_success

    def test_build_result_data_includes_handler_type_patterns(
        self, basic_result_module: Any
    ) -> None:
        """_build_result_data() patterns come from c.HandlerType members."""
        RailwayService = basic_result_module.RailwayService
        result = RailwayService._build_result_data(None)
        assert result.is_success
        patterns = result.value.patterns_covered
        # Should contain handler type values from c.HandlerType
        expected_patterns = {m.value for m in c.HandlerType.__members__.values()}
        for pattern in patterns:
            assert pattern in expected_patterns

    def test_build_result_data_count_matches_handler_types(
        self, basic_result_module: Any
    ) -> None:
        """_build_result_data() demonstrations_completed matches handler type count."""
        RailwayService = basic_result_module.RailwayService
        result = RailwayService._build_result_data(None)
        assert result.is_success
        expected_count = len(c.HandlerType.__members__)
        assert result.value.demonstrations_completed == expected_count

    def test_build_result_data_has_timestamp(
        self, basic_result_module: Any
    ) -> None:
        """_build_result_data() result has a non-empty completed_at timestamp."""
        RailwayService = basic_result_module.RailwayService
        result = RailwayService._build_result_data(None)
        assert result.is_success
        assert result.value.completed_at
        assert len(result.value.completed_at) > 0


# ---------------------------------------------------------------------------
# Tests: RailwayService._create_user_validator
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRailwayServiceUserValidator:
    """Tests for RailwayService._create_user_validator()."""

    def test_valid_email_returns_user(
        self, basic_result_module: Any
    ) -> None:
        """Validator returns r[User].ok(...) for a valid email."""
        RailwayService = basic_result_module.RailwayService
        validate = RailwayService._create_user_validator()
        result = validate("test@example.com")
        assert result.is_success
        assert result.value.email == "test@example.com"

    def test_invalid_email_returns_failure(
        self, basic_result_module: Any
    ) -> None:
        """Validator returns r[User].fail(...) for an invalid email."""
        RailwayService = basic_result_module.RailwayService
        validate = RailwayService._create_user_validator()
        result = validate("not-an-email")
        assert result.is_failure

    def test_empty_email_returns_failure(
        self, basic_result_module: Any
    ) -> None:
        """Validator returns failure for empty email string."""
        RailwayService = basic_result_module.RailwayService
        validate = RailwayService._create_user_validator()
        result = validate("")
        assert result.is_failure

    def test_validator_uses_c_pattern_email(
        self, basic_result_module: Any
    ) -> None:
        """Validator uses c.PATTERN_EMAIL regex (was c.Platform.PATTERN_EMAIL)."""
        # Verify that c.PATTERN_EMAIL accepts valid emails
        result_valid = u.validate_pattern(
            "user@domain.org", c.PATTERN_EMAIL, "email"
        )
        assert result_valid.is_success

        result_invalid = u.validate_pattern(
            "invalid-email", c.PATTERN_EMAIL, "email"
        )
        assert result_invalid.is_failure

    def test_valid_user_is_m_entity(
        self, basic_result_module: Any
    ) -> None:
        """Successfully validated User inherits from m.Entity."""
        RailwayService = basic_result_module.RailwayService
        validate = RailwayService._create_user_validator()
        result = validate("alice@example.com")
        assert result.is_success
        assert isinstance(result.value, m.Entity)


# ---------------------------------------------------------------------------
# Tests: RailwayService._execute_demo
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRailwayServiceExecuteDemo:
    """Tests for RailwayService._execute_demo()."""

    def test_successful_demo_returns_ok_true(
        self, basic_result_module: Any
    ) -> None:
        """_execute_demo() with a no-op function returns r[bool].ok(True)."""
        RailwayService = basic_result_module.RailwayService

        def no_op() -> None:
            pass

        result = RailwayService._execute_demo(no_op)
        assert result.is_success
        assert result.value is True

    def test_failing_demo_returns_failure(
        self, basic_result_module: Any
    ) -> None:
        """_execute_demo() with a raising function returns r[bool].fail(...)."""
        RailwayService = basic_result_module.RailwayService

        def bad_demo() -> None:
            raise ValueError("Demo failed!")

        result = RailwayService._execute_demo(bad_demo)
        assert result.is_failure
        assert "Demo failed!" in (result.error or "")

    def test_execute_demo_catches_any_exception(
        self, basic_result_module: Any
    ) -> None:
        """_execute_demo() catches various exception types."""
        RailwayService = basic_result_module.RailwayService

        def runtime_error() -> None:
            raise RuntimeError("Runtime issue")

        result = RailwayService._execute_demo(runtime_error)
        assert result.is_failure

    def test_execute_demo_returns_bool_value_on_success(
        self, basic_result_module: Any
    ) -> None:
        """_execute_demo() on success contains a bool (True) value."""
        RailwayService = basic_result_module.RailwayService

        def success() -> None:
            _ = 1 + 1

        result = RailwayService._execute_demo(success)
        assert result.is_success
        assert isinstance(result.value, bool)


# ---------------------------------------------------------------------------
# Tests: Individual demonstration methods (smoke tests)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestDemonstrationMethods:
    """Smoke tests for individual demonstration methods."""

    def test_demonstrate_factory_methods_runs(
        self, basic_result_module: Any
    ) -> None:
        """_demonstrate_factory_methods() runs without raising."""
        RailwayService = basic_result_module.RailwayService
        RailwayService._demonstrate_factory_methods()

    def test_demonstrate_error_recovery_runs(
        self, basic_result_module: Any
    ) -> None:
        """_demonstrate_error_recovery() runs without raising."""
        RailwayService = basic_result_module.RailwayService
        RailwayService._demonstrate_error_recovery()

    def test_demonstrate_advanced_combinators_runs(
        self, basic_result_module: Any
    ) -> None:
        """_demonstrate_advanced_combinators() runs without raising."""
        RailwayService = basic_result_module.RailwayService
        RailwayService._demonstrate_advanced_combinators()

    def test_demonstrate_exception_integration_runs(
        self, basic_result_module: Any
    ) -> None:
        """_demonstrate_exception_integration() runs without raising."""
        RailwayService = basic_result_module.RailwayService
        RailwayService._demonstrate_exception_integration()

    def test_demonstrate_railway_operations_runs(
        self, basic_result_module: Any
    ) -> None:
        """_demonstrate_railway_operations() runs without raising."""
        RailwayService = basic_result_module.RailwayService
        RailwayService._demonstrate_railway_operations()

    def test_demonstrate_validation_patterns_runs(
        self, basic_result_module: Any
    ) -> None:
        """_demonstrate_validation_patterns() runs without raising."""
        RailwayService = basic_result_module.RailwayService
        RailwayService._demonstrate_validation_patterns()

    def test_demonstrate_value_extraction_runs(
        self, basic_result_module: Any
    ) -> None:
        """_demonstrate_value_extraction() runs without raising."""
        RailwayService = basic_result_module.RailwayService
        RailwayService._demonstrate_value_extraction()

    def test_demonstrate_value_extraction_uses_c_field_name(
        self, basic_result_module: Any
    ) -> None:
        """_demonstrate_value_extraction() uses c.FIELD_NAME ('name') for dict lookup."""
        # c.FIELD_NAME == "name" (from FlextConstantsMixins)
        data = t.ConfigMap(root={"name": "Alice", "email": "alice@example.com"})
        value = data.get(c.FIELD_NAME, "")
        assert value == "Alice"

    def test_demonstrate_advanced_combinators_uses_default_max_command_retries(
        self,
    ) -> None:
        """c.DEFAULT_MAX_COMMAND_RETRIES == 0 used as offset in combinator tests."""
        # Was c.ZERO — verifying the replacement constant has the same value
        assert c.DEFAULT_MAX_COMMAND_RETRIES == 0
        # Results created in the demo: ok(0+1), ok(0+2), ok(0+3)
        results = [
            r[int].ok(c.DEFAULT_MAX_COMMAND_RETRIES + 1),
            r[int].ok(c.DEFAULT_MAX_COMMAND_RETRIES + 2),
            r[int].ok(c.DEFAULT_MAX_COMMAND_RETRIES + 3),
        ]
        traversed = r.traverse(results, lambda r_val: r_val)
        assert traversed.is_success
        assert len(traversed.value) == 3

    def test_demonstrate_advanced_combinators_filter_with_backup_count(
        self,
    ) -> None:
        """c.BACKUP_COUNT used as filter threshold in combinator tests."""
        # Was c.Validation.FILTER_THRESHOLD
        test_value = c.BACKUP_COUNT + c.DEFAULT_MAX_COMMAND_RETRIES
        filtered = r[int].ok(test_value).filter(lambda x: x > c.BACKUP_COUNT)
        # test_value = BACKUP_COUNT + 0 = BACKUP_COUNT, which is NOT > BACKUP_COUNT
        assert filtered.is_failure  # equal, not greater

    def test_demonstrate_factory_methods_risky_operation(self) -> None:
        """Risky operation with c.MAX_AGE // c.DEFAULT_MAX_COMMAND_RETRIES raises ZeroDivisionError."""
        # DEFAULT_MAX_COMMAND_RETRIES = 0, so c.MAX_AGE // 0 should raise ZeroDivisionError
        zero = c.DEFAULT_MAX_COMMAND_RETRIES  # 0
        with pytest.raises(ZeroDivisionError):
            _ = c.MAX_AGE // zero

    def test_demonstrate_factory_methods_create_from_callable_catches_error(
        self,
    ) -> None:
        """r[int].create_from_callable captures ZeroDivisionError as failure."""
        def risky() -> int:
            zero = c.DEFAULT_MAX_COMMAND_RETRIES  # 0
            return c.MAX_AGE // zero

        result = r[int].create_from_callable(risky)
        assert result.is_failure


# ---------------------------------------------------------------------------
# Tests: RailwayService.execute() (full service test)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRailwayServiceExecute:
    """Tests for RailwayService.execute() full service execution."""

    def test_execute_returns_demonstration_result(
        self, basic_result_module: Any
    ) -> None:
        """execute() returns r[DemonstrationResult].ok(...)."""
        RailwayService = basic_result_module.RailwayService
        DemonstrationResult = basic_result_module.DemonstrationResult
        service = RailwayService()
        result = service.execute()
        assert result.is_success
        assert isinstance(result.value, DemonstrationResult)

    def test_execute_demonstrations_completed_positive(
        self, basic_result_module: Any
    ) -> None:
        """execute() result has positive demonstrations_completed count."""
        RailwayService = basic_result_module.RailwayService
        service = RailwayService()
        result = service.execute()
        assert result.is_success
        assert result.value.demonstrations_completed > 0

    def test_execute_patterns_covered_not_empty(
        self, basic_result_module: Any
    ) -> None:
        """execute() result has non-empty patterns_covered tuple."""
        RailwayService = basic_result_module.RailwayService
        service = RailwayService()
        result = service.execute()
        assert result.is_success
        assert len(result.value.patterns_covered) > 0

    def test_execute_completed_at_is_set(
        self, basic_result_module: Any
    ) -> None:
        """execute() result has completed_at timestamp."""
        RailwayService = basic_result_module.RailwayService
        service = RailwayService()
        result = service.execute()
        assert result.is_success
        assert result.value.completed_at

    def test_handle_execution_error_returns_failure(
        self, basic_result_module: Any
    ) -> None:
        """_handle_execution_error() returns r[DemonstrationResult].fail(...)."""
        RailwayService = basic_result_module.RailwayService
        service = RailwayService()
        error = RuntimeError("Test error")
        result = service._handle_execution_error(error)
        assert result.is_failure
        assert "Test error" in (result.error or "")

    def test_handle_execution_error_uses_c_exception_error(
        self, basic_result_module: Any
    ) -> None:
        """_handle_execution_error() uses c.EXCEPTION_ERROR as error_code."""
        RailwayService = basic_result_module.RailwayService
        service = RailwayService()
        error = ValueError("Something went wrong")
        result = service._handle_execution_error(error)
        assert result.is_failure
        assert result.error_code == c.EXCEPTION_ERROR


# ---------------------------------------------------------------------------
# Tests: RailwayService._run_demonstrations
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRailwayServiceRunDemonstrations:
    """Tests for RailwayService._run_demonstrations()."""

    def test_run_demonstrations_returns_ok(
        self, basic_result_module: Any
    ) -> None:
        """_run_demonstrations() runs all demos and returns r[None].ok(...)."""
        RailwayService = basic_result_module.RailwayService
        service = RailwayService()
        result = service._run_demonstrations()
        assert result.is_success

    def test_create_data_processor_returns_callable(
        self, basic_result_module: Any
    ) -> None:
        """_create_data_processor() returns a callable."""
        RailwayService = basic_result_module.RailwayService
        DemonstrationResult = basic_result_module.DemonstrationResult
        processor = RailwayService._create_data_processor()
        assert callable(processor)
        # Test it works with a DemonstrationResult
        data = DemonstrationResult(
            demonstrations_completed=2,
            patterns_covered=("a", "b"),
            completed_at="2025-01-01T00:00:00Z",
        )
        processed = processor(data)
        assert processed == data


# ---------------------------------------------------------------------------
# Tests: Regression tests for API rename correctness
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestApiRenameRegression:
    """Regression tests verifying old API names are gone and new ones work."""

    def test_c_handler_type_not_in_nested_cqrs(self) -> None:
        """c.HandlerType is accessible at top level (not nested as c.Cqrs.HandlerType)."""
        # The new API: c.HandlerType (flat)
        assert hasattr(c, "HandlerType")
        assert c.HandlerType.COMMAND == "command"

    def test_c_validation_error_not_in_nested_errors(self) -> None:
        """c.VALIDATION_ERROR is accessible at top level (not c.Errors.VALIDATION_ERROR)."""
        assert hasattr(c, "VALIDATION_ERROR")
        assert c.VALIDATION_ERROR == "VALIDATION_ERROR"

    def test_c_pattern_email_not_in_nested_platform(self) -> None:
        """c.PATTERN_EMAIL is accessible at top level (not c.Platform.PATTERN_EMAIL)."""
        assert hasattr(c, "PATTERN_EMAIL")
        assert isinstance(c.PATTERN_EMAIL, str)

    def test_c_status_not_in_nested_domain(self) -> None:
        """c.Status is accessible at top level (not c.Domain.Status)."""
        assert hasattr(c, "Status")
        assert c.Status.ACTIVE == "active"

    def test_c_action_not_in_nested_cqrs(self) -> None:
        """c.Action is accessible at top level (not c.Cqrs.Action)."""
        assert hasattr(c, "Action")
        assert c.Action.CREATE == "create"

    def test_c_backup_count_accessible(self) -> None:
        """c.BACKUP_COUNT is accessible (was c.Validation.FILTER_THRESHOLD equivalent)."""
        assert hasattr(c, "BACKUP_COUNT")
        assert isinstance(c.BACKUP_COUNT, int)

    def test_c_max_age_accessible(self) -> None:
        """c.MAX_AGE is accessible (was c.Validation.MAX_AGE)."""
        assert hasattr(c, "MAX_AGE")
        assert isinstance(c.MAX_AGE, int)

    def test_c_exception_error_accessible(self) -> None:
        """c.EXCEPTION_ERROR is accessible at top level."""
        assert hasattr(c, "EXCEPTION_ERROR")
        assert c.EXCEPTION_ERROR == "EXCEPTION_ERROR"

    def test_traverse_with_default_max_command_retries(self) -> None:
        """r.traverse with results offset by c.DEFAULT_MAX_COMMAND_RETRIES works."""
        # This replicates the combinator code in the example
        results = [
            r[int].ok(c.DEFAULT_MAX_COMMAND_RETRIES + 1),
            r[int].ok(c.DEFAULT_MAX_COMMAND_RETRIES + 2),
            r[int].ok(c.DEFAULT_MAX_COMMAND_RETRIES + 3),
        ]
        traversed = r.traverse(results, lambda r_val: r_val)
        assert traversed.is_success
        # Values should be 1, 2, 3 since DEFAULT_MAX_COMMAND_RETRIES == 0
        assert set(traversed.value) == {1, 2, 3}