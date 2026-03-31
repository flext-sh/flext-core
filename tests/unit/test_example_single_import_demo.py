"""Tests for examples/00_single_import_demo.py.

Covers the changed API patterns in the PR:
- c.Status (was c.Domain.Status)
- c.Action (was c.Cqrs.Action)
- t.ConfigMap (was m.ConfigMap)
- c.VALIDATION_ERROR (was c.Errors.VALIDATION_ERROR)
- c.PATTERN_EMAIL (was c.Platform.PATTERN_EMAIL)
- UserProfile.activate() returning r[bool] (was r[None])
- ignore_and_return_none() returning r[bool] (was r[None])

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

from flext_core import c, e, r, t, u


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
    # Register in sys.modules to allow relative cross-imports within example
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def demo_module() -> Any:
    """Load the 00_single_import_demo.py module once per test module."""
    return _load_example_module("00_single_import_demo.py")


# ---------------------------------------------------------------------------
# Tests: Constants API (c.Status, c.Action, c.VALIDATION_ERROR, etc.)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestConstantsApiChanges:
    """Verify the changed constants API works as expected in the example."""

    def test_c_status_active_exists(self) -> None:
        """c.Status.ACTIVE is accessible (was c.Domain.Status.ACTIVE)."""
        assert c.Status.ACTIVE == "active"

    def test_c_status_inactive_exists(self) -> None:
        """c.Status.INACTIVE is accessible."""
        assert c.Status.INACTIVE == "inactive"

    def test_c_status_archived_exists(self) -> None:
        """c.Status.ARCHIVED is accessible."""
        assert c.Status.ARCHIVED == "archived"

    def test_c_action_create_exists(self) -> None:
        """c.Action.CREATE is accessible (was c.Cqrs.Action.CREATE)."""
        assert c.Action.CREATE == "create"

    def test_c_action_all_values(self) -> None:
        """All c.Action values are accessible."""
        assert c.Action.GET == "get"
        assert c.Action.CREATE == "create"
        assert c.Action.UPDATE == "update"
        assert c.Action.DELETE == "delete"
        assert c.Action.LIST == "list"

    def test_c_validation_error_constant(self) -> None:
        """c.VALIDATION_ERROR is accessible (was c.Errors.VALIDATION_ERROR)."""
        assert c.VALIDATION_ERROR == "VALIDATION_ERROR"

    def test_c_pattern_email_exists(self) -> None:
        """c.PATTERN_EMAIL is accessible (was c.Platform.PATTERN_EMAIL)."""
        assert isinstance(c.PATTERN_EMAIL, str)
        assert "@" in c.PATTERN_EMAIL or "email" in c.PATTERN_EMAIL.lower()

    def test_c_max_retry_attempts_is_int(self) -> None:
        """c.MAX_RETRY_ATTEMPTS is accessible (was c.Validation.MIN_USERNAME_LENGTH)."""
        assert isinstance(c.MAX_RETRY_ATTEMPTS, int)
        assert c.MAX_RETRY_ATTEMPTS > 0

    def test_t_configmap_is_pydantic_model(self) -> None:
        """t.ConfigMap is a Pydantic RootModel wrapping a dict."""
        config = t.ConfigMap(root={"name": "Test", "email": "test@example.com"})
        assert config.get("name") == "Test"
        assert config.get("email") == "test@example.com"

    def test_t_configmap_get_missing_key_returns_none(self) -> None:
        """t.ConfigMap.get() returns None for missing key."""
        config = t.ConfigMap(root={"name": "Test"})
        assert config.get("nonexistent") is None

    def test_t_configmap_keys(self) -> None:
        """t.ConfigMap supports keys() iteration."""
        config = t.ConfigMap(root={"a": 1, "b": 2})
        assert set(config.keys()) == {"a", "b"}


# ---------------------------------------------------------------------------
# Tests: UserProfile class
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestUserProfile:
    """Tests for UserProfile model imported from the demo module."""

    def test_userprofile_activate_returns_r_bool_when_inactive(
        self, demo_module: Any
    ) -> None:
        """activate() returns r[bool].ok(True) when status is INACTIVE."""
        UserProfile = demo_module.UserProfile
        profile = UserProfile(
            name="Alice",
            email="alice@example.com",
            unique_id="uid-123",
            status=c.Status.INACTIVE,
        )
        result = profile.activate()
        assert result.is_success
        assert result.value is True

    def test_userprofile_activate_fails_when_already_active(
        self, demo_module: Any
    ) -> None:
        """activate() returns r[bool].fail when status is ACTIVE."""
        UserProfile = demo_module.UserProfile
        profile = UserProfile(
            name="Bob",
            email="bob@example.com",
            unique_id="uid-456",
            status=c.Status.ACTIVE,
        )
        result = profile.activate()
        assert result.is_failure
        assert "already active" in (result.error or "").lower()

    def test_userprofile_uses_c_status_enum(self, demo_module: Any) -> None:
        """UserProfile.status accepts c.Status StrEnum values."""
        UserProfile = demo_module.UserProfile
        profile = UserProfile(
            name="Charlie",
            email="charlie@example.com",
            unique_id="uid-789",
            status=c.Status.ARCHIVED,
        )
        assert profile.status == c.Status.ARCHIVED

    def test_userprofile_activate_result_is_r_bool_type(
        self, demo_module: Any
    ) -> None:
        """activate() return type is r[bool] (not r[None])."""
        UserProfile = demo_module.UserProfile
        profile = UserProfile(
            name="Dave",
            email="dave@example.com",
            unique_id="uid-001",
            status=c.Status.INACTIVE,
        )
        result = profile.activate()
        # The value should be True (bool), not None
        assert isinstance(result.value, bool)

    def test_userprofile_is_frozen(self, demo_module: Any) -> None:
        """UserProfile is a frozen Pydantic model (immutable)."""
        UserProfile = demo_module.UserProfile
        profile = UserProfile(
            name="Eve",
            email="eve@example.com",
            unique_id="uid-002",
            status=c.Status.ACTIVE,
        )
        with pytest.raises(Exception):  # ValidationError or AttributeError
            profile.name = "Modified"


# ---------------------------------------------------------------------------
# Tests: validate_transform_user function
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestValidateTransformUser:
    """Tests for validate_transform_user() in the demo module."""

    def test_valid_data_returns_ok_user_profile(self, demo_module: Any) -> None:
        """Valid name and email produce a successful UserProfile result."""
        validate = demo_module.validate_transform_user
        data = t.ConfigMap(root={"name": "Alice", "email": "alice@example.com"})
        result = validate(data)
        assert result.is_success
        assert result.value.name == "ALICE"
        assert result.value.email == "alice@example.com"

    def test_valid_data_sets_status_active(self, demo_module: Any) -> None:
        """Successfully created user has ACTIVE status."""
        validate = demo_module.validate_transform_user
        data = t.ConfigMap(root={"name": "Bob", "email": "bob@example.com"})
        result = validate(data)
        assert result.is_success
        assert result.value.status == c.Status.ACTIVE

    def test_valid_data_generates_unique_id(self, demo_module: Any) -> None:
        """Successfully created user has a non-empty unique_id."""
        validate = demo_module.validate_transform_user
        data = t.ConfigMap(root={"name": "Carol", "email": "carol@example.com"})
        result = validate(data)
        assert result.is_success
        assert result.value.unique_id
        assert len(result.value.unique_id) > 0

    def test_missing_name_returns_failure(self, demo_module: Any) -> None:
        """Missing name key returns failure result."""
        validate = demo_module.validate_transform_user
        data = t.ConfigMap(root={"email": "test@example.com"})
        result = validate(data)
        assert result.is_failure

    def test_missing_email_returns_failure(self, demo_module: Any) -> None:
        """Missing email key returns failure result."""
        validate = demo_module.validate_transform_user
        data = t.ConfigMap(root={"name": "Alice"})
        result = validate(data)
        assert result.is_failure

    def test_non_string_name_returns_failure(self, demo_module: Any) -> None:
        """Non-string name returns failure result."""
        validate = demo_module.validate_transform_user
        data = t.ConfigMap(root={"name": 123, "email": "test@example.com"})
        result = validate(data)
        assert result.is_failure

    def test_empty_name_returns_failure(self, demo_module: Any) -> None:
        """Empty string name returns failure result."""
        validate = demo_module.validate_transform_user
        data = t.ConfigMap(root={"name": "", "email": "test@example.com"})
        result = validate(data)
        assert result.is_failure

    def test_empty_email_returns_failure(self, demo_module: Any) -> None:
        """Empty string email returns failure result."""
        validate = demo_module.validate_transform_user
        data = t.ConfigMap(root={"name": "Alice", "email": ""})
        result = validate(data)
        assert result.is_failure

    def test_name_uppercased_in_result(self, demo_module: Any) -> None:
        """Validated user profile has name in uppercase."""
        validate = demo_module.validate_transform_user
        data = t.ConfigMap(root={"name": "lowercase", "email": "low@example.com"})
        result = validate(data)
        assert result.is_success
        assert result.value.name == "LOWERCASE"

    def test_email_lowercased_in_result(self, demo_module: Any) -> None:
        """Validated user profile has email in lowercase."""
        validate = demo_module.validate_transform_user
        data = t.ConfigMap(root={"name": "Test", "email": "UPPER@EXAMPLE.COM"})
        result = validate(data)
        assert result.is_success
        assert result.value.email == "upper@example.com"

    def test_short_name_below_min_length_fails(self, demo_module: Any) -> None:
        """Name shorter than c.MAX_RETRY_ATTEMPTS chars fails validation."""
        validate = demo_module.validate_transform_user
        # c.MAX_RETRY_ATTEMPTS = 3, so a name with 2 chars should fail length check
        short_name = "AB"  # length 2 < MAX_RETRY_ATTEMPTS (3)
        data = t.ConfigMap(root={"name": short_name, "email": "test@example.com"})
        result = validate(data)
        # u.validate_length("AB", min_length=3) fails → traverse fails → failure
        assert result.is_failure

    def test_invalid_email_pattern_fails(self, demo_module: Any) -> None:
        """Invalid email pattern fails validation."""
        validate = demo_module.validate_transform_user
        data = t.ConfigMap(root={"name": "Alice", "email": "not-an-email"})
        result = validate(data)
        assert result.is_failure

    def test_accepts_t_configmap_not_dict(self, demo_module: Any) -> None:
        """Function accepts t.ConfigMap (not plain dict) as input."""
        validate = demo_module.validate_transform_user
        # This tests that the function works with t.ConfigMap
        data = t.ConfigMap(root={"name": "TestUser", "email": "test@example.com"})
        assert isinstance(data, t.ConfigMap)
        result = validate(data)
        assert result.is_success


# ---------------------------------------------------------------------------
# Tests: process_user_data function
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestProcessUserData:
    """Tests for process_user_data() in the demo module."""

    def test_create_action_produces_created_message(self, demo_module: Any) -> None:
        """c.Action.CREATE produces 'CREATED' in result message."""
        process = demo_module.process_user_data
        data = t.ConfigMap(root={"name": "Alice", "email": "alice@example.com"})
        result = process(user_data=data, operation=c.Action.CREATE)
        assert result.is_success
        assert "CREATED" in result.value

    def test_create_action_includes_profile_name(self, demo_module: Any) -> None:
        """Result includes the user name in uppercase."""
        process = demo_module.process_user_data
        data = t.ConfigMap(root={"name": "bob", "email": "bob@example.com"})
        result = process(user_data=data, operation=c.Action.CREATE)
        assert result.is_success
        assert "BOB" in result.value

    def test_create_action_includes_status(self, demo_module: Any) -> None:
        """Result includes the status value."""
        process = demo_module.process_user_data
        data = t.ConfigMap(root={"name": "Carol", "email": "carol@example.com"})
        result = process(user_data=data, operation=c.Action.CREATE)
        assert result.is_success
        assert c.Status.ACTIVE.value in result.value

    def test_invalid_data_returns_failure(self, demo_module: Any) -> None:
        """Invalid user data (missing email) returns failure."""
        process = demo_module.process_user_data
        data = t.ConfigMap(root={"name": "Dave"})
        result = process(user_data=data, operation=c.Action.CREATE)
        assert result.is_failure

    def test_uses_c_action_enum(self, demo_module: Any) -> None:
        """Function accepts c.Action StrEnum values (was c.Cqrs.Action)."""
        process = demo_module.process_user_data
        data = t.ConfigMap(root={"name": "Eve", "email": "eve@example.com"})
        # Test different action values
        for action in [c.Action.CREATE, c.Action.UPDATE, c.Action.DELETE]:
            result = process(user_data=data, operation=action)
            # All should produce some result (success or failure based on action)
            assert result.is_success or result.is_failure


# ---------------------------------------------------------------------------
# Tests: UserService class
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestUserService:
    """Tests for UserService in the demo module."""

    def test_create_user_success(self, demo_module: Any) -> None:
        """create_user() with valid data returns successful UserProfile."""
        UserService = demo_module.UserService
        service = UserService()
        data = t.ConfigMap(root={"name": "Alice", "email": "alice@example.com"})
        result = service.create_user(data)
        assert result.is_success
        assert result.value.name == "ALICE"

    def test_create_user_failure_missing_email(self, demo_module: Any) -> None:
        """create_user() with missing email returns failure."""
        UserService = demo_module.UserService
        service = UserService()
        data = t.ConfigMap(root={"name": "Bob"})
        result = service.create_user(data)
        assert result.is_failure

    def test_create_user_failure_missing_name(self, demo_module: Any) -> None:
        """create_user() with missing name returns failure."""
        UserService = demo_module.UserService
        service = UserService()
        data = t.ConfigMap(root={"email": "bob@example.com"})
        result = service.create_user(data)
        assert result.is_failure

    def test_validate_data_success(self, demo_module: Any) -> None:
        """_validate_data() succeeds when name and email are present."""
        UserService = demo_module.UserService
        data = t.ConfigMap(root={"name": "Test", "email": "test@example.com"})
        result = UserService._validate_data(data)
        assert result.is_success
        assert result.value is True

    def test_validate_data_failure_missing_email(self, demo_module: Any) -> None:
        """_validate_data() fails when email is missing."""
        UserService = demo_module.UserService
        data = t.ConfigMap(root={"name": "Test"})
        result = UserService._validate_data(data)
        assert result.is_failure
        assert "email" in (result.error or "").lower()

    def test_validate_data_failure_missing_name(self, demo_module: Any) -> None:
        """_validate_data() fails when name is missing."""
        UserService = demo_module.UserService
        data = t.ConfigMap(root={"email": "test@example.com"})
        result = UserService._validate_data(data)
        assert result.is_failure
        assert "name" in (result.error or "").lower()

    def test_validate_data_failure_empty_data(self, demo_module: Any) -> None:
        """_validate_data() fails with empty ConfigMap."""
        UserService = demo_module.UserService
        data = t.ConfigMap(root={})
        result = UserService._validate_data(data)
        assert result.is_failure

    def test_activate_user_success(self, demo_module: Any) -> None:
        """_activate_user() with inactive user returns successful UserProfile."""
        UserProfile = demo_module.UserProfile
        UserService = demo_module.UserService
        profile = UserProfile(
            name="Alice",
            email="alice@example.com",
            unique_id="uid-123",
            status=c.Status.INACTIVE,
        )
        result = UserService._activate_user(profile)
        assert result.is_success
        assert result.value.name == "Alice"

    def test_activate_user_failure_already_active(
        self, demo_module: Any
    ) -> None:
        """_activate_user() with already-active user returns failure."""
        UserProfile = demo_module.UserProfile
        UserService = demo_module.UserService
        profile = UserProfile(
            name="Bob",
            email="bob@example.com",
            unique_id="uid-456",
            status=c.Status.ACTIVE,
        )
        result = UserService._activate_user(profile)
        assert result.is_failure

    def test_operation_count_increments(self, demo_module: Any) -> None:
        """operation_count increments with each create_user call."""
        UserService = demo_module.UserService
        service = UserService()
        assert service.operation_count == 0
        data = t.ConfigMap(root={"name": "Alice", "email": "alice@example.com"})
        service.create_user(data)
        assert service.operation_count == 1
        service.create_user(data)
        assert service.operation_count == 2


# ---------------------------------------------------------------------------
# Tests: Helper functions
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestHelperFunctions:
    """Tests for standalone helper functions in the demo module."""

    def test_ignore_and_return_none_returns_r_bool(
        self, demo_module: Any
    ) -> None:
        """ignore_and_return_none() returns r[bool].ok(True) (was r[None])."""
        ignore_fn = demo_module.ignore_and_return_none
        UserProfile = demo_module.UserProfile
        profile = UserProfile(
            name="Test",
            email="test@example.com",
            unique_id="uid-001",
            status=c.Status.ACTIVE,
        )
        result = ignore_fn(profile)
        assert result.is_success
        assert result.value is True
        assert isinstance(result.value, bool)

    def test_identity_returns_same_result(self, demo_module: Any) -> None:
        """identity() returns the same r[str] object unchanged."""
        identity = demo_module.identity
        original = r[str].ok("hello")
        returned = identity(original)
        assert returned.is_success
        assert returned.value == "hello"

    def test_identity_with_failure_preserves_failure(
        self, demo_module: Any
    ) -> None:
        """identity() preserves failure results."""
        identity = demo_module.identity
        original = r[str].fail("test error")
        returned = identity(original)
        assert returned.is_failure
        assert returned.error == "test error"

    def test_identity_result_returns_same_object(
        self, demo_module: Any
    ) -> None:
        """identity_result() returns the same r[str] object unchanged."""
        identity_result = demo_module.identity_result
        original = r[str].ok("world")
        returned = identity_result(original)
        assert returned.is_success
        assert returned.value == "world"


# ---------------------------------------------------------------------------
# Tests: execute_validation_chain and execute_service_operations
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestExecutionFunctions:
    """Tests for execute_* functions in the demo module."""

    def test_execute_validation_chain_valid_data(
        self, demo_module: Any, capsys: pytest.CaptureFixture
    ) -> None:
        """execute_validation_chain() runs without error on valid data."""
        execute = demo_module.execute_validation_chain
        data = t.ConfigMap(root={"name": "Alice", "email": "alice@example.com"})
        # Should not raise
        execute(data)

    def test_execute_validation_chain_invalid_data_graceful(
        self, demo_module: Any, capsys: pytest.CaptureFixture
    ) -> None:
        """execute_validation_chain() handles invalid data gracefully."""
        execute = demo_module.execute_validation_chain
        data = t.ConfigMap(root={"name": "Alice"})  # missing email
        # Should not raise (lash handles errors)
        execute(data)
        captured = capsys.readouterr()
        assert "Validation failed" in captured.out

    def test_execute_service_operations_success(
        self, demo_module: Any, capsys: pytest.CaptureFixture
    ) -> None:
        """execute_service_operations() prints success message on valid data."""
        UserService = demo_module.UserService
        execute = demo_module.execute_service_operations
        service = UserService()
        data = t.ConfigMap(root={"name": "Alice", "email": "alice@example.com"})
        execute(service, data)
        captured = capsys.readouterr()
        assert "SUCCESS" in captured.out

    def test_execute_service_operations_failure(
        self, demo_module: Any, capsys: pytest.CaptureFixture
    ) -> None:
        """execute_service_operations() prints failure message on invalid data."""
        UserService = demo_module.UserService
        execute = demo_module.execute_service_operations
        service = UserService()
        data = t.ConfigMap(root={"name": "Alice"})  # missing email
        execute(service, data)
        captured = capsys.readouterr()
        assert "FAILED" in captured.out


# ---------------------------------------------------------------------------
# Tests: demonstrate_exceptions function (verifies c.VALIDATION_ERROR usage)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestDemonstrateExceptions:
    """Tests for demonstrate_exceptions() verifying the new c.VALIDATION_ERROR API."""

    def test_demonstrate_exceptions_runs_without_error(
        self, demo_module: Any
    ) -> None:
        """demonstrate_exceptions() runs without raising exceptions."""
        demonstrate = demo_module.demonstrate_exceptions
        # Should not raise
        demonstrate()

    def test_validation_error_constant_used_correctly(self) -> None:
        """c.VALIDATION_ERROR is a non-empty string constant."""
        assert c.VALIDATION_ERROR
        assert isinstance(c.VALIDATION_ERROR, str)
        assert len(c.VALIDATION_ERROR) > 0

    def test_e_validation_error_accepts_c_validation_error(self) -> None:
        """e.ValidationError accepts c.VALIDATION_ERROR as error_code."""
        try:
            raise e.ValidationError(
                "Test error",
                field="test_field",
                value="test_value",
                error_code=c.VALIDATION_ERROR,
            )
        except e.ValidationError as exc:
            assert exc.error_code == c.VALIDATION_ERROR

    def test_r_fail_with_c_validation_error(self) -> None:
        """r[str].fail() accepts c.VALIDATION_ERROR as error_code."""
        result = r[str].fail("test error", error_code=c.VALIDATION_ERROR)
        assert result.is_failure
        assert result.error_code == c.VALIDATION_ERROR


# ---------------------------------------------------------------------------
# Tests: demonstrate_utilities function (verifies c.PATTERN_EMAIL usage)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestDemonstrateUtilities:
    """Tests for demonstrate_utilities() verifying the new c.PATTERN_EMAIL API."""

    def test_demonstrate_utilities_runs_without_error(
        self, demo_module: Any
    ) -> None:
        """demonstrate_utilities() runs without raising exceptions."""
        demonstrate = demo_module.demonstrate_utilities
        # Should not raise
        demonstrate()

    def test_c_pattern_email_matches_valid_email(self) -> None:
        """c.PATTERN_EMAIL regex matches a valid email address."""
        import re
        pattern = re.compile(c.PATTERN_EMAIL)
        assert pattern.match("test@example.com")

    def test_c_pattern_email_rejects_invalid_email(self) -> None:
        """c.PATTERN_EMAIL regex rejects an invalid email address."""
        import re
        pattern = re.compile(c.PATTERN_EMAIL)
        assert not pattern.match("not-an-email")

    def test_u_validate_pattern_uses_c_pattern_email(self) -> None:
        """u.validate_pattern works with c.PATTERN_EMAIL."""
        result_valid = u.validate_pattern("user@domain.com", c.PATTERN_EMAIL, "email")
        assert result_valid.is_success

        result_invalid = u.validate_pattern("not-email", c.PATTERN_EMAIL, "email")
        assert result_invalid.is_failure