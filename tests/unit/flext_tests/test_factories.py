"""Unit tests for flext_tests.factories module.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import pytest
from pydantic import BaseModel as _BaseModel

from flext_core import FlextResult, r
from flext_tests.factories import tt
from flext_tests.models import m
from flext_tests.typings import t as t_test
from tests.test_utils import assertion_helpers

# Access models from centralized m.Tests.Factory namespace
User = m.Tests.Factory.User
Config = m.Tests.Factory.Config
Service = m.Tests.Factory.Service
Entity = m.Tests.Factory.Entity
ValueObject = m.Tests.Factory.ValueObject


def _extract_model(
    result: (
        _BaseModel
        | list[_BaseModel]
        | Mapping[str, _BaseModel]
        | FlextResult[_BaseModel]
        | FlextResult[list[_BaseModel]]
        | FlextResult[Mapping[str, _BaseModel]]
    ),
) -> _BaseModel:
    """Extract BaseModel from union type returned by tt.model().

    Args:
        result: Union type from tt.model()

    Returns:
        BaseModel instance

    Raises:
        AssertionError: If result is not a single BaseModel

    """
    if isinstance(result, FlextResult):
        unwrapped = result.value
        if isinstance(unwrapped, _BaseModel):
            return unwrapped
        if (
            isinstance(unwrapped, list)
            and unwrapped
            and isinstance(unwrapped[0], _BaseModel)
        ):
            return unwrapped[0]
        if isinstance(unwrapped, Mapping):
            first_value = next(iter(unwrapped.values()), None)
            if isinstance(first_value, _BaseModel):
                return first_value
        msg = f"Expected BaseModel, got {type(unwrapped)}"
        raise AssertionError(msg)
    if isinstance(result, _BaseModel):
        return result
    msg = f"Expected BaseModel, got {type(result)}"
    raise AssertionError(msg)


def _as_single_payload_result(
    value: r[t_test.Tests.PayloadValue] | list[r[t_test.Tests.PayloadValue]],
) -> r[t_test.Tests.PayloadValue]:
    return value if isinstance(value, r) else value[0]


def _as_payload_list(
    value: list[t_test.Tests.PayloadValue] | r[list[t_test.Tests.PayloadValue]],
) -> list[t_test.Tests.PayloadValue]:
    return value if isinstance(value, list) else value.value


def _as_payload_mapping(
    value: Mapping[str, t_test.Tests.PayloadValue]
    | r[Mapping[str, t_test.Tests.PayloadValue]],
) -> Mapping[str, t_test.Tests.PayloadValue]:
    return value if isinstance(value, Mapping) else value.value


class TestUser:
    """Test suite for User model."""

    def test_user_creation_default(self) -> None:
        """Test User model creation with defaults."""
        user = User(
            id="test-123",
            name="Test User",
            email="test@example.com",
        )

        assert user.id == "test-123"
        assert user.name == "Test User"
        assert user.email == "test@example.com"
        assert user.active is True

    def test_user_creation_with_active(self) -> None:
        """Test User model creation with active flag."""
        user = User(
            id="test-123",
            name="Test User",
            email="test@example.com",
            active=False,
        )

        assert user.active is False


class TestConfig:
    """Test suite for Config model."""

    def test_config_creation_default(self) -> None:
        """Test Config model creation with defaults."""
        config = Config()

        assert config.service_type == "api"
        assert config.environment == "test"
        assert config.debug is True
        assert config.log_level == "DEBUG"
        assert config.timeout == 30
        assert config.max_retries == 3

    def test_config_creation_custom(self) -> None:
        """Test Config model creation with custom values."""
        config = Config(
            service_type="database",
            environment="production",
            debug=False,
            timeout=60,
        )

        assert config.service_type == "database"
        assert config.environment == "production"
        assert config.debug is False
        assert config.timeout == 60


class TestService:
    """Test suite for Service model."""

    def test_service_creation_minimal(self) -> None:
        """Test Service model creation with minimal fields."""
        service = Service(id="test-123")

        assert service.id == "test-123"
        assert service.type == "api"
        assert service.name == ""
        assert service.status == "active"

    def test_service_creation_complete(self) -> None:
        """Test Service model creation with all fields."""
        service = Service(
            id="test-123",
            type="database",
            name="Database Service",
            status="inactive",
        )

        assert service.id == "test-123"
        assert service.type == "database"
        assert service.name == "Database Service"
        assert service.status == "inactive"


class TestFlextTestsFactoriesModernAPI:
    """Test suite for FlextTestsFactories using modern API (tt.model, tt.op, tt.svc)."""

    def test_model_user_default(self) -> None:
        """Test tt.model('user') with default parameters."""
        user_result = tt.model("user")
        user = _extract_model(user_result)
        assert isinstance(user, User)
        assert user.id is not None
        assert user.name == "Test User"
        assert "@example.com" in user.email
        assert user.active is True

    def test_model_user_custom(self) -> None:
        """Test tt.model('user') with custom parameters."""
        user_result = tt.model(
            "user",
            model_id="custom-123",
            name="Custom User",
            email="custom@test.com",
        )
        user = _extract_model(user_result)
        assert isinstance(user, User)
        assert user.id == "custom-123"
        assert user.name == "Custom User"
        assert user.email == "custom@test.com"

    def test_model_user_with_overrides(self) -> None:
        """Test tt.model('user') with overrides."""
        user_result = tt.model(
            "user",
            name="Base User",
            active=False,
        )
        user = _extract_model(user_result)
        assert isinstance(user, User)
        assert user.name == "Base User"
        assert user.active is False

    def test_model_config_default(self) -> None:
        """Test tt.model('config') with default parameters."""
        config_result = tt.model("config")
        config = _extract_model(config_result)
        assert isinstance(config, Config)
        assert config.service_type == "api"
        assert config.environment == "test"
        assert config.debug is True

    def test_model_config_custom(self) -> None:
        """Test tt.model('config') with custom parameters."""
        config_result = tt.model(
            "config",
            service_type="database",
            environment="production",
            debug=False,
            timeout=60,
        )
        config = _extract_model(config_result)
        assert isinstance(config, Config)
        assert config.service_type == "database"
        assert config.environment == "production"
        assert config.debug is False
        assert config.timeout == 60

    def test_model_config_with_overrides(self) -> None:
        """Test tt.model('config') with overrides."""
        config_result = tt.model(
            "config",
            log_level="INFO",
            max_retries=5,
        )
        config = _extract_model(config_result)
        assert isinstance(config, Config)
        assert config.log_level == "INFO"
        assert config.max_retries == 5

    def test_model_service_default(self) -> None:
        """Test tt.model('service') with default parameters."""
        service_result = tt.model("service")
        service = _extract_model(service_result)
        assert isinstance(service, Service)
        assert service.id is not None
        assert service.type == "api"
        assert "Test api Service" in service.name
        assert service.status == "active"

    def test_model_service_custom(self) -> None:
        """Test tt.model('service') with custom parameters."""
        service_result = tt.model(
            "service",
            service_type="database",
            model_id="custom-123",
            name="Custom Service",
        )
        service = _extract_model(service_result)
        assert isinstance(service, Service)
        assert service.id == "custom-123"
        assert service.type == "database"
        assert service.name == "Custom Service"

    def test_model_service_with_overrides(self) -> None:
        """Test tt.model('service') with overrides."""
        service_result = tt.model(
            "service",
            status="inactive",
        )
        service = _extract_model(service_result)
        assert isinstance(service, Service)
        assert service.status == "inactive"

    def test_batch_users_default(self) -> None:
        """Test tt.batch('user') with default count."""
        users_result = tt.batch("user")
        if isinstance(users_result, list):
            users = users_result
        elif isinstance(users_result, FlextResult):
            users = users_result.value
            assert isinstance(users, list)
        else:
            msg = f"Expected list, got {type(users_result)}"
            raise AssertionError(msg)
        users_typed: list[User] = [u for u in users if isinstance(u, User)]
        assert len(users_typed) == 5
        assert all(isinstance(user, User) for user in users_typed)
        assert users_typed[0].name == "User 0"
        assert users_typed[1].name == "User 1"

    def test_batch_users_custom_count(self) -> None:
        """Test tt.batch('user') with custom count."""
        users_result = tt.batch("user", count=3)
        assert isinstance(users_result, list)
        users: list[User] = [u for u in users_result if isinstance(u, User)]
        assert len(users) == 3
        assert all(isinstance(user, User) for user in users)

    def test_op_simple(self) -> None:
        """Test tt.op('simple') operation."""
        operation = tt.op("simple")

        assert callable(operation)
        result = operation()
        assert result == "success"

    def test_op_add(self) -> None:
        """Test tt.op('add') operation."""
        operation = tt.op("add")

        assert callable(operation)
        result = operation(2, 3)
        assert result == 5

    def test_op_format(self) -> None:
        """Test tt.op('format') operation."""
        operation = tt.op("format")

        assert callable(operation)
        result = operation("name", value=20)
        assert result == "name: 20"

    def test_op_error(self) -> None:
        """Test tt.op('error') operation."""
        operation = tt.op(
            "error",
            error_message="Custom error",
        )

        assert callable(operation)
        with pytest.raises(ValueError, match="Custom error"):
            operation()

    def test_op_type_error(self) -> None:
        """Test tt.op('type_error') operation."""
        operation = tt.op(
            "type_error",
            error_message="Type mismatch",
        )

        assert callable(operation)
        with pytest.raises(TypeError, match="Type mismatch"):
            operation()

    def test_svc_default(self) -> None:
        """Test tt.svc() with default type."""
        service_class = tt.svc()
        service = service_class()

        assert service.name is None
        assert service.amount is None
        assert service.enabled is None

        result = service.execute()
        assertion_helpers.assert_flext_result_success(result)
        assert result.value == {"service_type": "test"}

    def test_svc_user(self) -> None:
        """Test tt.svc('user') with 'user' type."""
        service_class = tt.svc("user")
        service = service_class()

        result = service.execute()
        assertion_helpers.assert_flext_result_success(result)
        assert "user_id" in result.value
        assert result.value["user_id"] == "test_123"

    def test_svc_user_with_default(self) -> None:
        """Test tt.svc('user') with 'user' type and default flag."""
        # default should be passed as override via svc() kwargs
        service_class = tt.svc("user", default=True)
        service = service_class()

        result = service.execute()
        assertion_helpers.assert_flext_result_success(result)
        assert result.value["user_id"] == "default_123"

    def test_svc_complex_valid(self) -> None:
        """Test tt.svc('complex') with valid data."""
        service_class = tt.svc("complex")
        service = service_class(name="Test", amount=100, enabled=True)

        result = service.execute()
        assertion_helpers.assert_flext_result_success(result)
        assert result.value == {"result": "success"}

    def test_svc_complex_empty_name(self) -> None:
        """Test tt.svc('complex') with empty name."""
        service_class = tt.svc("complex")
        service = service_class(name="")

        result = service.execute()
        assertion_helpers.assert_flext_result_failure(result)
        assert "Name is required" in result.error

    def test_svc_complex_negative_amount(self) -> None:
        """Test tt.svc('complex') with negative amount."""
        service_class = tt.svc("complex")
        service = service_class(amount=-10)

        result = service.execute()
        assertion_helpers.assert_flext_result_failure(result)
        assert "Amount must be non-negative" in result.error

    def test_svc_complex_disabled_with_amount(self) -> None:
        """Test tt.svc('complex') disabled with amount."""
        service_class = tt.svc("complex")
        service = service_class(enabled=False, amount=100)

        result = service.execute()
        assertion_helpers.assert_flext_result_failure(result)
        assert "Cannot have amount when disabled" in result.error

    def test_svc_validate_business_rules_complex_valid(self) -> None:
        """Test validate_business_rules for complex service with valid data."""
        service_class = tt.svc("complex")
        service = service_class(name="Test", amount=100, enabled=True)

        result = service.validate_business_rules()
        assertion_helpers.assert_flext_result_success(result)
        assert result.value is True

    def test_svc_validate_business_rules_complex_invalid(self) -> None:
        """Test validate_business_rules for complex service with invalid data."""
        service_class = tt.svc("complex")
        service = service_class(name="")

        result = service.validate_business_rules()
        assertion_helpers.assert_flext_result_failure(result)
        assert "Name is required" in result.error

    def test_svc_validate_config_complex_valid(self) -> None:
        """Test validate_config for complex service with valid data."""
        service_class = tt.svc("complex")
        service = service_class(name="Test", amount=100)

        result = service.validate_config()
        assertion_helpers.assert_flext_result_success(result)
        assert result.value is True

    def test_svc_validate_config_name_too_long(self) -> None:
        """Test validate_config for complex service with name too long."""
        service_class = tt.svc("complex")
        long_name = "a" * 51
        service = service_class(name=long_name)

        result = service.validate_config()
        assertion_helpers.assert_flext_result_failure(result)
        assert "Name too long" in result.error

    def test_svc_validate_config_amount_too_large(self) -> None:
        """Test validate_config for complex service with amount too large."""
        service_class = tt.svc("complex")
        service = service_class(amount=1001)

        result = service.validate_config()
        assertion_helpers.assert_flext_result_failure(result)
        assert "Value too large" in result.error

    def test_svc_validate_config_non_complex(self) -> None:
        """Test validate_config for non-complex service."""
        service_class = tt.svc("test")
        service = service_class()

        result = service.validate_config()
        assertion_helpers.assert_flext_result_success(result)
        assert result.value is True

    def test_svc_validate_business_rules_non_complex(self) -> None:
        """Test validate_business_rules for non-complex service."""
        service_class = tt.svc("test")
        service = service_class()

        result = service.validate_business_rules()
        # Should call super() which returns success
        assert isinstance(result, FlextResult)


# =============================================================================
# PHASE 7: Tests for Unified Factory Methods (tt.model, tt.res, tt.list, tt.dict, tt.generic)
# =============================================================================
class TestsFlextTestsFactoriesModel:
    """Tests for tt.model() unified method."""

    def test_model_user_default(self) -> None:
        """Test user model creation with defaults."""
        user_result = tt.model("user")
        user = _extract_model(user_result)
        assert isinstance(user, User)
        assert user.id is not None
        assert user.name == "Test User"
        assert "@example.com" in user.email
        assert user.active is True

    def test_model_user_custom(self) -> None:
        """Test user model creation with custom parameters."""
        user_result = tt.model("user", name="Custom User", email="custom@test.com")
        user = _extract_model(user_result)
        assert isinstance(user, User)
        assert user.name == "Custom User"
        assert user.email == "custom@test.com"

    def test_model_batch(self) -> None:
        """Test batch model creation."""
        users = tt.model("user", count=5)
        assert isinstance(users, list)
        assert len(users) == 5
        assert all(isinstance(user, User) for user in users)

    def test_model_as_result(self) -> None:
        """Test model wrapped in FlextResult."""
        result = tt.model("user", as_result=True)
        assert isinstance(result, FlextResult)
        typed_result = cast(FlextResult[_BaseModel], cast(object, result))
        assertion_helpers.assert_flext_result_success(typed_result)
        assert isinstance(result.value, User)

    def test_model_as_dict(self) -> None:
        """Test model returned as dict."""
        users_dict = tt.model("user", count=3, as_dict=True)
        assert isinstance(users_dict, dict)
        assert len(users_dict) == 3
        assert all(isinstance(v, User) for v in users_dict.values())

    def test_model_config(self) -> None:
        """Test config model creation."""
        config = tt.model("config", environment="production")
        assert isinstance(config, Config)
        assert config.environment == "production"

    def test_model_service(self) -> None:
        """Test service model creation."""
        service_result = tt.model("service", service_type="database")
        service = _extract_model(service_result)
        assert isinstance(service, Service)
        assert service.type == "database"

    def test_model_entity(self) -> None:
        """Test entity model creation."""
        entity = tt.model("entity", name="Test Entity", value=42)
        assert isinstance(entity, Entity)
        assert entity.name == "Test Entity"

    def test_model_value_object(self) -> None:
        """Test value object model creation."""
        value_obj = tt.model("value", data="test_data", value_count=3)
        assert isinstance(value_obj, ValueObject)
        assert value_obj.data == "test_data"

    def test_model_with_transform(self) -> None:
        """Test model creation with transform function."""
        user_result = tt.model(
            "user",
            name="Original",
            transform=cast(
                t_test.Tests.TestResultValue,
                cast(
                    object,
                    lambda u: User(
                        id=cast(User, u).id,
                        name="Transformed",
                        email=cast(User, u).email,
                        active=cast(User, u).active,
                    ),
                ),
            ),
        )
        user = _extract_model(user_result)
        assert isinstance(user, User)
        assert user.name == "Transformed"

    def test_model_with_validate(self) -> None:
        """Test model creation with validation."""
        user_result = tt.model(
            "user",
            active=True,
            validate=cast(
                t_test.Tests.TestResultValue,
                cast(object, lambda u: cast(User, u).active),
            ),
        )
        user = _extract_model(user_result)
        assert isinstance(user, User)
        assert user.active is True

        result_raw = tt.model(
            "user",
            active=False,
            validate=cast(
                t_test.Tests.TestResultValue,
                cast(object, lambda u: cast(User, u).active),
            ),
            as_result=True,
        )
        if isinstance(result_raw, r):
            result = result_raw
        elif isinstance(result_raw, _BaseModel):
            msg = f"Expected r[BaseModel], got BaseModel: {type(result_raw)}"
            raise AssertionError(msg)
        else:
            msg = f"Expected r[BaseModel], got {type(result_raw)}"
            raise AssertionError(msg)
        assert isinstance(result, FlextResult)
        assert result.is_failure


class TestsFlextTestsFactoriesRes:
    """Tests for tt.res() unified method."""

    def test_res_ok(self) -> None:
        """Test successful result creation."""
        result_raw = tt.res("ok", value=42)
        result = cast(r[int], cast(object, _as_single_payload_result(result_raw)))
        assert isinstance(result, FlextResult)
        assertion_helpers.assert_flext_result_success(result)
        assert result.value == 42

    def test_res_fail(self) -> None:
        """Test failed result creation."""
        result_raw = tt.res("fail", error="Error message")
        result = cast(
            r[t_test.Tests.PayloadValue], _as_single_payload_result(result_raw)
        )
        assert isinstance(result, FlextResult)
        assertion_helpers.assert_flext_result_failure(result)
        assert result.error == "Error message"

    def test_res_fail_with_code(self) -> None:
        """Test failed result creation with error code."""
        result_raw = tt.res(
            "fail",
            error="Error message",
            error_code="ERR001",
        )
        result = cast(
            r[t_test.Tests.PayloadValue], _as_single_payload_result(result_raw)
        )
        assertion_helpers.assert_flext_result_failure(result)
        assert result.error == "Error message"
        # Note: error_code may be stored in result metadata

    def test_res_from_value_success(self) -> None:
        """Test from_value with non-None value."""
        result_raw = tt.res("from_value", value=42)
        result = cast(r[int], cast(object, _as_single_payload_result(result_raw)))
        assertion_helpers.assert_flext_result_success(result)
        assert result.value == 42

    def test_res_from_value_none(self) -> None:
        """Test from_value with None value."""
        result_raw = tt.res(
            "from_value",
            value=None,
            error_on_none="Value is required",
        )
        result = cast(
            r[t_test.Tests.PayloadValue], _as_single_payload_result(result_raw)
        )
        assertion_helpers.assert_flext_result_failure(result)
        error_msg = result.error or ""
        assert "required" in error_msg.lower()

    def test_res_batch_values(self) -> None:
        """Test batch result creation from values."""
        results_raw = tt.res("ok", values=[1, 2, 3])
        results = cast(
            list[r[int]],
            cast(
                object,
                results_raw if isinstance(results_raw, list) else [results_raw],
            ),
        )
        assert isinstance(results, list)
        assert len(results) == 3
        assert all(result.is_success for result in results)
        assert [result.value for result in results] == [1, 2, 3]

    def test_res_batch_errors(self) -> None:
        """Test batch result creation from errors."""
        results_raw = tt.res(
            "fail",
            errors=["err1", "err2"],
        )
        results = cast(
            list[r[t_test.Tests.PayloadValue]],
            results_raw if isinstance(results_raw, list) else [results_raw],
        )
        assert isinstance(results, list)
        assert len(results) == 2
        assert all(result.is_failure for result in results)

    def test_res_mix_pattern(self) -> None:
        """Test batch result creation with mix pattern."""
        results_raw = tt.res(
            "ok",
            values=[1, 2],
            errors=["e1", "e2"],
            mix_pattern=[True, False, True, False],
        )
        results = cast(
            list[r[int]],
            cast(
                object,
                results_raw if isinstance(results_raw, list) else [results_raw],
            ),
        )
        assert len(results) == 4
        assert results[0].is_success and results[0].value == 1
        assert results[1].is_failure
        assert results[2].is_success and results[2].value == 2
        assert results[3].is_failure

    def test_res_with_transform(self) -> None:
        """Test result creation with transform function."""
        # transform is a special kwarg processed by ResultFactoryParams
        # Type ignore needed because transform is Callable, not TestResultValue
        result_raw = tt.res(
            "ok",
            value=5,
            transform=cast(
                t_test.Tests.TestResultValue,
                cast(object, lambda x: cast(int, x) * 2),
            ),
        )
        result = cast(r[int], cast(object, _as_single_payload_result(result_raw)))
        assertion_helpers.assert_flext_result_success(result)
        assert result.value == 10


class TestsFlextTestsFactoriesList:
    """Tests for tt.list() method."""

    def test_list_from_model(self) -> None:
        """Test list creation from model kind."""
        users_raw = tt.list("user", count=3)
        users = cast(list[User], cast(object, _as_payload_list(users_raw)))
        assert isinstance(users, list)
        assert len(users) == 3
        assert all(isinstance(u, User) for u in users)

    def test_list_from_callable(self) -> None:
        """Test list creation from callable factory."""
        numbers_raw = tt.list(lambda: 42, count=5)
        numbers = cast(list[int], cast(object, _as_payload_list(numbers_raw)))
        assert numbers == [42, 42, 42, 42, 42]

    def test_list_from_sequence(self) -> None:
        """Test list creation from sequence."""
        # transform is a special kwarg processed by ListFactoryParams
        # Type ignore needed because transform is Callable, not TestResultValue
        doubled_raw = tt.list(
            [1, 2, 3],
            transform=cast(
                t_test.Tests.TestResultValue,
                cast(object, lambda x: cast(int, x) * 2),
            ),
        )
        doubled = cast(list[int], cast(object, _as_payload_list(doubled_raw)))
        assert doubled == [2, 4, 6]

    def test_list_with_filter(self) -> None:
        """Test list creation with filter."""
        # filter_ is a special kwarg processed by ListFactoryParams
        # Type ignore needed because filter_ is Callable, not TestResultValue
        evens_raw = tt.list(
            [1, 2, 3, 4, 5],
            filter_=cast(
                t_test.Tests.TestResultValue,
                cast(object, lambda x: cast(int, x) % 2 == 0),
            ),
        )
        evens = cast(list[int], cast(object, _as_payload_list(evens_raw)))
        assert evens == [2, 4]

    def test_list_with_unique(self) -> None:
        """Test list creation with uniqueness."""
        # Create list with duplicates
        items_raw = tt.list([1, 2, 2, 3, 3, 3], unique=True)
        items = cast(list[int], cast(object, _as_payload_list(items_raw)))
        assert len(items) == 3
        assert set(items) == {1, 2, 3}

    def test_list_as_result(self) -> None:
        """Test list creation wrapped in result."""
        result_raw = tt.list(
            "user",
            count=3,
            as_result=True,
        )
        assert isinstance(result_raw, r)
        result = cast(r[list[User]], cast(object, result_raw))
        assert isinstance(result, FlextResult)
        assertion_helpers.assert_flext_result_success(result)
        assert len(result.value) == 3


class TestsFlextTestsFactoriesDict:
    """Tests for tt.dict_factory() method."""

    def test_dict_from_model(self) -> None:
        """Test dict creation from model kind."""
        users_raw = tt.dict_factory(
            "user",
            count=3,
        )
        users = cast(dict[str, User], cast(object, _as_payload_mapping(users_raw)))
        assert isinstance(users, dict)
        assert len(users) == 3
        assert all(isinstance(u, User) for u in users.values())

    def test_dict_with_key_factory(self) -> None:
        """Test dict creation with key factory."""
        # key_factory is a special kwarg processed by DictFactoryParams
        # Type ignore needed because key_factory is Callable, not TestResultValue
        users_raw = tt.dict_factory(
            "user",
            count=3,
            key_factory=cast(
                t_test.Tests.TestResultValue,
                cast(object, lambda i: f"user_{cast(int, i)}"),
            ),
        )
        users = cast(dict[str, User], cast(object, _as_payload_mapping(users_raw)))
        assert set(users.keys()) == {"user_0", "user_1", "user_2"}

    def test_dict_with_value_factory(self) -> None:
        """Test dict creation with value factory."""

        def value_factory(key: str) -> User:
            return User(id=key, name=f"User {key}", email=f"{key}@test.com")

        # value_factory is a special kwarg processed by DictFactoryParams
        # Type ignore needed because value_factory is Callable, not TestResultValue
        users_raw = tt.dict_factory(
            "user",
            count=2,
            value_factory=cast(
                t_test.Tests.TestResultValue,
                cast(object, value_factory),
            ),
        )
        users = cast(dict[str, User], cast(object, _as_payload_mapping(users_raw)))
        assert len(users) == 2

    def test_dict_from_mapping(self) -> None:
        """Test dict creation from existing mapping."""
        existing = {"a": 1, "b": 2}
        merged_raw = tt.dict_factory(
            existing,
            merge_with={"c": 3},
        )
        merged = cast(dict[str, int], cast(object, _as_payload_mapping(merged_raw)))
        assert merged == {"a": 1, "b": 2, "c": 3}

    def test_dict_as_result(self) -> None:
        """Test dict creation wrapped in result."""
        result_raw = tt.dict_factory(
            "user",
            count=3,
            as_result=True,
        )
        assert isinstance(result_raw, r)
        result = cast(r[dict[str, User]], cast(object, result_raw))
        assert isinstance(result, FlextResult)
        assertion_helpers.assert_flext_result_success(result)
        assert len(result.value) == 3


class TestsFlextTestsFactoriesGeneric:
    """Tests for tt.generic() method."""

    def test_generic_simple(self) -> None:
        """Test generic type instantiation."""

        class SimpleClass:
            def __init__(self, name: str) -> None:
                self.name = name

        obj = tt.generic(SimpleClass, kwargs={"name": "test"})
        assert isinstance(obj, SimpleClass)
        assert obj.name == "test"

    def test_generic_with_args(self) -> None:
        """Test generic type instantiation with positional args."""

        class ArgsClass:
            def __init__(self, a: int, b: int, c: str = "default") -> None:
                self.a = a
                self.b = b
                self.c = c

        obj_result = tt.generic(ArgsClass, args=[1, 2], kwargs={"c": "custom"})
        if isinstance(obj_result, r):
            obj = obj_result.value
        elif isinstance(obj_result, list):
            obj = obj_result[0]
        else:
            obj = obj_result
        assert isinstance(obj, ArgsClass)
        assert obj.a == 1
        assert obj.b == 2
        assert obj.c == "custom"

    def test_generic_batch(self) -> None:
        """Test batch generic type instantiation."""

        class BatchClass:
            def __init__(self, value: int) -> None:
                self.value = value

        objs = tt.generic(BatchClass, kwargs={"value": 42}, count=5)
        assert isinstance(objs, list)
        assert len(objs) == 5
        assert all(isinstance(o, BatchClass) for o in objs)
        assert all(o.value == 42 for o in objs)

    def test_generic_with_validate(self) -> None:
        """Test generic type instantiation with validation."""

        class ValidatedClass:
            def __init__(self, age: int) -> None:
                self.age = age

        obj_result = tt.generic(
            ValidatedClass,
            kwargs={"age": 25},
            validate=cast(
                t_test.Tests.TestResultValue,
                cast(object, lambda o: cast(ValidatedClass, o).age >= 18),
            ),
        )
        if isinstance(obj_result, r):
            obj = obj_result.value
        elif isinstance(obj_result, list):
            obj = obj_result[0]
        else:
            obj = obj_result
        assert isinstance(obj, ValidatedClass)
        assert obj.age == 25

        with pytest.raises(ValueError, match="Validation failed"):
            tt.generic(
                ValidatedClass,
                kwargs={"age": 15},
                validate=cast(
                    t_test.Tests.TestResultValue,
                    cast(object, lambda o: cast(ValidatedClass, o).age >= 18),
                ),
            )

    def test_generic_as_result(self) -> None:
        """Test generic type instantiation wrapped in result."""

        class ResultClass:
            def __init__(self, value: str) -> None:
                self.value = value

        result = tt.generic(ResultClass, kwargs={"value": "test"}, as_result=True)
        assert isinstance(result, FlextResult)
        typed_result = cast(FlextResult[ResultClass], cast(object, result))
        assertion_helpers.assert_flext_result_success(typed_result)
        assert isinstance(result.value, ResultClass)
        assert result.value.value == "test"
