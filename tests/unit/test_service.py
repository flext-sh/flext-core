"""Behavioral tests for the public service contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar, override

import pytest
from pydantic import ConfigDict

from tests import c, e, m, p, r, s


class UserData(m.Value):
    """Public result model used by service tests."""

    user_id: int
    name: str


class UserService(s[UserData]):
    """Simple successful service."""

    @override
    def execute(self) -> p.Result[UserData]:
        return r[UserData].ok(UserData(user_id=1, name="test_user"))


class ValidatingService(s[str]):
    """Service with public validation behavior."""

    model_config: ClassVar[ConfigDict] = ConfigDict(validate_assignment=True)
    value_input: str = "valid"
    min_length: int = 3

    @override
    def validate_business_rules(self) -> p.Result[bool]:
        if len(self.value_input) < self.min_length:
            return r[bool].fail("Value is too short")
        return r[bool].ok(True)

    @override
    def execute(self) -> p.Result[str]:
        if len(self.value_input) < self.min_length:
            return r[str].fail("Value is too short")
        return r[str].ok(f"Processed: {self.value_input}")


class FailingService(s[bool]):
    """Service that fails execution through r."""

    @override
    def execute(self) -> p.Result[bool]:
        return r[bool].fail(
            "Missing required data",
            error_code=c.ErrorCode.VALIDATION_ERROR,
            error_data={
                "field": "name",
                c.ContextKey.CORRELATION_ID: "svc-corr-1",
            },
        )


class RaisingValidationService(s[str]):
    """Service whose validation raises and must be flattened by valid()."""

    should_raise: bool = False

    @override
    def validate_business_rules(self) -> p.Result[bool]:
        if self.should_raise:
            message = "validation exploded"
            raise ValueError(message)
        return r[bool].ok(True)

    @override
    def execute(self) -> p.Result[str]:
        return r[str].ok("ok")


class TestService:
    """Validate the public behavior of FlextService subclasses."""

    def test_service_creation_exposes_runtime_contract(self) -> None:
        service = UserService(subproject="core-tests")
        assert isinstance(service, s)
        assert isinstance(service.model_config, Mapping)
        assert service.model_config.get("validate_assignment") is True
        assert service.settings is not None
        assert service.context is not None
        assert service.container is not None

    def test_execute_returns_successful_result(self) -> None:
        service = UserService()
        result = service.execute()
        assert result.success
        assert result.value == UserData(user_id=1, name="test_user")

    def test_result_property_unwraps_successful_execution(self) -> None:
        service = UserService()
        assert service.result == UserData(user_id=1, name="test_user")

    def test_result_property_raises_structured_base_error_on_failure(self) -> None:
        service = FailingService()
        with pytest.raises(e.BaseError) as captured:
            _ = service.result
        error = captured.value
        assert error.error_code == c.ErrorCode.VALIDATION_ERROR
        assert error.correlation_id == "svc-corr-1"
        assert error.metadata.attributes["field"] == "name"
        assert error.metadata.attributes["operation"] == "service execution"

    def test_default_business_rule_validation_is_successful(self) -> None:
        service = UserService()
        result = service.validate_business_rules()
        assert result.success
        assert result.value is True

    def test_valid_reflects_business_rule_result(self) -> None:
        assert ValidatingService(value_input="valid").valid() is True
        assert ValidatingService(value_input="x", min_length=2).valid() is False

    def test_valid_returns_false_when_validation_raises(self) -> None:
        service = RaisingValidationService(should_raise=True)
        assert service.valid() is False

    def test_execute_exposes_validation_failure_without_exceptions(self) -> None:
        service = ValidatingService(value_input="x", min_length=2)
        result = service.execute()
        assert result.failure
        assert result.error == "Value is too short"

    def test_service_info_exposes_public_runtime_metadata(self) -> None:
        service = UserService(subproject="core-tests")
        info = service.service_info()
        assert info["service_name"] == "UserService"
        assert info["service_module"] == UserService.__module__
        assert info["settings_class"] == service.settings.__class__.__name__
        assert info["app_name"] == service.settings.app_name
        assert info["version"] == service.settings.version
        assert info["subproject"] == "core-tests"
        assert info["handler_count"] == 0

    def test_service_model_dump_exposes_runtime_snapshot(self) -> None:
        service = UserService()
        payload = service.model_dump(mode="python")
        assert "runtime" in payload
