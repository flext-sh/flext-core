"""Behavioral tests for the public service contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, ClassVar, override

import pytest

from tests import c, e, m, p, r, s, u


class TestsFlextCoreServiceUserData(m.Value):
    """Public result model used by service tests."""

    user_id: int
    name: str


class TestsFlextCoreServiceUserService(s[TestsFlextCoreServiceUserData]):
    """Simple successful service."""

    @override
    def execute(self) -> p.Result[TestsFlextCoreServiceUserData]:
        return r[TestsFlextCoreServiceUserData].ok(
            TestsFlextCoreServiceUserData(user_id=1, name="test_user")
        )


class TestsFlextCoreServiceValidatingService(s[str]):
    """Service with public validation behavior."""

    model_config: ClassVar[m.ConfigDict] = m.ConfigDict(validate_assignment=True)
    value_input: Annotated[
        str,
        u.Field(description="Text value subjected to length validation."),
    ] = "valid"
    min_length: Annotated[
        int,
        u.Field(description="Minimum accepted length for value_input."),
    ] = 3

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


class TestsFlextCoreServiceFailingService(s[bool]):
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


class TestsFlextCoreServiceRaisingValidationService(s[str]):
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
        service = TestsFlextCoreServiceUserService(subproject="core-tests")
        assert isinstance(service, s)
        assert isinstance(service.model_config, Mapping)
        assert service.model_config.get("validate_assignment") is True
        assert service.settings is not None
        assert service.context is not None
        assert service.container is not None

    def test_execute_returns_successful_result(self) -> None:
        service = TestsFlextCoreServiceUserService()
        result = service.execute()
        assert result.success
        assert result.value == TestsFlextCoreServiceUserData(
            user_id=1, name="test_user"
        )

    def test_result_property_unwraps_successful_execution(self) -> None:
        service = TestsFlextCoreServiceUserService()
        assert service.result == TestsFlextCoreServiceUserData(
            user_id=1, name="test_user"
        )

    def test_result_property_raises_structured_base_error_on_failure(self) -> None:
        service = TestsFlextCoreServiceFailingService()
        with pytest.raises(e.BaseError) as captured:
            _ = service.result
        error = captured.value
        assert error.error_code == c.ErrorCode.VALIDATION_ERROR
        assert error.correlation_id == "svc-corr-1"
        assert error.metadata.attributes["field"] == "name"
        assert error.metadata.attributes["operation"] == "service execution"

    def test_default_business_rule_validation_is_successful(self) -> None:
        service = TestsFlextCoreServiceUserService()
        result = service.validate_business_rules()
        assert result.success
        assert result.value is True

    def test_valid_reflects_business_rule_result(self) -> None:
        assert (
            TestsFlextCoreServiceValidatingService(value_input="valid").valid() is True
        )
        assert (
            TestsFlextCoreServiceValidatingService(
                value_input="x", min_length=2
            ).valid()
            is False
        )

    def test_valid_returns_false_when_validation_raises(self) -> None:
        service = TestsFlextCoreServiceRaisingValidationService(should_raise=True)
        assert service.valid() is False

    def test_execute_exposes_validation_failure_without_exceptions(self) -> None:
        service = TestsFlextCoreServiceValidatingService(value_input="x", min_length=2)
        result = service.execute()
        assert result.failure
        assert result.error == "Value is too short"

    def test_service_info_exposes_public_runtime_metadata(self) -> None:
        service = TestsFlextCoreServiceUserService(subproject="core-tests")
        info = service.service_info()
        assert info["service_name"] == "TestsFlextCoreServiceUserService"
        assert info["service_module"] == TestsFlextCoreServiceUserService.__module__
        assert info["settings_class"] == service.settings.__class__.__name__
        assert info["app_name"] == service.settings.app_name
        assert info["version"] == service.settings.version
        assert info["subproject"] == "core-tests"
        assert info["handler_count"] == 0

    def test_service_model_dump_exposes_runtime_snapshot(self) -> None:
        service = TestsFlextCoreServiceUserService()
        payload = service.model_dump(mode="python")
        assert "runtime" in payload
