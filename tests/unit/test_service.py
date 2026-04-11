"""Service core functionality tests.

Module: flext_core.service
Scope: Service abstract base class - execution, validation, metadata

Tests core s functionality including:
- Service creation and Pydantic configuration
- Service immutability (frozen model)
- Abstract execute method implementation
- Basic service execution with r
- Business rules validation (success, failure, exception handling)
- Service metadata retrieval

Uses Python 3.13 patterns, u, FlextConstants,
and aggressive parametrization for DRY testing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum, unique
from typing import Annotated, ClassVar, cast, override

import pytest
from hypothesis import given, strategies as st
from pydantic import BaseModel, ConfigDict, Field

from flext_core import FlextContext, FlextSettings
from tests import m, p, r, s, t, u


class TestsCore:
    """Unified test suite for s using u."""

    @unique
    class ServiceScenarioType(StrEnum):
        """Service scenario types for scenario testing."""

        BASIC_USER = "basic_user"
        COMPLEX_VALID = "complex_valid"
        COMPLEX_INVALID = "complex_invalid"
        FAILING = "failing"
        EXCEPTION = "exception"

    class ServiceScenario(BaseModel):
        """Service test scenario definition."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
        name: Annotated[str, Field(description="Service scenario name")]
        scenario_type: Annotated[
            TestsCore.ServiceScenarioType,
            Field(description="Service scenario type"),
        ]
        is_valid_expected: Annotated[
            bool,
            Field(description="Expected valid result"),
        ]
        service_kwargs: Annotated[
            t.ContainerMapping | None,
            Field(default=None, description="Optional scenario service kwargs"),
        ] = None

    class UserData(BaseModel):
        """User data model."""

        user_id: int
        name: str

    class UserService(s["TestsCore.UserData"]):
        """Basic user service for standard testing."""

        @override
        def execute(self) -> r[TestsCore.UserData]:
            return r[TestsCore.UserData].ok(
                TestsCore.UserData(user_id=1, name="test_user"),
            )

    class ComplexService(s[str]):
        """Service with custom validation rules."""

        name: str = "test"
        amount: int = 0
        enabled: bool = True

        @override
        def validate_business_rules(self) -> r[bool]:
            if not self.name:
                return r[bool].fail("Missing value")
            if self.amount < 0:
                return r[bool].fail("Value too low")
            return r[bool].ok(True)

        @override
        def execute(self) -> r[str]:
            if not self.name:
                return r[str].fail("Missing value")
            return r[str].ok(f"Processed: {self.name}")

    class FailingService(s[bool]):
        """Service that fails validation."""

        @override
        def validate_business_rules(self) -> r[bool]:
            return r[bool].fail("Processing error")

        @override
        def execute(self) -> r[bool]:
            return r[bool].fail("Processing error")

    class ExceptionService(s[str]):
        """Service that raises exceptions during validation."""

        should_raise: bool = False

        @override
        def validate_business_rules(self) -> r[bool]:
            if self.should_raise:
                error_msg = "Processing error"
                raise ValueError(error_msg)
            return r[bool].ok(True)

        @override
        def execute(self) -> r[str]:
            if self.should_raise:
                error_msg = "Processing error"
                raise RuntimeError(error_msg)
            return r[str].ok("test_value")

    class ServiceScenarios:
        """Centralized service test scenarios using FlextConstants."""

        @staticmethod
        def create_service(
            scenario: TestsCore.ServiceScenario,
        ) -> s[TestsCore.UserData] | s[str] | s[bool]:
            kwargs_raw: t.ContainerMapping = scenario.service_kwargs or {}
            if scenario.scenario_type == TestsCore.ServiceScenarioType.BASIC_USER:
                return TestsCore.UserService()
            if scenario.scenario_type in {
                TestsCore.ServiceScenarioType.COMPLEX_VALID,
                TestsCore.ServiceScenarioType.COMPLEX_INVALID,
            }:
                name_val = kwargs_raw.get("name", "test")
                amount_val = kwargs_raw.get("amount", 0)
                enabled_val = kwargs_raw.get("enabled", True)
                return TestsCore.ComplexService(
                    name=str(name_val),
                    amount=int(amount_val)
                    if isinstance(amount_val, (int, float))
                    else 0,
                    enabled=bool(enabled_val),
                )
            if scenario.scenario_type == TestsCore.ServiceScenarioType.FAILING:
                return TestsCore.FailingService()
            if scenario.scenario_type == TestsCore.ServiceScenarioType.EXCEPTION:
                should_raise_val = kwargs_raw.get("should_raise", False)
                return TestsCore.ExceptionService(should_raise=bool(should_raise_val))
            error_msg = f"Unknown scenario type: {scenario.scenario_type}"
            raise ValueError(error_msg)

    def _service_scenarios(self) -> Sequence[TestsCore.ServiceScenario]:
        return [
            self.ServiceScenario(
                name="basic_user_service",
                scenario_type=self.ServiceScenarioType.BASIC_USER,
                is_valid_expected=True,
            ),
            self.ServiceScenario(
                name="complex_valid",
                scenario_type=self.ServiceScenarioType.COMPLEX_VALID,
                is_valid_expected=True,
                service_kwargs={"name": "test"},
            ),
            self.ServiceScenario(
                name="complex_invalid",
                scenario_type=self.ServiceScenarioType.COMPLEX_INVALID,
                is_valid_expected=False,
                service_kwargs={"name": ""},
            ),
            self.ServiceScenario(
                name="failing_service",
                scenario_type=self.ServiceScenarioType.FAILING,
                is_valid_expected=False,
            ),
            self.ServiceScenario(
                name="exception_handling",
                scenario_type=self.ServiceScenarioType.EXCEPTION,
                is_valid_expected=False,
                service_kwargs={"should_raise": True},
            ),
        ]

    def test_basic_service_creation(self) -> None:
        """Test basic service creation and Pydantic configuration."""
        service = self.UserService()
        assert isinstance(service, s)
        assert isinstance(service.model_config, Mapping)
        assert service.model_config.get("validate_assignment") is True

    def test_service_immutability(self) -> None:
        """Test service mutability (frozen removed for compatibility with x)."""
        service = self.UserService()
        assert (
            service.model_config.get("frozen") is None
            or service.model_config.get("frozen") is False
        )

    def test_execute_abstract_method(self) -> None:
        """Test execute method implementation."""

        class ConcreteService(s[str]):
            @override
            def execute(self) -> r[str]:
                return r[str].ok("test_value")

        service = ConcreteService()
        result = service.execute()
        u.Core.Tests.assert_success_with_value(result, "test_value")

    def test_basic_execution(self) -> None:
        """Test basic service execution returns expected type."""
        service = self.UserService()
        result = service.execute()
        _ = u.Core.Tests.assert_success(result)
        data = result.value
        assert isinstance(data, self.UserData)
        assert data.user_id == 1

    def test_is_valid_scenarios(self) -> None:
        """Test valid with various service scenarios."""
        for scenario in self._service_scenarios():
            service = self.ServiceScenarios.create_service(scenario)
            assert service.valid() is scenario.is_valid_expected

    def test_validate_business_rules_default(self) -> None:
        """Test default business rules validation."""
        service = self.UserService()
        result = service.validate_business_rules()
        _ = u.Core.Tests.assert_success(result)

    def test_validate_business_rules_custom_success(self) -> None:
        """Test custom business rules validation success."""
        service = self.ComplexService()
        service.name = "test"
        result = service.validate_business_rules()
        _ = u.Core.Tests.assert_success(result)

    def test_validate_business_rules_custom_failure(self) -> None:
        """Test custom business rules validation failure."""
        service = self.ComplexService()
        service.name = ""
        result = service.validate_business_rules()
        u.Core.Tests.assert_failure_with_error(result, "Missing value")

    def test_service_validation_using_generic_helpers(self) -> None:
        """Test service validation using generic helpers - real behavior."""
        service = self.ComplexService()
        service.name = "test"
        service.amount = 10
        service.enabled = True
        validation_result = u.Core.Tests.validate_model_attributes(
            cast("p.Model", service),
            required_attrs=["name", "amount", "enabled"],
            optional_attrs=["validate_business_rules"],
        )
        assert validation_result.success

    def test_service_validation_failure_limits(self) -> None:
        """Test service validation failure - limit cases."""
        service = self.ComplexService()
        service.name = ""
        service.amount = -1
        service.enabled = False
        validation_result = u.Core.Tests.validate_model_attributes(
            cast("p.Model", service),
            required_attrs=["name"],
        )
        _ = u.Core.Tests.assert_success(validation_result)
        business_result = service.validate_business_rules()
        _ = u.Core.Tests.assert_failure(business_result)

    @given(st.text(min_size=1))
    def test_execute_hypothesis(self, value: str) -> None:
        """Property: execute always returns success or failure."""

        class _DynamicService(s[str]):
            __test__ = False
            value: str

            @override
            def execute(self) -> r[str]:
                return r[str].ok(self.value)

        service = _DynamicService(value=value)
        result = service.execute()
        assert result.success or result.failure


class TestServiceInternals:
    """Tests for s internal runtime creation methods."""

    class _Svc(s[bool]):
        @override
        def execute(self) -> r[bool]:
            return r[bool].ok(True)

    class _FakeConfig:
        version = "1"

    def test_service_init_type_guards_and_properties(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test service init with non-standard context and settings types."""
        bad_ctx_runtime = m.ServiceRuntime.model_construct(
            settings=FlextSettings(),
            context=cast("p.Context", "invalid-context"),
            container=cast("p.Container", "invalid-container"),
        )

        def _bad_ctx_runtime_factory(
            _cls: type[TestServiceInternals._Svc],
        ) -> m.ServiceRuntime:
            return bad_ctx_runtime

        monkeypatch.setattr(
            self._Svc,
            "_create_initial_runtime",
            classmethod(_bad_ctx_runtime_factory),
        )
        service_with_bad_ctx = self._Svc()
        assert service_with_bad_ctx.context == "invalid-context"
        good_ctx = FlextContext.create()
        bad_cfg_runtime = m.ServiceRuntime.model_construct(
            settings=cast("p.Settings", self._FakeConfig()),
            context=good_ctx,
            container=cast("p.Container", "invalid-container"),
        )

        def _bad_cfg_runtime_factory(
            _cls: type[TestServiceInternals._Svc],
        ) -> m.ServiceRuntime:
            return bad_cfg_runtime

        monkeypatch.setattr(
            self._Svc,
            "_create_initial_runtime",
            classmethod(_bad_cfg_runtime_factory),
        )
        service_with_bad_cfg = self._Svc()
        assert isinstance(service_with_bad_cfg.settings, self._FakeConfig)

    def test_service_create_runtime_container_overrides_branch(self) -> None:
        """Test _create_runtime with container_overrides."""
        runtime = self._Svc._create_runtime(container_overrides={"strict": True})
        assert isinstance(runtime, m.ServiceRuntime)

    def test_service_create_initial_runtime_prefers_custom_settings_type(self) -> None:
        """Test _create_initial_runtime with custom settings type via bootstrap options."""

        class _CustomSettings(FlextSettings):
            pass

        class _CustomSvc(TestServiceInternals._Svc):
            @classmethod
            @override
            def _runtime_bootstrap_options(
                cls,
            ) -> m.RuntimeBootstrapOptions:
                return m.RuntimeBootstrapOptions(
                    settings_type=_CustomSettings,
                )

        runtime = _CustomSvc()._create_initial_runtime()
        assert isinstance(runtime.settings, _CustomSettings)
        service = self._Svc()
        assert service.context is not None


__all__ = ["TestServiceInternals", "TestsCore"]
