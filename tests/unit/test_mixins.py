"""Tests for FlextMixins infrastructure - Container, Context, Logging, Metrics, Service.

Module: flext_core.mixins
Scope: FlextMixins - all nested mixin classes

Tests FlextMixins functionality including:
- Container mixin (_register_in_container)
- Context mixin (context property, _propagate_context, correlation IDs)
- Logging mixin (_log_with_context)
- Metrics mixin (track context manager)
- Service mixin (_init_service, _enrich_context, _with_operation_context)
- ModelConversion nested class (to_dict conversions)
- ResultHandling nested class (ensure_result wrapping)

Uses Python 3.13 patterns (StrEnum, frozen dataclasses with slots),
centralized constants, and parametrization for DRY testing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, ClassVar

import pytest
from pydantic import BaseModel

from flext_core import FlextContext, FlextMixins, FlextResult

# =========================================================================
# Mixin Scenario Type Enumerations
# =========================================================================


class ServiceMixinScenarioType(StrEnum):
    """Service mixin test scenario types."""

    CONTAINER_REGISTER = "container_register"
    CONTEXT_PROPERTY = "context_property"
    CONTEXT_PROPAGATE = "context_propagate"
    CONTEXT_CORRELATION = "context_correlation"
    LOGGING_WITH_CONTEXT = "logging_with_context"
    METRICS_TRACK = "metrics_track"
    SERVICE_INIT = "service_init"
    SERVICE_ENRICH = "service_enrich"


class ModelConversionScenarioType(StrEnum):
    """ModelConversion test scenario types."""

    WITH_BASEMODEL = "with_basemodel"
    WITH_DICT = "with_dict"
    WITH_NONE = "with_none"


class ResultHandlingScenarioType(StrEnum):
    """ResultHandling test scenario types."""

    RAW_VALUE = "raw_value"
    EXISTING_RESULT = "existing_result"
    TYPE_PRESERVATION = "type_preservation"


# =========================================================================
# Test Case Structures
# =========================================================================


@dataclass(frozen=True, slots=True)
class ServiceMixinScenario:
    """Service mixin test scenario definition."""

    name: str
    scenario_type: ServiceMixinScenarioType
    needs_init: bool = False
    operation_context: str | None = None


@dataclass(frozen=True, slots=True)
class ModelConversionScenario:
    """ModelConversion test scenario definition."""

    name: str
    scenario_type: ModelConversionScenarioType
    input_value: Any
    expected_output: dict[str, object]


@dataclass(frozen=True, slots=True)
class ResultHandlingScenario:
    """ResultHandling test scenario definition."""

    name: str
    scenario_type: ResultHandlingScenarioType
    input_value: Any


# =========================================================================
# Test Scenario Factories
# =========================================================================


class ServiceMixinScenarios:
    """Factory for service mixin test scenarios."""

    SCENARIOS: ClassVar[list[ServiceMixinScenario]] = [
        ServiceMixinScenario(
            name="container_register_in_container",
            scenario_type=ServiceMixinScenarioType.CONTAINER_REGISTER,
        ),
        ServiceMixinScenario(
            name="context_mixin_property",
            scenario_type=ServiceMixinScenarioType.CONTEXT_PROPERTY,
        ),
        ServiceMixinScenario(
            name="context_propagate",
            scenario_type=ServiceMixinScenarioType.CONTEXT_PROPAGATE,
        ),
        ServiceMixinScenario(
            name="context_correlation_id",
            scenario_type=ServiceMixinScenarioType.CONTEXT_CORRELATION,
        ),
        ServiceMixinScenario(
            name="logging_with_context",
            scenario_type=ServiceMixinScenarioType.LOGGING_WITH_CONTEXT,
        ),
        ServiceMixinScenario(
            name="metrics_track",
            scenario_type=ServiceMixinScenarioType.METRICS_TRACK,
        ),
        ServiceMixinScenario(
            name="service_init_service",
            scenario_type=ServiceMixinScenarioType.SERVICE_INIT,
            needs_init=True,
        ),
        ServiceMixinScenario(
            name="service_enrich_context",
            scenario_type=ServiceMixinScenarioType.SERVICE_ENRICH,
            needs_init=True,
        ),
    ]


class ModelConversionScenarios:
    """Factory for ModelConversion test scenarios."""

    SCENARIOS: ClassVar[list[ModelConversionScenario]] = [
        ModelConversionScenario(
            name="to_dict_with_basemodel",
            scenario_type=ModelConversionScenarioType.WITH_BASEMODEL,
            input_value=None,  # Created in test
            expected_output={"name": "test", "value": 42},
        ),
        ModelConversionScenario(
            name="to_dict_with_dict",
            scenario_type=ModelConversionScenarioType.WITH_DICT,
            input_value={"key": "value", "number": 123},
            expected_output={"key": "value", "number": 123},
        ),
        ModelConversionScenario(
            name="to_dict_with_none",
            scenario_type=ModelConversionScenarioType.WITH_NONE,
            input_value=None,
            expected_output={},
        ),
    ]


class ResultHandlingScenarios:
    """Factory for ResultHandling test scenarios."""

    SCENARIOS: ClassVar[list[ResultHandlingScenario]] = [
        ResultHandlingScenario(
            name="ensure_result_raw_value",
            scenario_type=ResultHandlingScenarioType.RAW_VALUE,
            input_value=42,
        ),
        ResultHandlingScenario(
            name="ensure_result_existing_result",
            scenario_type=ResultHandlingScenarioType.EXISTING_RESULT,
            input_value=None,  # Created in test
        ),
        ResultHandlingScenario(
            name="ensure_result_type_preservation",
            scenario_type=ResultHandlingScenarioType.TYPE_PRESERVATION,
            input_value=None,  # Created in test
        ),
    ]


# =========================================================================
# Test Suite
# =========================================================================


class TestFlextMixinsNestedClasses:
    """Comprehensive test suite for nested mixin classes."""

    @pytest.mark.parametrize(
        "scenario",
        ServiceMixinScenarios.SCENARIOS,
        ids=lambda s: s.name,
    )
    def test_service_mixin_scenarios(self, scenario: ServiceMixinScenario) -> None:
        """Test service mixin functionality across scenarios."""

        class MyService(FlextMixins):
            """Test service for mixin scenarios."""

            def __init__(self) -> None:
                super().__init__()
                if scenario.needs_init:
                    self._init_service("MyTestService")

            def process(self) -> str:
                """Process operation for metrics testing."""
                with self.track("test_op") as metrics:
                    assert isinstance(metrics, dict)
                    time.sleep(0.01)
                    return "done"

        service = MyService()

        # Container mixin tests
        if scenario.scenario_type == ServiceMixinScenarioType.CONTAINER_REGISTER:
            result = service._register_in_container("test_service")
            assert result.is_success

        # Context mixin tests
        elif scenario.scenario_type == ServiceMixinScenarioType.CONTEXT_PROPERTY:
            assert isinstance(service.context, FlextContext)

        elif scenario.scenario_type == ServiceMixinScenarioType.CONTEXT_PROPAGATE:
            service._propagate_context("test_operation")

        elif scenario.scenario_type == ServiceMixinScenarioType.CONTEXT_CORRELATION:
            service._set_correlation_id("test-123")
            corr_id = service._get_correlation_id()
            assert corr_id == "test-123"

        # Logging mixin tests
        elif scenario.scenario_type == ServiceMixinScenarioType.LOGGING_WITH_CONTEXT:
            service._log_with_context("info", "Test message", extra_data="value")

        # Metrics mixin tests
        elif scenario.scenario_type == ServiceMixinScenarioType.METRICS_TRACK:
            result = service.process()
            assert result == "done"

        # Service mixin tests
        elif scenario.scenario_type == ServiceMixinScenarioType.SERVICE_INIT:
            assert hasattr(service, "logger")
            assert hasattr(service, "container")
            assert hasattr(service, "config")

        elif scenario.scenario_type == ServiceMixinScenarioType.SERVICE_ENRICH:
            service._enrich_context(version="1.0.0", team="test")

    @pytest.mark.parametrize(
        "scenario",
        ModelConversionScenarios.SCENARIOS,
        ids=lambda s: s.name,
    )
    def test_model_conversion_scenarios(
        self, scenario: ModelConversionScenario
    ) -> None:
        """Test ModelConversion.to_dict() with various input types."""
        if scenario.scenario_type == ModelConversionScenarioType.WITH_BASEMODEL:

            class TestModel(BaseModel):
                """Test model for conversion."""

                name: str
                value: int

            input_value = TestModel(name="test", value=42)
        else:
            input_value = scenario.input_value

        result = FlextMixins.ModelConversion.to_dict(input_value)

        assert isinstance(result, dict)
        assert result == scenario.expected_output

        if scenario.scenario_type == ModelConversionScenarioType.WITH_DICT:
            assert result is input_value  # Should return same dict instance

    @pytest.mark.parametrize(
        "scenario",
        ResultHandlingScenarios.SCENARIOS,
        ids=lambda s: s.name,
    )
    def test_result_handling_scenarios(self, scenario: ResultHandlingScenario) -> None:
        """Test ResultHandling.ensure_result() with various inputs."""
        if scenario.scenario_type == ResultHandlingScenarioType.RAW_VALUE:
            result = FlextMixins.ResultHandling.ensure_result(42)

            assert isinstance(result, FlextResult)
            assert result.is_success
            assert result.value == 42

        elif scenario.scenario_type == ResultHandlingScenarioType.EXISTING_RESULT:
            original = FlextResult[int].ok(100)
            result: FlextResult[int] = FlextMixins.ResultHandling.ensure_result(
                original
            )

            assert result is original  # Should return same instance
            assert result.is_success
            assert result.value == 100

        elif scenario.scenario_type == ResultHandlingScenarioType.TYPE_PRESERVATION:
            int_result = FlextMixins.ResultHandling.ensure_result(42)
            str_result = FlextMixins.ResultHandling.ensure_result("hello")
            list_result = FlextMixins.ResultHandling.ensure_result([1, 2, 3])

            assert int_result.value == 42
            assert str_result.value == "hello"
            assert list_result.value == [1, 2, 3]

    def test_service_mixin_with_operation_context(self) -> None:
        """Test Service mixin operation context workflow."""

        class MyService(FlextMixins):
            """Test service with operation context."""

            def __init__(self) -> None:
                super().__init__()
                self._init_service()

        service = MyService()
        service._with_operation_context("process_order", order_id="123")
        service._clear_operation_context()


__all__ = ["TestFlextMixinsNestedClasses"]
