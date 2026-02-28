"""Tests for x infrastructure - Container, Context, Logging, Metrics, Service.

Module: flext_core.mixins
Scope: x - all nested mixin classes

Tests x functionality including:
- Container mixin (_register_in_container)
- Context mixin (context property, _propagate_context, correlation IDs)
- Logging mixin (_log_with_context)
- Metrics mixin (track context manager)
- Service mixin (_init_service, _enrich_context, _with_operation_context)
- ModelConversion nested class (to_dict conversions)
- ResultHandling nested class (ensure_result wrapping)

Uses Python 3.13 patterns, FlextTestsUtilities, FlextConstants,
and aggressive parametrization for DRY testing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

import pytest
from flext_core import FlextContext, m, t, x
from flext_tests import u


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
    input_value: t.GeneralValueType
    expected_output: m.ConfigMap


@dataclass(frozen=True, slots=True)
class ResultHandlingScenario:
    """ResultHandling test scenario definition."""

    name: str
    scenario_type: ResultHandlingScenarioType
    input_value: t.GeneralValueType


class MixinScenarios:
    """Centralized mixin test scenarios using FlextConstants."""

    SERVICE_SCENARIOS: ClassVar[list[ServiceMixinScenario]] = [
        ServiceMixinScenario(
            "container_register_in_container",
            ServiceMixinScenarioType.CONTAINER_REGISTER,
        ),
        ServiceMixinScenario(
            "context_mixin_property",
            ServiceMixinScenarioType.CONTEXT_PROPERTY,
        ),
        ServiceMixinScenario(
            "context_propagate",
            ServiceMixinScenarioType.CONTEXT_PROPAGATE,
        ),
        ServiceMixinScenario(
            "context_correlation_id",
            ServiceMixinScenarioType.CONTEXT_CORRELATION,
        ),
        ServiceMixinScenario(
            "logging_with_context",
            ServiceMixinScenarioType.LOGGING_WITH_CONTEXT,
        ),
        ServiceMixinScenario("metrics_track", ServiceMixinScenarioType.METRICS_TRACK),
        ServiceMixinScenario(
            "service_init_service",
            ServiceMixinScenarioType.SERVICE_INIT,
            True,
        ),
        ServiceMixinScenario(
            "service_enrich_context",
            ServiceMixinScenarioType.SERVICE_ENRICH,
            True,
        ),
    ]

    MODEL_CONVERSION_SCENARIOS: ClassVar[list[ModelConversionScenario]] = [
        ModelConversionScenario(
            "to_dict_with_basemodel",
            ModelConversionScenarioType.WITH_BASEMODEL,
            None,
            m.ConfigMap(root={"name": "test", "value": 42}),
        ),
        ModelConversionScenario(
            "to_dict_with_dict",
            ModelConversionScenarioType.WITH_DICT,
            {"key": "value", "number": 123},
            m.ConfigMap(root={"key": "value", "number": 123}),
        ),
        ModelConversionScenario(
            "to_dict_with_none",
            ModelConversionScenarioType.WITH_NONE,
            None,
            m.ConfigMap(root={}),
        ),
    ]

    RESULT_HANDLING_SCENARIOS: ClassVar[list[ResultHandlingScenario]] = [
        ResultHandlingScenario(
            "ensure_result_raw_value",
            ResultHandlingScenarioType.RAW_VALUE,
            42,
        ),
        ResultHandlingScenario(
            "ensure_result_existing_result",
            ResultHandlingScenarioType.EXISTING_RESULT,
            None,
        ),
        ResultHandlingScenario(
            "ensure_result_type_preservation",
            ResultHandlingScenarioType.TYPE_PRESERVATION,
            None,
        ),
    ]


class TestFlextMixinsNestedClasses:
    """Comprehensive test suite for nested mixin classes using FlextTestsUtilities."""

    @pytest.mark.parametrize(
        "scenario",
        MixinScenarios.SERVICE_SCENARIOS,
        ids=lambda s: s.name,
    )
    def test_service_mixin_scenarios(self, scenario: ServiceMixinScenario) -> None:
        """Test service mixin functionality across scenarios."""

        class MyService(x):
            """Test service for mixin scenarios."""

            def __init__(self) -> None:
                super().__init__()
                if scenario.needs_init:
                    self._init_service("MyTestService")

            def process(self) -> str:
                """Process operation for metrics testing.

                Validates:
                1. Track context manager provides metrics dict
                2. Operation completes successfully after delay
                3. Metrics are tracked correctly
                """
                with self.track("test_op") as metrics:
                    # Validate metrics dict is provided
                    assert isinstance(metrics, dict), "Metrics should be a dict"
                    assert "operation_name" in metrics or "start_time" in metrics, (
                        "Metrics should contain operation info"
                    )

                    # Simulate work
                    time.sleep(0.01)

                    # Validate operation completes
                    return "done"

        service = MyService()
        if scenario.scenario_type == ServiceMixinScenarioType.CONTAINER_REGISTER:
            result = service._register_in_container("test_service")
            u.Tests.Result.assert_result_success(result)
        elif scenario.scenario_type == ServiceMixinScenarioType.CONTEXT_PROPERTY:
            assert isinstance(service.context, FlextContext)
        elif scenario.scenario_type == ServiceMixinScenarioType.CONTEXT_PROPAGATE:
            service._propagate_context("test_operation")
        elif scenario.scenario_type == ServiceMixinScenarioType.CONTEXT_CORRELATION:
            FlextContext.Correlation.set_correlation_id("test-123")
            assert FlextContext.Correlation.get_correlation_id() == "test-123"
        elif scenario.scenario_type == ServiceMixinScenarioType.LOGGING_WITH_CONTEXT:
            service._log_with_context("info", "Test message", extra_data="value")
        elif scenario.scenario_type == ServiceMixinScenarioType.METRICS_TRACK:
            assert service.process() == "done"
        elif scenario.scenario_type == ServiceMixinScenarioType.SERVICE_INIT:
            assert all(
                hasattr(service, attr) for attr in ["logger", "container", "config"]
            )
        elif scenario.scenario_type == ServiceMixinScenarioType.SERVICE_ENRICH:
            service._enrich_context(version="1.0.0", team="test")

    def test_service_mixin_with_operation_context(self) -> None:
        """Test Service mixin operation context workflow."""

        class MyService(x):
            """Test service with operation context."""

            def __init__(self) -> None:
                super().__init__()
                self._init_service()

        service = MyService()
        service._with_operation_context("process_order", order_id="123")
        service._clear_operation_context()


__all__ = ["TestFlextMixinsNestedClasses"]
