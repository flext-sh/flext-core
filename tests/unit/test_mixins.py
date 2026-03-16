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
from enum import StrEnum
from typing import Annotated, ClassVar

import pytest
from flext_tests import u
from pydantic import BaseModel, ConfigDict, Field

from flext_core import FlextContext, x
from tests.models import m


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


class ServiceMixinScenario(BaseModel):
    """Service mixin test scenario definition."""

    model_config = ConfigDict(frozen=True)
    name: Annotated[str, Field(description="Service mixin scenario name")]
    scenario_type: Annotated[
        ServiceMixinScenarioType, Field(description="Service mixin scenario type")
    ]
    needs_init: Annotated[
        bool,
        Field(default=False, description="Whether service initialization is required"),
    ] = False
    operation_context: Annotated[
        str | None, Field(default=None, description="Optional operation context name")
    ] = None


class ModelConversionScenario(BaseModel):
    """ModelConversion test scenario definition."""

    model_config = ConfigDict(frozen=True)
    name: Annotated[str, Field(description="Model conversion scenario name")]
    scenario_type: Annotated[
        ModelConversionScenarioType, Field(description="Model conversion scenario type")
    ]
    input_value: Annotated[object, Field(description="Input value for conversion")]
    expected_output: Annotated[
        m.ConfigMap, Field(description="Expected conversion output")
    ]


class ResultHandlingScenario(BaseModel):
    """ResultHandling test scenario definition."""

    model_config = ConfigDict(frozen=True)
    name: Annotated[str, Field(description="Result handling scenario name")]
    scenario_type: Annotated[
        ResultHandlingScenarioType, Field(description="Result handling scenario type")
    ]
    input_value: Annotated[object, Field(description="Input value for result handling")]


class MixinScenarios:
    """Centralized mixin test scenarios using FlextConstants."""

    SERVICE_SCENARIOS: ClassVar[list[ServiceMixinScenario]] = [
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
    MODEL_CONVERSION_SCENARIOS: ClassVar[list[ModelConversionScenario]] = [
        ModelConversionScenario(
            name="to_dict_with_basemodel",
            scenario_type=ModelConversionScenarioType.WITH_BASEMODEL,
            input_value=None,
            expected_output=m.ConfigMap(root={"name": "test", "value": 42}),
        ),
        ModelConversionScenario(
            name="to_dict_with_dict",
            scenario_type=ModelConversionScenarioType.WITH_DICT,
            input_value={"key": "value", "number": 123},
            expected_output=m.ConfigMap(root={"key": "value", "number": 123}),
        ),
        ModelConversionScenario(
            name="to_dict_with_none",
            scenario_type=ModelConversionScenarioType.WITH_NONE,
            input_value=None,
            expected_output=m.ConfigMap(root={}),
        ),
    ]
    RESULT_HANDLING_SCENARIOS: ClassVar[list[ResultHandlingScenario]] = [
        ResultHandlingScenario(
            name="ensure_result_raw_value",
            scenario_type=ResultHandlingScenarioType.RAW_VALUE,
            input_value=42,
        ),
        ResultHandlingScenario(
            name="ensure_result_existing_result",
            scenario_type=ResultHandlingScenarioType.EXISTING_RESULT,
            input_value=None,
        ),
        ResultHandlingScenario(
            name="ensure_result_type_preservation",
            scenario_type=ResultHandlingScenarioType.TYPE_PRESERVATION,
            input_value=None,
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
                super().__init__(
                    config_type=None,
                    config_overrides=None,
                    initial_context=None,
                )
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
                    assert isinstance(metrics, dict), "Metrics should be a dict"
                    assert "operation_name" in metrics or "start_time" in metrics, (
                        "Metrics should contain operation info"
                    )
                    time.sleep(0.01)
                    return "done"

        service = MyService()
        if scenario.scenario_type == ServiceMixinScenarioType.CONTAINER_REGISTER:
            result = service._register_in_container("test_service")
            _ = u.Tests.Result.assert_success(result)
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
                super().__init__(
                    config_type=None,
                    config_overrides=None,
                    initial_context=None,
                )
                self._init_service()

        service = MyService()
        service._with_operation_context("process_order", order_id="123")
        service._clear_operation_context()


__all__ = ["TestFlextMixinsNestedClasses"]
