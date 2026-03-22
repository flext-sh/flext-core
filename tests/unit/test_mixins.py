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

Uses Python 3.13 patterns, u, FlextConstants,
and aggressive parametrization for DRY testing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time
from enum import StrEnum, unique
from typing import Annotated, ClassVar

from flext_tests import u
from pydantic import BaseModel, ConfigDict, Field

from flext_core import FlextContext, p, t, x


class TestFlextMixinsNestedClasses:
    """Comprehensive test suite for nested mixin classes using u."""

    @unique
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

    @unique
    class ModelConversionScenarioType(StrEnum):
        """ModelConversion test scenario types."""

        WITH_BASEMODEL = "with_basemodel"
        WITH_DICT = "with_dict"
        WITH_NONE = "with_none"

    @unique
    class ResultHandlingScenarioType(StrEnum):
        """ResultHandling test scenario types."""

        RAW_VALUE = "raw_value"
        EXISTING_RESULT = "existing_result"
        TYPE_PRESERVATION = "type_preservation"

    class ServiceMixinScenario(BaseModel):
        """Service mixin test scenario definition."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
        name: Annotated[str, Field(description="Service mixin scenario name")]
        scenario_type: Annotated[
            TestFlextMixinsNestedClasses.ServiceMixinScenarioType,
            Field(description="Service mixin scenario type"),
        ]
        needs_init: Annotated[
            bool,
            Field(
                default=False,
                description="Whether service initialization is required",
            ),
        ] = False
        operation_context: Annotated[
            str | None,
            Field(default=None, description="Optional operation context name"),
        ] = None

    class ModelConversionScenario(BaseModel):
        """ModelConversion test scenario definition."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
        name: Annotated[str, Field(description="Model conversion scenario name")]
        scenario_type: Annotated[
            TestFlextMixinsNestedClasses.ModelConversionScenarioType,
            Field(description="Model conversion scenario type"),
        ]
        input_value: Annotated[
            t.NormalizedValue, Field(description="Input value for conversion")
        ]
        expected_output: Annotated[
            t.ConfigMap,
            Field(description="Expected conversion output"),
        ]

    class ResultHandlingScenario(BaseModel):
        """ResultHandling test scenario definition."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
        name: Annotated[str, Field(description="Result handling scenario name")]
        scenario_type: Annotated[
            TestFlextMixinsNestedClasses.ResultHandlingScenarioType,
            Field(description="Result handling scenario type"),
        ]
        input_value: Annotated[
            t.NormalizedValue,
            Field(description="Input value for result handling"),
        ]

    def _service_scenarios(
        self,
    ) -> list[TestFlextMixinsNestedClasses.ServiceMixinScenario]:
        return [
            self.ServiceMixinScenario(
                name="container_register_in_container",
                scenario_type=self.ServiceMixinScenarioType.CONTAINER_REGISTER,
            ),
            self.ServiceMixinScenario(
                name="context_mixin_property",
                scenario_type=self.ServiceMixinScenarioType.CONTEXT_PROPERTY,
            ),
            self.ServiceMixinScenario(
                name="context_propagate",
                scenario_type=self.ServiceMixinScenarioType.CONTEXT_PROPAGATE,
            ),
            self.ServiceMixinScenario(
                name="context_correlation_id",
                scenario_type=self.ServiceMixinScenarioType.CONTEXT_CORRELATION,
            ),
            self.ServiceMixinScenario(
                name="logging_with_context",
                scenario_type=self.ServiceMixinScenarioType.LOGGING_WITH_CONTEXT,
            ),
            self.ServiceMixinScenario(
                name="metrics_track",
                scenario_type=self.ServiceMixinScenarioType.METRICS_TRACK,
            ),
            self.ServiceMixinScenario(
                name="service_init_service",
                scenario_type=self.ServiceMixinScenarioType.SERVICE_INIT,
                needs_init=True,
            ),
            self.ServiceMixinScenario(
                name="service_enrich_context",
                scenario_type=self.ServiceMixinScenarioType.SERVICE_ENRICH,
                needs_init=True,
            ),
        ]

    def test_service_mixin_scenarios(self) -> None:
        """Test service mixin functionality across scenarios."""
        for scenario in self._service_scenarios():
            self._assert_service_mixin_scenario(scenario)

    def _assert_service_mixin_scenario(
        self,
        scenario: TestFlextMixinsNestedClasses.ServiceMixinScenario,
    ) -> None:

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
        if scenario.scenario_type == self.ServiceMixinScenarioType.CONTAINER_REGISTER:
            result = service._register_in_container("test_service")
            _ = u.Tests.Result.assert_success(result)
        elif scenario.scenario_type == self.ServiceMixinScenarioType.CONTEXT_PROPERTY:
            assert isinstance(service.context, p.Context)
        elif scenario.scenario_type == self.ServiceMixinScenarioType.CONTEXT_PROPAGATE:
            service._propagate_context("test_operation")
        elif (
            scenario.scenario_type == self.ServiceMixinScenarioType.CONTEXT_CORRELATION
        ):
            FlextContext.Correlation.set_correlation_id("test-123")
            assert FlextContext.Correlation.get_correlation_id() == "test-123"
        elif (
            scenario.scenario_type == self.ServiceMixinScenarioType.LOGGING_WITH_CONTEXT
        ):
            service._log_with_context("info", "Test message", extra_data="value")
        elif scenario.scenario_type == self.ServiceMixinScenarioType.METRICS_TRACK:
            assert service.process() == "done"
        elif scenario.scenario_type == self.ServiceMixinScenarioType.SERVICE_INIT:
            assert all(
                hasattr(service, attr) for attr in ["logger", "container", "config"]
            )
        elif scenario.scenario_type == self.ServiceMixinScenarioType.SERVICE_ENRICH:
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
