"""Tests for x infrastructure - Container, Context, Logging, Metrics, Service.

Module: flext_core
Scope: x - all nested mixin classes

Tests x functionality including:
- Container mixin (_register_in_container)
- Context mixin (context property, correlation IDs)
- Metrics mixin (track context manager)
- Service mixin (_init_service)

Uses Python 3.13 patterns, u, FlextConstants,
and aggressive parametrization for DRY testing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time
from enum import StrEnum, unique
from typing import Annotated, ClassVar

from flext_core import FlextContext
from tests import m, p, t, u, x


class TestsFlextMixins:
    """Comprehensive test suite for nested mixin classes using u."""

    @unique
    class ServiceMixinScenarioType(StrEnum):
        """Service mixin test scenario types."""

        CONTAINER_REGISTER = "container_register"
        CONTEXT_PROPERTY = "context_property"
        CONTEXT_CORRELATION = "context_correlation"
        METRICS_TRACK = "metrics_track"
        SERVICE_INIT = "service_init"

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

    class ServiceMixinScenario(m.BaseModel):
        """Service mixin test scenario definition."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)
        name: Annotated[str, m.Field(description="Service mixin scenario name")]
        scenario_type: Annotated[
            TestsFlextMixins.ServiceMixinScenarioType,
            m.Field(description="Service mixin scenario type"),
        ]
        needs_init: Annotated[
            bool,
            m.Field(
                description="Whether service initialization is required",
            ),
        ] = False
        operation_context: Annotated[
            str | None, m.Field(description="Optional operation context name")
        ] = None

    class ModelConversionScenario(m.BaseModel):
        """ModelConversion test scenario definition."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)
        name: Annotated[str, m.Field(description="Model conversion scenario name")]
        scenario_type: Annotated[
            TestsFlextMixins.ModelConversionScenarioType,
            m.Field(description="Model conversion scenario type"),
        ]
        input_value: Annotated[
            t.JsonValue,
            m.Field(description="Input value for conversion"),
        ]
        expected_output: Annotated[
            m.ConfigMap,
            m.Field(description="Expected conversion output"),
        ]

    class ResultHandlingScenario(m.BaseModel):
        """ResultHandling test scenario definition."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)
        name: Annotated[str, m.Field(description="Result handling scenario name")]
        scenario_type: Annotated[
            TestsFlextMixins.ResultHandlingScenarioType,
            m.Field(description="Result handling scenario type"),
        ]
        input_value: Annotated[
            t.JsonValue,
            m.Field(description="Input value for result handling"),
        ]

    def _service_scenarios(
        self,
    ) -> t.SequenceOf[TestsFlextMixins.ServiceMixinScenario]:
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
                name="context_correlation_id",
                scenario_type=self.ServiceMixinScenarioType.CONTEXT_CORRELATION,
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
        ]

    def test_service_mixin_scenarios(self) -> None:
        """Test service mixin functionality across scenarios."""
        for scenario in self._service_scenarios():
            self._assert_service_mixin_scenario(scenario)

    def _assert_service_mixin_scenario(
        self,
        scenario: TestsFlextMixins.ServiceMixinScenario,
    ) -> None:

        class MyService(x):
            """Test service for mixin scenarios."""

            def __init__(self) -> None:
                super().__init__(
                    settings_type=None,
                    settings_overrides=None,
                    initial_context=None,
                )
                if scenario.needs_init:
                    self._init_service("MyTestService")

            def run_process(self) -> str:
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
            _ = u.Tests.assert_success(result)
        elif scenario.scenario_type == self.ServiceMixinScenarioType.CONTEXT_PROPERTY:
            assert isinstance(service.context, p.Context)
        elif (
            scenario.scenario_type == self.ServiceMixinScenarioType.CONTEXT_CORRELATION
        ):
            FlextContext.apply_correlation_id("test-123")
            assert FlextContext.resolve_correlation_id() == "test-123"
        elif scenario.scenario_type == self.ServiceMixinScenarioType.METRICS_TRACK:
            assert service.run_process() == "done"
        elif scenario.scenario_type == self.ServiceMixinScenarioType.SERVICE_INIT:
            assert all(
                hasattr(service, attr) for attr in ["logger", "container", "settings"]
            )


__all__: t.MutableSequenceOf[str] = ["TestsFlextMixins"]
