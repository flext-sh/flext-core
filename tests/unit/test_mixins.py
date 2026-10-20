"""Behavioral tests for FlextMixins service infrastructure.

Module: flext_core
Scope: FlextMixins public surface (settings/container/context/logger
properties, ``track`` operation context manager, container registration,
runtime bootstrap setters, FlextContext correlation).

Every test asserts OBSERVABLE public behavior of the mixin contract -- the
value a service author actually depends on -- never private internals or
implementation collaborators.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from flext_core import FlextContext
from tests import c, m, p, tm, x

if TYPE_CHECKING:
    from collections.abc import Mapping

    from tests import t


class TestsFlextMixins:
    """Behavioral suite for the FlextMixins public contract."""

    class _Service(x):
        """Minimal service composing FlextMixins for behavioral exercise."""

    def _service(self) -> x:
        """Build a fresh service instance exposing the mixin surface.

        Returns:
            The resulting ``x``.

        """
        return self._Service()

    def test_settings_property_satisfies_settings_protocol(self) -> None:
        """Test settings property satisfies settings protocol."""
        service = self._service()

        tm.that(
            {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"},
            has=service.settings.model_dump()["log_level"],
        )

    def test_container_property_satisfies_container_protocol(self) -> None:
        """Test container property satisfies container protocol."""
        service = self._service()

        tm.that(isinstance(service.container, p.Container), truth=True)

    def test_context_property_satisfies_context_protocol(self) -> None:
        """Test context property satisfies context protocol."""
        service = self._service()

        tm.that(isinstance(service.context, p.Context), truth=True)

    def test_logger_property_satisfies_logger_protocol(self) -> None:
        """Test logger property satisfies logger protocol."""
        service = self._service()

        tm.that(isinstance(service.logger, p.Logger), truth=True)

    def test_track_yields_metrics_mapping_with_operation_metadata(self) -> None:
        """Test track yields metrics mapping with operation metadata."""
        service = self._service()

        with service.track("load") as metrics:
            captured: Mapping[str, t.JsonValue] = metrics

        tm.that("operation_name" in captured, truth=True)
        tm.that("start_time" in captured, truth=True)
        tm.that(captured["operation_count"], eq=1)

    def test_track_increments_operation_count_across_invocations(self) -> None:
        """Test track increments operation count across invocations."""
        service = self._service()

        with service.track("load") as first:
            first_count = first["operation_count"]
        with service.track("load") as second:
            second_count = second["operation_count"]

        tm.that(first_count, eq=1)
        tm.that(second_count, eq=2)

    def test_track_returns_body_value_on_success(self) -> None:
        """Test track returns body value on success."""
        service = self._service()

        def run() -> str:
            with service.track("compute"):
                return "done"

        tm.that(run(), eq="done")

    def test_track_propagates_exception_raised_in_body(self) -> None:
        """Test track propagates exception raised in body."""
        service = self._service()
        boom = ValueError("boom")

        with pytest.raises(ValueError, match="boom"), service.track("fail"):
            raise boom

    def test_track_recovers_after_failed_operation(self) -> None:
        """Test track recovers after failed operation."""
        service = self._service()
        boom = ValueError("boom")

        with pytest.raises(ValueError, match="boom"), service.track("op"):
            raise boom
        with service.track("op") as metrics:
            recovered = metrics["operation_count"]

        tm.that(recovered, eq=2)

    def test_initial_context_defaults_to_none(self) -> None:
        """Test initial context defaults to none."""
        service = self._service()

        tm.that(service.initial_context, none=True)

    @staticmethod
    def test_contract_model_composition_preserves_validation() -> None:
        """A service composes immutable options without inheriting their model."""

        class Options(m.ContractModel):
            label: str

        class ComposedService(x):
            options: Options = m.Field(
                description="Validated immutable service options",
            )

        label = "  census-options  "
        service = ComposedService(options=Options(label=label))
        expected = (
            label.strip() if Options.model_config.get("str_strip_whitespace") else label
        )
        tm.that(service.options.label, eq=expected)
        tm.that(service.model_dump()["options"], eq=service.options.model_dump())
        tm.that(isinstance(service.settings, p.Settings), truth=True)
        with pytest.raises(c.ValidationError):
            ComposedService.model_validate({"options": {"label": []}})
        tm.rejects_assignment(
            service.options,
            "label",
            "changed",
            expected=c.ValidationError,
        )

    def test_settings_overrides_setter_round_trips(self) -> None:
        """Test settings overrides setter round trips."""
        service = self._service()
        overrides: t.MappingKV[str, t.JsonValue] = {"feature_flag": True, "retries": 3}

        service.settings_overrides = overrides

        tm.that(service.settings_overrides, eq=overrides)

    @pytest.mark.parametrize(
        ("field_name", "bad_value"),
        [("settings_overrides", "not-a-mapping"), ("settings_type", str)],
    )
    def test_constructor_rejects_invalid_bootstrap_value(
        self,
        field_name: str,
        bad_value: p.GuardInput,
    ) -> None:
        """Test constructor rejects invalid bootstrap value."""
        with pytest.raises(c.ValidationError):
            self._Service.model_validate({field_name: bad_value})

    @staticmethod
    def test_correlation_id_round_trips_through_flext_context() -> None:
        """Test correlation id round trips through flext context."""
        FlextContext.apply_correlation_id("trace-42")

        tm.that(FlextContext.resolve_correlation_id(), eq="trace-42")
