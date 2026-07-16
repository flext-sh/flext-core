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
from flext_tests import x

from flext_core import FlextContext
from tests.protocols import p
from tests.utilities import u

if TYPE_CHECKING:
    from collections.abc import Mapping

    from tests.typings import p, t


class TestsFlextMixins:
    """Behavioral suite for the FlextMixins public contract."""

    class _Service(x):
        """Minimal service composing FlextMixins for behavioral exercise."""

    def _service(self) -> x:
        """Build a fresh service instance exposing the mixin surface."""
        return self._Service()

    def test_settings_property_satisfies_settings_protocol(self) -> None:
        service = self._service()

        assert service.settings.model_dump()["log_level"] in {
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
            "CRITICAL",
        }

    def test_container_property_satisfies_container_protocol(self) -> None:
        service = self._service()

        assert isinstance(service.container, p.Container)

    def test_context_property_satisfies_context_protocol(self) -> None:
        service = self._service()

        assert isinstance(service.context, p.Context)

    def test_logger_property_satisfies_logger_protocol(self) -> None:
        service = self._service()

        assert isinstance(service.logger, p.Logger)

    def test_track_yields_metrics_mapping_with_operation_metadata(self) -> None:
        service = self._service()

        with service.track("load") as metrics:
            captured: Mapping[str, t.JsonPayload] = metrics

        assert "operation_name" in captured
        assert "start_time" in captured
        assert captured["operation_count"] == 1

    def test_track_increments_operation_count_across_invocations(self) -> None:
        service = self._service()

        with service.track("load") as first:
            first_count = first["operation_count"]
        with service.track("load") as second:
            second_count = second["operation_count"]

        assert first_count == 1
        assert second_count == 2

    def test_track_returns_body_value_on_success(self) -> None:
        service = self._service()

        def run() -> str:
            with service.track("compute"):
                return "done"

        assert run() == "done"

    def test_track_propagates_exception_raised_in_body(self) -> None:
        service = self._service()
        boom = ValueError("boom")

        with pytest.raises(ValueError, match="boom"), service.track("fail"):
            raise boom

    def test_track_recovers_after_failed_operation(self) -> None:
        service = self._service()
        boom = ValueError("boom")

        with pytest.raises(ValueError, match="boom"), service.track("op"):
            raise boom
        with service.track("op") as metrics:
            recovered = metrics["operation_count"]

        assert recovered == 2

    def test_register_in_container_succeeds_and_reports_true(self) -> None:
        service = self._service()

        result = service._register_in_container("discovery_target")
        registered = u.Tests.assert_success(result)

        assert registered is True

    def test_register_in_container_is_idempotent(self) -> None:
        service = self._service()

        _ = u.Tests.assert_success(service._register_in_container("target"))
        second = u.Tests.assert_success(service._register_in_container("target"))

        assert second is True
        assert service.container.has("target") is True

    def test_init_service_makes_service_discoverable_in_container(self) -> None:
        service = self._service()

        assert service.container.has("DiscoverableService") is False
        service._init_service("DiscoverableService")

        assert service.container.has("DiscoverableService") is True

    def test_init_service_defaults_registration_name_to_class(self) -> None:
        service = self._service()

        service._init_service()

        assert service.container.has(type(service).__name__) is True

    def test_initial_context_defaults_to_none(self) -> None:
        service = self._service()

        assert service.initial_context is None

    def test_settings_overrides_setter_round_trips(self) -> None:
        service = self._service()
        overrides: t.JsonMapping = {"feature_flag": True, "retries": 3}

        service.settings_overrides = overrides

        assert service.settings_overrides == overrides

    @pytest.mark.parametrize(
        ("field_name", "bad_value"),
        [
            ("settings_overrides", "not-a-mapping"),
            ("runtime_settings", 123),
            ("initial_context", "not-a-context"),
            ("settings_type", str),
        ],
    )
    def test_constructor_rejects_invalid_bootstrap_value(
        self,
        field_name: str,
        bad_value: t.GuardInput,
    ) -> None:
        invalid_bootstrap: dict[str, t.GuardInput] = {field_name: bad_value}
        with pytest.raises(TypeError):
            self._Service(**invalid_bootstrap)

    def test_correlation_id_round_trips_through_flext_context(self) -> None:
        FlextContext.apply_correlation_id("trace-42")

        assert FlextContext.resolve_correlation_id() == "trace-42"


__all__: t.MutableSequenceOf[str] = ["TestsFlextMixins"]
