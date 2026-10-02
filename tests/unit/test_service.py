"""Behavioral tests for the slim ``FlextService`` public contract.

Every assertion targets observable public behavior: the ``r[T]`` outcome of
``execute``, the shared-runtime singleton returned by ``fetch_global``, the
public ``settings``/``logger`` accessors, the ``with_settings`` snapshot, the
``isolated_test_runtime`` isolation invariant, the ``track`` context-manager
contract, and the public state of the ``ServiceUserData`` result model. No
private attributes, internal collaborators, or implementation details are
inspected.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import operator
from collections.abc import Mapping
from typing import override

import pytest
from flext_tests import FlextTestsCase, FlextTestsSettings, r

from tests.base import s
from tests.models import m
from tests.protocols import p


class TestsFlextService(FlextTestsCase):
    """Validate stable, caller-facing behavior of ``FlextService`` subclasses."""

    class _PureService(s[bool]):
        """Minimal service whose ``execute`` always succeeds."""

        @override
        def execute(self) -> p.Result[bool]:
            return r[bool].ok(value=True)

    class _FailingService(s[bool]):
        """Minimal service whose ``execute`` always fails with a known error."""

        @override
        def execute(self) -> p.Result[bool]:
            return r[bool].fail("execute-boom")

    # --- execute(): the r[T] contract ------------------------------------

    @staticmethod
    def test_execute_reports_success_and_typed_payload() -> None:
        """Test execute reports success and typed payload."""
        service = m.Tests.ServiceUserService()

        result = service.execute()

        assert result.success
        assert not result.failure
        assert result.unwrap() == m.Tests.ServiceUserData(user_id=1, name="test_user")

    @staticmethod
    def test_execute_success_value_exposes_public_model_fields() -> None:
        """Test execute success value exposes public model fields."""
        result = m.Tests.ServiceUserService().execute()

        payload = result.value
        assert isinstance(payload, m.Tests.ServiceUserData)
        assert payload.user_id == 1
        assert payload.name == "test_user"

    def test_execute_failure_propagates_error_through_result(self) -> None:
        """Test execute failure propagates error through result."""
        result = self._FailingService().execute()

        assert result.failure
        assert not result.success
        assert result.error == "execute-boom"

    def test_execute_success_result_supports_combinators(self) -> None:
        """Test execute success result supports combinators."""
        result = self._PureService().execute()

        assert result.map(lambda ok: ok and True).unwrap() is True
        assert result.flat_map(lambda ok: r[bool].ok(value=not ok)).unwrap() is False

    def test_execute_failure_result_short_circuits_combinators(self) -> None:
        """Test execute failure result short circuits combinators."""
        result = self._FailingService().execute()

        assert result.map(operator.not_).failure
        assert result.unwrap_or(default=False) is False

    @staticmethod
    def _satisfies_service_protocol(candidate: p.Base) -> bool:
        """Report structural conformance without a type-narrowed argument.

        Returns:
            The resulting ``bool``.

        """
        return isinstance(candidate, p.Service)

    def test_real_service_satisfies_the_service_protocol(self) -> None:
        """Every real service satisfies p.Service structurally (S4 contract)."""
        assert self._satisfies_service_protocol(m.Tests.ServiceUserService())
        assert self._satisfies_service_protocol(self._PureService())

    def test_model_without_service_runtime_is_not_a_service(self) -> None:
        """A plain model lacks the runtime surface and execute: not a service."""
        data = m.Tests.ServiceUserData(user_id=1, name="test_user")

        assert not self._satisfies_service_protocol(data)

    # --- ServiceUserData: public model state -----------------------------

    @pytest.mark.parametrize(
        ("user_id", "name"),
        [(1, "test_user"), (2, "other"), (99, "édge-café")],
    )
    def test_service_user_data_round_trips_public_state(
        self,
        user_id: int,
        name: str,
    ) -> None:
        """Test service user data round trips public state."""
        data = m.Tests.ServiceUserData(user_id=user_id, name=name)

        assert data.model_dump() == {"user_id": user_id, "name": name}
        assert data == m.Tests.ServiceUserData(user_id=user_id, name=name)

    # --- fetch_global(): shared-runtime singleton ------------------------

    def test_fetch_global_returns_the_shared_singleton(self) -> None:
        """Test fetch global returns the shared singleton."""
        first = type(self.service).fetch_global()
        second = type(self.service).fetch_global()

        assert first is self.service
        assert second is first

    # --- settings / logger public accessors ------------------------------

    @staticmethod
    def test_settings_expose_tests_namespace() -> None:
        """Test settings expose tests namespace."""
        settings = m.Tests.ServiceUserService().settings

        assert isinstance(settings, FlextTestsSettings)
        assert isinstance(settings.Tests, m.BaseModel)

    def test_fetch_settings_returns_typed_tests_settings(self) -> None:
        """Test fetch settings returns typed tests settings."""
        with self._PureService.isolated_test_runtime():
            settings = self._PureService.fetch_settings()

            assert isinstance(settings, FlextTestsSettings)
            assert isinstance(settings.Tests, m.BaseModel)

    def test_fetch_logger_matches_shared_service_logger(self) -> None:
        """Test fetch logger matches shared service logger."""
        with self._PureService.isolated_test_runtime():
            assert (
                self._PureService.fetch_logger()
                is self._PureService.fetch_global().logger
            )

    # --- with_settings(): runtime snapshot -------------------------------

    def test_with_settings_applies_provided_snapshot(self) -> None:
        """Test with settings applies provided snapshot."""
        settings = m.Tests.ServiceUserService().settings.clone(log_level="ERROR")

        service = self._PureService.with_settings(settings)

        dumped = service.settings.model_dump()
        assert dumped == settings.model_dump()
        assert dumped["log_level"] == "ERROR"

    # --- isolated_test_runtime(): isolation invariant --------------------

    def test_isolated_runtime_scopes_settings_without_leaking(self) -> None:
        """Test isolated runtime scopes settings without leaking."""
        baseline_level = self._PureService.fetch_settings().log_level

        with self._PureService.isolated_test_runtime(
            log_level="WARNING",
        ) as scoped_service:
            scoped_settings = FlextTestsSettings.model_validate(scoped_service.settings)

            assert scoped_settings.log_level == "WARNING"
            scoped_global = FlextTestsSettings.fetch_global()
            assert scoped_global.log_level != "WARNING"

        assert self._PureService.fetch_settings().log_level == baseline_level

    # --- track(): context-manager metrics contract -----------------------

    @staticmethod
    def test_track_yields_named_operation_metrics() -> None:
        """Test track yields named operation metrics."""
        service = m.Tests.ServiceUserService()

        with service.track("load_users") as metrics:
            assert isinstance(metrics, Mapping)
            assert metrics["operation_name"] == "load_users"
