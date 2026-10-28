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

from collections.abc import (
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
)

import operator
from collections.abc import Mapping
from typing import override

import pytest
from flext_tests import FlextTestsCase, FlextTestsSettings

from tests import c, m, p, r, s, tm


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

        tm.that(result.success, truth=True)
        tm.that(result.failure, truth=False)
        tm.that(
            result.unwrap(), eq=m.Tests.ServiceUserData(user_id=1, name="test_user")
        )

    @staticmethod
    def test_execute_success_value_exposes_public_model_fields() -> None:
        """Test execute success value exposes public model fields."""
        result = m.Tests.ServiceUserService().execute()

        payload = result.value
        tm.that(payload, is_=m.Tests.ServiceUserData)
        tm.that(payload.user_id, eq=1)
        tm.that(payload.name, eq="test_user")

    def test_execute_failure_propagates_error_through_result(self) -> None:
        """Test execute failure propagates error through result."""
        result = self._FailingService().execute()

        tm.that(result.failure, truth=True)
        tm.that(result.success, truth=False)
        tm.that(result.error, eq="execute-boom")

    def test_execute_success_result_supports_combinators(self) -> None:
        """Test execute success result supports combinators."""
        result = self._PureService().execute()

        tm.that(result.map(lambda ok: ok and True).unwrap() is True, truth=True)
        tm.that(
            result.flat_map(lambda ok: r[bool].ok(value=not ok)).unwrap() is False,
            truth=True,
        )

    def test_execute_failure_result_short_circuits_combinators(self) -> None:
        """Test execute failure result short circuits combinators."""
        result = self._FailingService().execute()

        tm.that(result.map(operator.not_).failure, truth=True)
        tm.that(result.unwrap_or(default=False) is False, truth=True)

    @staticmethod
    def _satisfies_service_protocol(candidate: p.Base) -> bool:
        """Report structural conformance without a type-narrowed argument.

        Returns:
            The resulting ``bool``.

        """
        return isinstance(candidate, p.Service)

    def test_real_service_satisfies_the_service_protocol(self) -> None:
        """Every real service satisfies p.Service structurally (S4 contract)."""
        tm.that(
            self._satisfies_service_protocol(m.Tests.ServiceUserService()), truth=True
        )
        tm.that(self._satisfies_service_protocol(self._PureService()), truth=True)

    def test_model_without_service_runtime_is_not_a_service(self) -> None:
        """A plain model lacks the runtime surface and execute: not a service."""
        data = m.Tests.ServiceUserData(user_id=1, name="test_user")

        tm.that(self._satisfies_service_protocol(data), truth=False)

    # --- ServiceUserData: public model state -----------------------------

    @staticmethod
    @pytest.mark.parametrize(
        (c.ContextKey.USER_ID, "name"),
        [(1, "test_user"), (2, "other"), (99, "édge-café")],
    )
    def test_service_user_data_round_trips_public_state(
        user_id: int,
        name: str,
    ) -> None:
        """Test service user data round trips public state."""
        data = m.Tests.ServiceUserData(user_id=user_id, name=name)

        tm.that(data.model_dump(), eq={c.ContextKey.USER_ID: user_id, "name": name})
        tm.that(data, eq=m.Tests.ServiceUserData(user_id=user_id, name=name))

    # --- fetch_global(): shared-runtime singleton ------------------------

    def test_fetch_global_returns_the_shared_singleton(self) -> None:
        """Test fetch global returns the shared singleton."""
        with self._PureService.isolated_test_runtime() as service:
            first = self._PureService.fetch_global()
            second = self._PureService.fetch_global()

            tm.that(first is service, truth=True)
            tm.that(second is first, truth=True)

    # --- settings / logger public accessors ------------------------------

    @staticmethod
    def test_settings_expose_tests_namespace() -> None:
        """Test settings expose tests namespace."""
        settings = m.Tests.ServiceUserService().settings

        tm.that(settings, is_=FlextTestsSettings)
        if isinstance(settings, FlextTestsSettings):
            tm.that(settings.Tests, is_=m.BaseModel)

    def test_fetch_settings_returns_typed_tests_settings(self) -> None:
        """Test fetch settings returns typed tests settings."""
        with self._PureService.isolated_test_runtime():
            settings = self._PureService.fetch_settings()

            tm.that(settings, is_=FlextTestsSettings)
            tm.that(settings.Tests, is_=m.BaseModel)

    def test_fetch_logger_matches_shared_service_logger(self) -> None:
        """Test fetch logger matches shared service logger."""
        with self._PureService.isolated_test_runtime():
            expected_logger = self._PureService.fetch_global().logger

            def resolves_logger(logger: object) -> bool:
                return logger is expected_logger

            tm.that(self._PureService.fetch_logger(), where=resolves_logger)

    # --- with_settings(): runtime snapshot -------------------------------

    def test_with_settings_applies_provided_snapshot(self) -> None:
        """Test with settings applies provided snapshot."""
        settings = m.Tests.ServiceUserService().settings.clone(log_level="ERROR")

        service = self._PureService.with_settings(settings)

        dumped = service.settings.model_dump()
        tm.that(dumped, eq=settings.model_dump())
        tm.that(dumped["log_level"], eq="ERROR")

    # --- isolated_test_runtime(): isolation invariant --------------------

    def test_isolated_runtime_scopes_settings_without_leaking(self) -> None:
        """Test isolated runtime scopes settings without leaking."""
        baseline_level = self._PureService.fetch_settings().log_level

        with self._PureService.isolated_test_runtime(
            log_level="WARNING",
        ) as scoped_service:
            scoped_settings = FlextTestsSettings.model_validate(scoped_service.settings)

            tm.that(scoped_settings.log_level, eq="WARNING")
            scoped_global = FlextTestsSettings.fetch_global()
            tm.that(scoped_global.log_level, ne="WARNING")

        tm.that(self._PureService.fetch_settings().log_level, eq=baseline_level)

    # --- track(): context-manager metrics contract -----------------------

    @staticmethod
    def test_track_yields_named_operation_metrics() -> None:
        """Test track yields named operation metrics."""
        service = m.Tests.ServiceUserService()

        with service.track("load_users") as metrics:
            tm.that(metrics, is_=Mapping)
            tm.that(metrics[c.ContextKey.OPERATION_NAME], eq="load_users")
