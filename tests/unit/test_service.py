"""Behavioral tests for the slim ``FlextService`` public contract.

Every assertion targets observable public behavior: the ``r[T]`` outcome of
``execute``, the shared-runtime singleton returned by ``fetch_global``, the
public ``settings``/``logger`` accessors, the ``with_settings`` snapshot, the
``isolated_test_runtime`` isolation invariant, the ``track`` context-manager
contract, and the public state of the ``ServiceUserData`` result model. No
private attributes, internal collaborators, or implementation details are
inspected.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import override

import pytest
from flext_tests import FlextTestsCase, r
from flext_tests.settings import FlextTestsSettings

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

    def test_execute_reports_success_and_typed_payload(self) -> None:
        service = m.Tests.ServiceUserService()

        result = service.execute()

        assert result.success
        assert not result.failure
        assert result.unwrap() == m.Tests.ServiceUserData(user_id=1, name="test_user")

    def test_execute_success_value_exposes_public_model_fields(self) -> None:
        result = m.Tests.ServiceUserService().execute()

        payload = result.value
        assert isinstance(payload, m.Tests.ServiceUserData)
        assert payload.user_id == 1
        assert payload.name == "test_user"

    def test_execute_failure_propagates_error_through_result(self) -> None:
        result = self._FailingService().execute()

        assert result.failure
        assert not result.success
        assert result.error == "execute-boom"

    def test_execute_success_result_supports_combinators(self) -> None:
        result = self._PureService().execute()

        assert result.map(lambda ok: ok and True).unwrap() is True
        assert result.flat_map(lambda ok: r[bool].ok(value=not ok)).unwrap() is False

    def test_execute_failure_result_short_circuits_combinators(self) -> None:
        result = self._FailingService().execute()

        assert result.map(lambda ok: not ok).failure
        assert result.unwrap_or(default=False) is False

    def test_service_instance_is_a_flext_service(self) -> None:
        service = m.Tests.ServiceUserService()

        assert isinstance(service, s)

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
        data = m.Tests.ServiceUserData(user_id=user_id, name=name)

        assert data.model_dump() == {"user_id": user_id, "name": name}
        assert data == m.Tests.ServiceUserData(user_id=user_id, name=name)

    # --- fetch_global(): shared-runtime singleton ------------------------

    def test_fetch_global_returns_the_shared_singleton(self) -> None:
        first = type(self.service).fetch_global()
        second = type(self.service).fetch_global()

        assert first is self.service
        assert second is first

    # --- settings / logger public accessors ------------------------------

    def test_settings_expose_tests_namespace(self) -> None:
        settings = m.Tests.ServiceUserService().settings

        assert isinstance(settings, FlextTestsSettings)
        assert isinstance(settings.Tests, m.SettingsValue)

    def test_fetch_settings_returns_typed_tests_settings(self) -> None:
        with self._PureService.isolated_test_runtime():
            settings = self._PureService.fetch_settings()

            assert isinstance(settings, FlextTestsSettings)
            assert isinstance(settings.Tests, m.SettingsValue)

    def test_fetch_logger_matches_shared_service_logger(self) -> None:
        with self._PureService.isolated_test_runtime():
            assert (
                self._PureService.fetch_logger()
                is self._PureService.fetch_global().logger
            )

    # --- with_settings(): runtime snapshot -------------------------------

    def test_with_settings_applies_provided_snapshot(self) -> None:
        settings = m.Tests.ServiceUserService().settings.clone(
            app_name="service-settings-override",
        )

        service = self._PureService.with_settings(settings)

        dumped = service.settings.model_dump()
        assert dumped == settings.model_dump()
        assert dumped["app_name"] == "service-settings-override"

    # --- isolated_test_runtime(): isolation invariant --------------------

    def test_isolated_runtime_scopes_settings_without_leaking(self) -> None:
        baseline_app_name = self._PureService.fetch_settings().app_name

        with self._PureService.isolated_test_runtime(app_name="service-scoped"):
            scoped_global = FlextTestsSettings.fetch_global()

            assert self._PureService.fetch_settings().app_name == "service-scoped"
            assert FlextTestsSettings.fetch_global() is scoped_global
            assert scoped_global.app_name != "service-scoped"

        assert self._PureService.fetch_settings().app_name == baseline_app_name

    # --- track(): context-manager metrics contract -----------------------

    def test_track_yields_named_operation_metrics(self) -> None:
        service = m.Tests.ServiceUserService()

        with service.track("load_users") as metrics:
            assert isinstance(metrics, Mapping)
            assert metrics["operation_name"] == "load_users"


__all__: list[str] = ["TestsFlextService"]
