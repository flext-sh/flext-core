"""Behavioral tests for the current slim service contract."""

from __future__ import annotations

from typing import override

from flext_tests import FlextTestsCase
from flext_tests.settings import FlextTestsSettings

from tests import c, e, m, p, r, s, t, u


class TestsFlextService(FlextTestsCase):
    """Validate stable behavior of FlextService subclasses."""

    def test_shared_pytest_runtime_binds_aliases_to_self(self) -> None:
        assert self.service is type(self.service).fetch_global()
        assert self.settings is self.service.settings
        assert self.logger is self.service.logger
        assert self.c is c
        assert self.e is e
        assert self.m is m
        assert self.p is p
        assert self.r is r
        assert self.t is t
        assert self.u is u

    def test_service_initializes_and_executes(self) -> None:
        service = m.Tests.ServiceUserService()
        assert isinstance(service, s)
        result = service.execute()
        assert result.success

    def test_execute_returns_typed_payload(self) -> None:
        result = m.Tests.ServiceUserService().execute()
        assert result.success
        assert result.value == m.Tests.ServiceUserData(
            user_id=1,
            name="test_user",
        )

    def test_test_service_settings_include_tests_namespace(self) -> None:
        settings = m.Tests.ServiceUserService().settings

        assert isinstance(settings, FlextTestsSettings)
        assert isinstance(settings.Tests, m.SettingsValue)

    class _PureService(s[bool]):
        @override
        def execute(self) -> p.Result[bool]:
            return r[bool].ok(True)

    def test_fetch_settings_returns_typed_tests_settings(self) -> None:
        with self._PureService.isolated_test_runtime():
            settings = self._PureService.fetch_settings()

            assert isinstance(settings, FlextTestsSettings)
            assert isinstance(settings.Tests, m.SettingsValue)

    def test_fetch_logger_reuses_service_logger(self) -> None:
        with self._PureService.isolated_test_runtime():
            logger = self._PureService.fetch_logger()

            assert logger is self._PureService.fetch_global().logger

    def test_with_settings_uses_provided_runtime_settings_snapshot(self) -> None:
        settings = m.Tests.ServiceUserService().settings.clone(
            app_name="service-settings-override",
        )

        service = self._PureService.with_settings(settings)
        service_settings_dump = service.settings.model_dump()

        assert service_settings_dump == settings.model_dump()
        assert service_settings_dump["app_name"] == "service-settings-override"

    def test_with_test_settings_clones_current_runtime_settings(self) -> None:
        with self._PureService.isolated_test_runtime(
            app_name="service-test-helper",
        ):
            global_settings = FlextTestsSettings.fetch_global()

            assert self._PureService.fetch_settings().app_name == "service-test-helper"
            assert FlextTestsSettings.fetch_global() is global_settings
            assert FlextTestsSettings.fetch_global().app_name != "service-test-helper"


__all__: list[str] = ["TestsFlextService"]
