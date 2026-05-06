"""Behavioral tests for the current slim service contract."""

from __future__ import annotations

from typing import override

from tests import m, p, r, s


class TestsFlextService:
    """Validate stable behavior of FlextService subclasses."""

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

    class _PureService(s[bool]):
        @override
        def execute(self) -> p.Result[bool]:
            return r[bool].ok(True)

    def test_with_settings_uses_provided_runtime_settings_snapshot(self) -> None:
        settings = m.Tests.ServiceUserService().settings.clone(
            app_name="service-settings-override",
        )

        service = self._PureService.with_settings(settings)
        service_settings_dump = service.settings.model_dump()

        assert service_settings_dump == settings.model_dump()
        assert service_settings_dump["app_name"] == "service-settings-override"


__all__: list[str] = ["TestsFlextService"]
