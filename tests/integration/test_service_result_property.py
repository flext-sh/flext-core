"""Service/result integration smoke tests."""

from __future__ import annotations

from tests import p, r, s


class _IntegrationService(s[str]):
    def execute(self) -> p.Result[str]:
        return r[str].ok("ok")


class TestServiceResultProperty:
    def test_service_execute_returns_success(self) -> None:
        result = _IntegrationService().execute()
        assert result.success
        assert result.value == "ok"
