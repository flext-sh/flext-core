"""Service/result integration smoke tests."""

from __future__ import annotations

from typing import override

from flext_tests.base import s

from tests import p, r


class _IntegrationService(s[str]):
    @override
    def execute(self) -> p.Result[str]:
        return r[str].ok("ok")


class TestsFlextServiceResultProperty:
    def test_service_execute_returns_success(self) -> None:
        result = _IntegrationService().execute()
        assert result.success
        assert result.value == "ok"
