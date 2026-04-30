"""Behavioral tests for the current slim service contract."""

from __future__ import annotations

from tests import m, s


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


__all__: list[str] = ["TestsFlextService"]
