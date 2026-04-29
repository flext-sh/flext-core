"""Behavioral tests for the current slim service contract."""

from __future__ import annotations

from typing import Annotated, override

from tests import m, p, r, s


class TestsFlextServiceUserData(m.Value):
    """Public result model used by service tests."""

    __test__ = False

    user_id: Annotated[int, m.Field(description="User identifier")]
    name: Annotated[str, m.Field(description="User name")]


class TestsFlextServiceUserService(s):
    """Simple successful service."""

    __test__ = False

    @override
    def execute(self) -> p.Result[TestsFlextServiceUserData]:
        return r[TestsFlextServiceUserData].ok(
            TestsFlextServiceUserData(user_id=1, name="test_user")
        )


class TestsFlextService:
    """Validate stable behavior of FlextService subclasses."""

    def test_service_initializes_and_executes(self) -> None:
        service = TestsFlextServiceUserService()
        assert isinstance(service, s)
        result = service.execute()
        assert result.success

    def test_execute_returns_typed_payload(self) -> None:
        result = TestsFlextServiceUserService().execute()
        assert result.success
        assert result.value == TestsFlextServiceUserData(
            user_id=1,
            name="test_user",
        )
