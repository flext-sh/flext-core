"""Behavioral tests for the current slim service contract."""

from __future__ import annotations

from tests import m, p, r, s


class TestsFlextCoreServiceUserData(m.Value):
    """Public result model used by service tests."""

    __test__ = False

    user_id: int
    name: str


class TestsFlextCoreServiceUserService(s):
    """Simple successful service."""

    __test__ = False

    def execute(self) -> p.Result[TestsFlextCoreServiceUserData]:
        return r[TestsFlextCoreServiceUserData].ok(
            TestsFlextCoreServiceUserData(user_id=1, name="test_user")
        )


class TestsFlextCoreService:
    """Validate stable behavior of FlextService subclasses."""

    def test_service_initializes_and_executes(self) -> None:
        service = TestsFlextCoreServiceUserService()
        assert isinstance(service, s)
        result = service.execute()
        assert result.success

    def test_execute_returns_typed_payload(self) -> None:
        result = TestsFlextCoreServiceUserService().execute()
        assert result.success
        assert result.value == TestsFlextCoreServiceUserData(
            user_id=1,
            name="test_user",
        )
