"""Tests for the current slim service bootstrap surface."""

from __future__ import annotations

from tests import p, r, s


class TestsFlextCoreServiceBootstrap:
    """Test service initialization and execution patterns."""

    class ConcreteTestService(s[bool]):
        """Concrete service for constructor/execute tests."""

        def execute(self) -> p.Result[bool]:
            return r[bool].ok(True)

    def test_service_constructor_accepts_runtime_data(self) -> None:
        service = self.ConcreteTestService()
        assert isinstance(service, self.ConcreteTestService)

    def test_service_execute_returns_ok_result(self) -> None:
        result = self.ConcreteTestService().execute()
        assert result.success
        assert result.value is True
