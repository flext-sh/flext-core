"""DI service access smoke tests using stable container/result contract."""

from __future__ import annotations

from flext_core import FlextContainer


class TestDIServicesAccess:
    def test_container_register_and_resolve_service(self) -> None:
        container = FlextContainer()
        container.bind("svc.test", "value")
        resolved = container.resolve("svc.test")
        assert resolved.success
        assert resolved.value == "value"

    def test_container_resolve_missing_service_returns_failure(self) -> None:
        container = FlextContainer()
        resolved = container.resolve("svc.missing")
        assert resolved.failure
