"""Incremental DI smoke tests aligned to current container/runtime contracts."""

from __future__ import annotations

from flext_core import FlextContainer


class TestDIIncremental:
    def test_container_bind_and_resolve_success(self) -> None:
        FlextContainer.reset_for_testing()
        container = FlextContainer.shared()
        container.bind("svc.di", "ok")
        resolved = container.resolve("svc.di")
        assert resolved.success
        assert resolved.value == "ok"

    def test_container_resolve_missing_returns_failure(self) -> None:
        FlextContainer.reset_for_testing()
        container = FlextContainer.shared()
        resolved = container.resolve("svc.missing")
        assert resolved.failure
