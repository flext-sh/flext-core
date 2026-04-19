"""Container benchmark smoke test."""

from __future__ import annotations

from flext_core import FlextContainer


def test_container_basic_resolution_path() -> None:
    container = FlextContainer()
    container.bind("bench.svc", "value")
    resolved = container.resolve("bench.svc")
    assert resolved.success


class TestContainerPerformance:
    """Compatibility benchmark class for lazy test exports."""

    def test_container_basic_resolution_path(self) -> None:
        test_container_basic_resolution_path()
