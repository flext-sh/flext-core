"""Container benchmark smoke test."""

from __future__ import annotations

from flext_core import FlextContainer


class TestsFlextCoreContainerPerformance:
    """Compatibility benchmark class for lazy test exports."""

    def test_container_basic_resolution_path(self) -> None:
        container = FlextContainer()
        container.bind("bench.svc", "value")
        resolved = container.resolve("bench.svc")
        assert resolved.success
