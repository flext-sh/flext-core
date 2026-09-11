"""Container benchmark smoke test."""

from __future__ import annotations

from flext_tests import tm

from flext_core import FlextContainer


class TestsFlextContainerPerformance:
    """Compatibility benchmark class for lazy test exports."""

    def test_container_basic_resolution_path(self) -> None:
        """Resolve a registered value through the public container path."""
        container = FlextContainer()
        container.bind("bench.svc", "value")
        resolved = container.resolve("bench.svc")
        tm.ok(resolved, eq="value")
