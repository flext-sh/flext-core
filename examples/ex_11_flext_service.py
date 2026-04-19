"""Service usage example aligned to current slim service API."""

from __future__ import annotations

from examples import p, r


class ExampleService:
    """Minimal runnable example service wrapper."""

    @staticmethod
    def run() -> None:
        """Execute the service example and enforce expected result contract."""
        result: p.Result[str] = r[str].ok("service-example")
        if not result.success or result.value != "service-example":
            msg = "Service example failed"
            raise RuntimeError(msg)


class Ex11FlextService:
    """Compatibility wrapper for lazy-exported example symbol."""

    @staticmethod
    def run() -> None:
        """Execute the service example compatibility entrypoint."""
        ExampleService.run()


if __name__ == "__main__":
    ExampleService.run()
