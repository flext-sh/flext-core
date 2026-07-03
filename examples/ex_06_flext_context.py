"""Context example aligned to current public context API."""

from __future__ import annotations

from flext_core import FlextContext


def run() -> None:
    """Set and read a value from context."""
    ctx = FlextContext()
    if not ctx.set("example", "context").success:
        msg = "context set failed"
        raise RuntimeError(msg)
    value = ctx.get("example")
    if not value.success:
        msg = "context get failed"
        raise RuntimeError(msg)


class Ex06FlextContext:
    """Compatibility wrapper expected by examples package exports."""

    @staticmethod
    def run() -> None:
        """Run context example."""
        run()


if __name__ == "__main__":
    run()
