"""Handlers example aligned to current stable result contract."""

from __future__ import annotations

from examples import p, r


def run() -> p.Result[str]:
    """Return a successful handler-like response."""
    return r[str].ok("handler-example")


class Ex10FlextHandlers:
    """Compatibility wrapper expected by examples package exports."""

    @staticmethod
    def run() -> p.Result[str]:
        """Run handlers example."""
        return run()


if __name__ == "__main__":
    result = run()
    if not result.success:
        msg = "handler example failed"
        raise RuntimeError(msg)
