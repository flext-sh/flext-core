"""Decorators example aligned to stable helper contracts."""

from __future__ import annotations

from examples import p, r


def run() -> p.Result[str]:
    """Return a deterministic decorators-like response."""
    return r[str].ok("decorator-example")


class Ex09FlextDecorators:
    """Compatibility wrapper expected by examples package exports."""

    @staticmethod
    def run() -> p.Result[str]:
        """Run decorators example."""
        return run()


if __name__ == "__main__":
    result = run()
    if not result.success:
        msg = "decorator example failed"
        raise RuntimeError(msg)
