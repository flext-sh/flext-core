"""Dispatcher example aligned to current stable result contract."""

from __future__ import annotations

from flext_core import p, r


def run() -> p.Result[str]:
    """Return a successful dispatcher-like response."""
    return r[str].ok("dispatcher-example")


class Ex04DispatchDsl:
    """Compatibility wrapper expected by examples package exports."""

    @staticmethod
    def run() -> p.Result[str]:
        """Run dispatcher example."""
        return run()


if __name__ == "__main__":
    result = run()
    if not result.success:
        msg = "dispatcher example failed"
        raise RuntimeError(msg)
    print("PASS: ex_04_flext_dispatcher")
