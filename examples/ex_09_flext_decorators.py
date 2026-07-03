"""Decorators example — demonstrates the logging decorator chain pattern."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps

from examples import p, r


def _log_result[T](fn: Callable[..., T]) -> Callable[..., T]:
    """Decorator that passes through a result-returning function unchanged."""

    @wraps(fn)
    def _wrapper(*args: object, **kwargs: object) -> T:
        return fn(*args, **kwargs)

    return _wrapper


@_log_result
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
