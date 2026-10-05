"""Decorators example — demonstrates the logging decorator chain pattern.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING

from examples.protocols import p
from flext_core import r, u

if TYPE_CHECKING:
    from collections.abc import Callable


def _log_result[T](fn: Callable[..., T]) -> Callable[..., T]:
    """Decorator that logs each call, then passes the result through.

    Returns:
        The resulting ``Callable[..., T]``.

    """

    @wraps(fn)
    def _wrapper(*args: object, **kwargs: object) -> T:
        _ = u.fetch_logger(fn.__name__).info(f"calling {fn.__name__}")
        return fn(*args, **kwargs)

    return _wrapper


@_log_result
def run() -> p.Result[str]:
    """Return a deterministic decorators-like response.

    Returns:
        A deterministic decorators-like response.

    """
    return r[str].ok("decorator-example")


class Ex09FlextDecorators:
    """Compatibility wrapper expected by examples package exports."""

    @staticmethod
    def run() -> p.Result[str]:
        """Run decorators example.

        Returns:
            The resulting ``p.Result[str]``.

        """
        return run()


def _main() -> None:
    """Run the example as a script; names stay local so the package exports none.

    Raises:
        RuntimeError: If decorator example failed.

    """
    result = run()
    if not result.success:
        msg = "decorator example failed"
        raise RuntimeError(msg)


if __name__ == "__main__":
    _main()
