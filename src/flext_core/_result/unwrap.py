"""Value extraction operations for FlextResult.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core._result import FlextResultComposition

if TYPE_CHECKING:
    from collections.abc import Callable


class FlextResultUnwrap[T](FlextResultComposition[T]):
    """Value extraction helpers for results.

    Provides the terminal operations of the railway: ``unwrap`` (raises on
    failure), ``unwrap_or`` (default on failure) and ``unwrap_or_else``
    (lazily computed default on failure).
    """

    def unwrap(self) -> T:
        """Return the success payload.

        Raises:
            BaseException: The original exception carried by a failure.
            RuntimeError: If a logical failure has no original exception.

        """
        return self.value

    def unwrap_or[DefaultT](self, default: DefaultT) -> T | DefaultT:
        """Return the payload on success, otherwise ``default``.

        Args:
            default: The fallback value used when this result is a failure.

        Returns:
            The success payload or ``default``.

        """
        if self.success:
            return self._payload
        return default

    def unwrap_or_else[DefaultT](self, func: Callable[[], DefaultT]) -> T | DefaultT:
        """Return the payload on success, otherwise ``func()``.

        Args:
            func: The callable producing the fallback value on failure.

        Returns:
            The success payload or the ``func()`` result.

        """
        if self.success:
            return self._payload
        return func()


__all__: list[str] = ["FlextResultUnwrap"]
