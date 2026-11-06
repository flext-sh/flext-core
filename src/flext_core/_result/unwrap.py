"""Terminal value unwrapping and extraction helpers for FlextResult.

Supplies operations to extract payloads from railway results, allowing controlled
exit from the railway track either by raising explicit causal exceptions (``unwrap``)
or by falling back to static or dynamically computed defaults (``unwrap_or``,
``unwrap_or_else``).

See Also:
    - Workspace ADR-001 (Railway-Oriented Programming):
      ``flext/docs/architecture/adr/001-railway-oriented-programming.md``
    - :mod:`flext_core._result.behavior`: Property accessors for value and state.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

from flext_core._result import FlextResultComposition

if TYPE_CHECKING:
    from collections.abc import Callable


class FlextResultUnwrap[T](FlextResultComposition[T], ABC):
    """Terminal value extraction helpers for railway results.

    Provides boundary extraction methods enabling consumers to safely retrieve
    underlying success values or handle failures with defaults or explicit exceptions.

    Type Parameters:
        T: The payload type carried upon successful completion.

    See Also:
        - Workspace ADR-001 (Railway-Oriented Programming):
          ``flext/docs/architecture/adr/001-railway-oriented-programming.md``

    """

    def unwrap(self) -> T:
        """Extract the success payload, raising a causal exception on failure.

        Returns:
            T: The underlying success payload.

        """
        return self.value

    def unwrap_or[DefaultT](self, default: DefaultT) -> T | DefaultT:
        """Return payload on success, otherwise return the provided default fallback.

        Args:
            default: The fallback value to return if this result is a failure.

        Returns:
            T | DefaultT: The success payload or ``default``.

        """
        if self.success:
            return self._payload
        return default

    def unwrap_or_else[DefaultT](self, func: Callable[[], DefaultT]) -> T | DefaultT:
        """Return payload on success, otherwise invoke ``func()`` for a fallback.

        Args:
            func: A zero-argument callable invoked lazily on failure to compute value.

        Returns:
            T | DefaultT: The success payload or the result of ``func()``.

        """
        if self.success:
            return self._payload
        return func()


__all__: list[str] = ["FlextResultUnwrap"]
