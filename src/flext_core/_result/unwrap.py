"""Value extraction operations for FlextResult."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core import c

from .composition import FlextResultComposition

if TYPE_CHECKING:
    from collections.abc import Callable


class FlextResultUnwrap[T](FlextResultComposition[T]):
    """Value extraction helpers for results."""

    def unwrap(self) -> T:
        if self.failure:
            msg = c.ERR_RESULT_CANNOT_UNWRAP.format(error=self.error)
            raise RuntimeError(msg)
        return self.value

    def unwrap_or[DefaultT](self, default: DefaultT) -> T | DefaultT:
        if self.success:
            return self._payload
        return default

    def unwrap_or_else[DefaultT](self, func: Callable[[], DefaultT]) -> T | DefaultT:
        if self.success:
            return self._payload
        return func()


__all__: list[str] = ["FlextResultUnwrap"]
