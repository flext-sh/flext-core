"""Value extraction operations for FlextResult."""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

from flext_core._constants.errors import FlextConstantsErrors as c

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

    @overload
    def unwrap_or(self, default: T) -> T: ...
    @overload
    def unwrap_or[DefaultT](self, default: DefaultT) -> T | DefaultT: ...

    def unwrap_or[DefaultT](self, default: DefaultT) -> T | DefaultT:
        if self.success:
            return self._payload
        return default

    @overload
    def unwrap_or_else(self, func: Callable[[], T]) -> T: ...
    @overload
    def unwrap_or_else[DefaultT](
        self, func: Callable[[], DefaultT]
    ) -> T | DefaultT: ...

    def unwrap_or_else[DefaultT](self, func: Callable[[], DefaultT]) -> T | DefaultT:
        if self.success:
            return self._payload
        return func()


__all__: list[str] = ["FlextResultUnwrap"]
