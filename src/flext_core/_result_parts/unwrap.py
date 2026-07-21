"""Value extraction operations for FlextResult."""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, overload, override

from flext_core._constants.errors import FlextConstantsErrors as c

from .behavior import FlextResultBehaviorMixin

if TYPE_CHECKING:
    from collections.abc import Callable


class FlextResultUnwrapMixin[T](FlextResultBehaviorMixin[T], ABC):
    """Value extraction helpers for results."""

    @override
    def unwrap(self) -> T:
        """Unwrap the success value or raise RuntimeError."""
        if self.failure:
            msg = c.ERR_RESULT_CANNOT_UNWRAP.format(error=self.error)
            raise RuntimeError(msg)
        return self.value

    @overload
    def unwrap_or(self, default: T) -> T: ...
    @overload
    def unwrap_or[DefaultT](self, default: DefaultT) -> T | DefaultT: ...

    @override
    def unwrap_or[DefaultT](self, default: DefaultT) -> T | DefaultT:
        """Return success value or default; safe extraction."""
        if self.success and self.value is not None:
            return self.value
        return default

    @overload
    def unwrap_or_else(self, func: Callable[[], T]) -> T: ...
    @overload
    def unwrap_or_else[DefaultT](
        self, func: Callable[[], DefaultT]
    ) -> T | DefaultT: ...

    @override
    def unwrap_or_else[DefaultT](self, func: Callable[[], DefaultT]) -> T | DefaultT:
        """Return the success value or call func if failed."""
        if self.success and self.value is not None:
            return self.value
        return func()


__all__: list[str] = ["FlextResultUnwrapMixin"]
