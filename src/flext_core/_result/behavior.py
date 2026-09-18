"""Shared behavior contract for FlextResult."""

from __future__ import annotations

from typing import Self, override

from .._protocols.result import FlextProtocolsResult as prt
from .base import FlextResultBase


class FlextResultBehavior[T](FlextResultBase[T]):
    """Behavior layer: context manager, dunder methods, error accessors."""

    @property
    def failure(self) -> bool:
        return not self.success

    @property
    def value(self) -> T:
        if not self.success:
            error_msg = self.error or ""
            msg = f"Cannot access value of failed result: {error_msg}"
            raise RuntimeError(msg)
        return self._payload

    @property
    def exception(self) -> BaseException | None:
        return self._exception

    @classmethod
    def _factory(cls) -> type[prt.ResultFactory]:
        """Return the concrete MRO only after structural factory validation."""
        if isinstance(cls, prt.ResultFactory):
            return cls
        msg = f"{cls.__name__} does not implement the result factory contract"
        raise TypeError(msg)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: object,
    ) -> None:
        pass

    def __or__[D](self, default: D) -> T | D:
        if self.success:
            return self._payload
        return default

    def __bool__(self) -> bool:
        return self.success

    @override
    def __repr__(self) -> str:
        if self.success:
            return f"r[T].ok({self._payload!r})"
        return f"r[T].fail({self.error!r})"


__all__: list[str] = ["FlextResultBehavior"]
