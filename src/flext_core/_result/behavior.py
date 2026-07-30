"""Shared behavior contract for FlextResult."""

from __future__ import annotations

from typing import Self, overload, override

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
            msg = f"Cannot access value on failed result: {error_msg}"
            raise ValueError(msg)
        return self._payload

    @property
    def exception(self) -> BaseException | None:
        return self._exception

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: object,
    ) -> None:
        pass

    @overload
    def __or__(self, default: T) -> T: ...
    @overload
    def __or__[D](self, default: D) -> T | D: ...
    def __or__[D](self, default: T | D) -> T | D:
        if self.success:
            return self._payload
        return default

    def __bool__(self) -> bool:
        return self.success

    @override
    def __repr__(self) -> str:
        return f"{type(self).__name__}(success={self.success})"


__all__: list[str] = ["FlextResultBehavior"]
