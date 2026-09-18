"""Shared behavior contract for FlextResult."""

from __future__ import annotations

from typing import Self, TypeIs, override

from .._protocols.result import FlextProtocolsResult as prt
from .base import FlextResultBase

_RESULT_FACTORY_CONTRACT: tuple[str, ...] = (
    "reject_banned_result_parameterization",
    "reject_banned_success_payload",
    "require_error",
    "fail",
    "from_result",
    "from_validation",
    "failed_result",
    "successful_result",
)


def _is_result_factory(cls: type[object]) -> TypeIs[type[prt.ResultFactory]]:
    """Narrow a class to the result factory contract after member validation."""
    return all(
        callable(getattr(cls, member, None)) for member in _RESULT_FACTORY_CONTRACT
    )


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
        factory_cls: type[object] = cls
        if not _is_result_factory(factory_cls):
            msg = f"{cls.__name__} does not implement the result factory contract"
            raise TypeError(msg)
        return factory_cls

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
