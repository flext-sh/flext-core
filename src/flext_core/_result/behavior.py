"""Shared behavior contract for FlextResult.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn, Self, override

from flext_core import c
from flext_core._result import FlextResultBase

if TYPE_CHECKING:
    from types import TracebackType


class FlextResultBehavior[T](FlextResultBase[T]):
    """Behavior layer: context manager, dunder methods, error accessors.

    Adds the pythonic surface of the result — ``__enter__``/``__exit__`` for
    scoped usage, truthiness via ``__bool__``, the ``|`` rail operator, and
    the ``failure``, ``value`` and ``exception`` accessors that expose the
    failure state.
    """

    @property
    def failure(self) -> bool:
        """Whether this result represents a failure."""
        return not self.success

    @property
    def value(self) -> T:
        """The success payload bound to this result."""
        if not self.success:
            self._raise_failure()
        return self._payload

    def _raise_failure(self) -> NoReturn:
        """Raise one causal failure while retaining its structured result context."""
        error = self._exception
        if error is None:
            error = RuntimeError(c.ERR_RESULT_CANNOT_UNWRAP.format(error=self.error))
        if self.error or self.error_code is not None or self.error_data is not None:
            note = self.model_dump_json(
                include={"error", "error_code", "error_data"},
                exclude_none=True,
            )
            if not hasattr(error, "__notes__") or note not in error.__notes__:
                error.add_note(note)
        raise error

    @property
    def exception(self) -> BaseException | None:
        """The exception that produced this failure, if any."""
        return self._exception

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: TracebackType | None,
    ) -> None:
        pass

    def __or__[D](self, default: D) -> T | D:
        """Return the payload on success, otherwise ``default``."""
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
