"""Behavioral extensions and Pythonic protocol implementations for FlextResult.

Implements context management, operator overloading, truthiness evaluation,
unwrapping helpers, and failure-raising semantics for the railway container.

See Also:
    - Workspace ADR-001 (Railway-Oriented Programming):
      ``flext/docs/architecture/adr/001-railway-oriented-programming.md``
    - :mod:`flext_core._result.base`: Data schema and strict parameter validation.
    - :mod:`flext_core._result.construction`: Result factory methods and combinators.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn, Self, override

import flext_core
from flext_core._result import FlextResultBase

if TYPE_CHECKING:
    from types import TracebackType


class FlextResultBehavior[T](FlextResultBase[T]):
    """Pythonic operator and behavior layer for FlextResult.

    Provides context manager protocols (``__enter__``/``__exit__``), Boolean
    truthiness evaluation, pipeline rail operators (``|``), and safe property
    accessors for success payloads and causal failure metadata.

    Type Parameters:
        T: The payload type carried upon successful completion.

    See Also:
        - Workspace ADR-001 (Railway-Oriented Programming):
          ``flext/docs/architecture/adr/001-railway-oriented-programming.md``

    """

    @property
    def failure(self) -> bool:
        """Whether this result represents an operational failure."""
        return not self.success

    @property
    def value(self) -> T:
        """The success payload bound to this result.

        Returns:
            T: The underlying success value.

        """
        if not self.success:
            self._raise_failure()
        return self._payload

    def _raise_failure(self) -> NoReturn:
        """Raise the causal failure while retaining its structured result context."""
        error = self._exception
        if error is None:
            error = RuntimeError(
                flext_core.c.ERR_RESULT_CANNOT_UNWRAP.format(error=self.error)
            )
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
        """The causal exception bound to this failure, or None."""
        return self._exception

    def __enter__(self) -> Self:
        """Enter the runtime context manager, returning self.

        Returns:
            Self: The active context manager instance.

        """
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: TracebackType | None,
    ) -> None:
        """Exit the runtime context manager without suppressing exceptions."""
        _ = _exc_type, _exc_val, _exc_tb

    def __or__[D](self, default: D) -> T | D:
        """Extract payload on success, or return the provided fallback default.

        Args:
            default: Fallback value returned if this result is a failure.

        Returns:
            T | D: The success payload if successful, otherwise ``default``.

        """
        if self.success:
            return self._payload
        return default

    def __bool__(self) -> bool:
        """Evaluate truthiness based on whether the operation succeeded.

        Returns:
            bool: True if the result represents success, False otherwise.

        """
        return self.success

    @override
    def __repr__(self) -> str:
        """Return the official string representation of the result instance.

        Returns:
            str: Diagnostic string detailing the success payload or failure message.

        """
        if self.success:
            return f"r[T].ok({self._payload!r})"
        return f"r[T].fail({self.error!r})"


__all__: list[str] = ["FlextResultBehavior"]
