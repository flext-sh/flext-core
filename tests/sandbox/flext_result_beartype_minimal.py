"""Versão mínima de FlextResult com @beartype para validação incremental.

Estratégia: Começar com poucos métodos, fazer funcionar, depois expandir.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Self, cast

from beartype import beartype

from flext_core.exceptions import FlextExceptions


@beartype  # ← SOLUÇÃO: Decorar a CLASSE inteira!
class FlextResultBeartype[T_co]:
    """Versão mínima de FlextResult com @beartype - FASE 1: Construtores + Hot Path."""

    _data: T_co | None
    _error: str | None

    def __init__(
        self,
        *,
        data: T_co | None = None,
        error: str | None = None,
    ) -> None:
        """Initialize result with either success data or error."""
        super().__init__()
        if error is not None:
            self._data = None
            self._error = error
        else:
            self._data = data
            self._error = None

    @property
    def is_success(self) -> bool:
        """Return True when the result carries a successful payload."""
        return self._error is None

    @property
    def is_failure(self) -> bool:
        """Return True when the result represents a failure."""
        return self._error is not None

    @property
    def value(self) -> T_co:
        """Return the success payload, raising ValidationError on failure."""
        if self.is_failure:
            msg = "Attempted to access value on failed result"
            raise FlextExceptions.ValidationError(
                message=msg,
                error_code="VALIDATION_ERROR",
            )
        return cast("T_co", self._data)

    @property
    def error(self) -> str | None:
        """Return the captured error message for failure results."""
        return self._error

    # =========================================================================
    # TESTE 1: Construtores com Self - Validados por @beartype na classe
    # =========================================================================

    @classmethod
    def ok(cls, data: T_co) -> Self:
        """Create a successful result - VALIDADO por @beartype na classe."""
        return cls(data=data)

    @classmethod
    def fail(cls, error: str | None) -> Self:
        """Create a failed result - VALIDADO por @beartype na classe."""
        actual_error = (
            error if error and not error.isspace() else "Unknown error occurred"
        )
        return cls(error=actual_error)

    # =========================================================================
    # TESTE 2: Hot Path - Validados por @beartype na classe
    # =========================================================================

    def map[U](self, func: Callable[[T_co], U]) -> FlextResultBeartype[U]:
        """Transform the success payload using func while preserving errors."""
        if self.is_failure:
            return FlextResultBeartype[U].fail(self._error)

        try:
            result = func(cast("T_co", self._data))
            return FlextResultBeartype[U].ok(result)
        except Exception as e:
            return FlextResultBeartype[U].fail(f"Transformation failed: {e}")

    def flat_map[U](
        self, func: Callable[[T_co], FlextResultBeartype[U]]
    ) -> FlextResultBeartype[U]:
        """Chain operations returning FlextResultBeartype."""
        if self.is_failure:
            return FlextResultBeartype[U].fail(self._error)

        try:
            return func(cast("T_co", self._data))
        except Exception as e:
            return FlextResultBeartype[U].fail(f"Flat map operation failed: {e}")

    def unwrap(self) -> T_co:
        """Get value or raise if failed."""
        if self.is_success:
            return cast("T_co", self._data)
        error_msg = self._error or "Operation failed"
        raise FlextExceptions.BaseError(
            message=error_msg,
            error_code="OPERATION_ERROR",
        )

    def unwrap_or(self, default: T_co) -> T_co:
        """Return value or default if failed."""
        if self.is_success:
            return cast("T_co", self._data)
        return default


__all__ = ["FlextResultBeartype"]
