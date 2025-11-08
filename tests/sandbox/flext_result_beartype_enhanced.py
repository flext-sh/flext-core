"""FlextResult ENHANCED com beartype - TÉCNICA 2: Decoração Dinâmica.

NOVO VALOR ADICIONADO:
✅ Valida tipos DENTRO de funções passadas como parâmetros
✅ Detecta erros que Pyright não pega (código dinâmico)
✅ API transparente (sem mudanças visíveis)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Generic, Self, TypeVar, cast

from beartype import beartype
from beartype.door import is_bearable

T_co = TypeVar("T_co", covariant=True)
U = TypeVar("U")


@beartype  # ← Decorar a CLASSE para validar todos os métodos
class FlextResultEnhanced(Generic[T_co]):
    """FlextResult com validação runtime MÁXIMA usando beartype.

    FEATURES ADICIONADAS:
    1. Validação de Callables (automática via @beartype na classe)
    2. Validação de tipos dentro de funções (decoração dinâmica)
    3. Validação de tipos genéricos (opcional via _type_hint)
    4. Validação de unwrap_or(default) (opcional)

    USAGE:
        # Básico (sem overhead extra)
        result = FlextResultEnhanced[int].ok(42)

        # Com validação de tipo genérico
        result = FlextResultEnhanced.ok(42, _type_hint=int)
        result.unwrap_or("string")  # ❌ TypeError

        # Validação automática de funções
        def bad_func(x: int) -> str:
            return 42  # ❌ BeartypeCallHintReturnViolation

        result.map(bad_func)  # Detecta erro!
    """

    def __init__(
        self,
        *,
        data: T_co | None = None,
        error: str | None = None,
        _type_hint: Any = None,
    ):
        """Initialize result with optional type validation."""
        super().__init__()

        # Validação opcional de tipo genérico
        if _type_hint is not None and data is not None:
            if not is_bearable(data, _type_hint):
                raise TypeError(
                    f"Data {data!r} is not bearable as {_type_hint}. "
                    f"Got type {type(data)}, expected {_type_hint}"
                )

        if error is not None:
            self._data = None
            self._error = error
        else:
            self._data = data
            self._error = None

        self._type_hint = _type_hint

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
        """Return the success payload, raising on failure."""
        if self.is_failure:
            raise ValueError("Attempted to access value on failed result")
        return cast("T_co", self._data)

    @property
    def error(self) -> str | None:
        """Return the captured error message for failure results."""
        return self._error

    @classmethod
    def ok(cls, data: T_co, _type_hint: Any = None) -> Self:
        """Create a successful result with optional type validation.

        Args:
            data: The success data to wrap.
            _type_hint: Optional type hint for runtime validation.
                       Example: FlextResultEnhanced.ok(42, _type_hint=int)

        Returns:
            Self: A successful result.

        Raises:
            TypeError: If data doesn't match _type_hint.

        """
        return cls(data=data, _type_hint=_type_hint)

    @classmethod
    def fail(cls, error: str | None) -> Self:
        """Create a failed result with error message.

        Args:
            error: The error message.

        Returns:
            Self: A failed result.

        """
        actual_error = error if error and not error.isspace() else "Unknown error"
        return cls(error=actual_error)

    # =========================================================================
    # TÉCNICA 2: Decorar Callables Dinamicamente
    # =========================================================================

    def map[U](
        self,
        func: Callable[[T_co], U],
        *,
        validate_func: bool = True,
    ) -> FlextResultEnhanced[U]:
        """Transform success value with RUNTIME TYPE VALIDATION of func.

        TÉCNICA 2: Decora func com @beartype antes de executar,
        adicionando validação runtime de tipos dentro da função.

        Args:
            func: Function to transform the success value.
            validate_func: If True, decorates func with beartype for validation.
                          Set to False for performance-critical code.

        Returns:
            FlextResultEnhanced[U]: New result with transformed value or error.

        Examples:
            >>> def good_func(x: int) -> str:
            ...     return str(x * 2)
            >>> result = FlextResultEnhanced[int].ok(5).map(good_func)
            >>> result.value
            '10'

            >>> def bad_func(x: int) -> str:
            ...     return 42  # ❌ Retorna int, declara str!
            >>> result = FlextResultEnhanced[int].ok(5).map(bad_func)
            >>> # BeartypeCallHintReturnViolation!

        """
        if self.is_failure:
            return FlextResultEnhanced[U](error=self._error)

        try:
            # TÉCNICA 2: Decorar func com beartype dinamicamente
            if validate_func:
                func_validated = beartype(func)
            else:
                func_validated = func

            result = func_validated(cast("T_co", self._data))
            return FlextResultEnhanced[U].ok(result)
        except Exception as e:
            error_msg = f"Map transformation failed: {type(e).__name__}: {e}"
            return FlextResultEnhanced[U].fail(error_msg)

    def flat_map[U](
        self,
        func: Callable[[T_co], FlextResultEnhanced[U]],
        *,
        validate_func: bool = True,
    ) -> FlextResultEnhanced[U]:
        """Chain operations with RUNTIME TYPE VALIDATION of func.

        TÉCNICA 2: Decora func com @beartype antes de executar.

        Args:
            func: Function returning FlextResultEnhanced[U].
            validate_func: If True, decorates func with beartype for validation.

        Returns:
            FlextResultEnhanced[U]: Result from func or error.

        """
        if self.is_failure:
            return FlextResultEnhanced[U](error=self._error)

        try:
            # TÉCNICA 2: Decorar func com beartype dinamicamente
            if validate_func:
                func_validated = beartype(func)
            else:
                func_validated = func

            return func_validated(cast("T_co", self._data))
        except Exception as e:
            error_msg = f"Flat map failed: {type(e).__name__}: {e}"
            return FlextResultEnhanced[U].fail(error_msg)

    def unwrap(self) -> T_co:
        """Get value or raise if failed."""
        if self.is_success:
            return cast("T_co", self._data)
        raise ValueError(self._error or "Operation failed")

    def unwrap_or(self, default: T_co) -> T_co:
        """Return value or default if failed, with optional type validation.

        If result was created with _type_hint, validates that default
        matches the expected type.

        Args:
            default: Default value if result is failure.

        Returns:
            T_co: The success value or default.

        Raises:
            TypeError: If default doesn't match _type_hint.

        """
        # Validação de tipo se _type_hint foi fornecido
        if self._type_hint is not None:
            if not is_bearable(default, self._type_hint):
                raise TypeError(
                    f"default {default!r} (type {type(default)}) "
                    f"doesn't match expected type {self._type_hint}"
                )

        if self.is_success:
            return cast("T_co", self._data)
        return default


__all__ = ["FlextResultEnhanced"]
