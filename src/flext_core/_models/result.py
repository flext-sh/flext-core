from __future__ import annotations

from collections.abc import Callable
from types import TracebackType
from typing import Annotated, Self, cast, override

import structlog
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from flext_core.protocols import FlextProtocols as p
from flext_core.typings import FlextTypes as t


class FlextModelsResult:
    """Result pattern model (Layer 0.5)."""

    class RuntimeResult[T](BaseModel):
        """Lightweight implementation of Result pattern (Layer 0.5).

        Implements basic success/failure handling with Pydantic integration.
        Compatible with p.Result and r usage patterns.
        """

        model_config = ConfigDict(
            arbitrary_types_allowed=True,
            frozen=False,
            populate_by_name=True,
        )

        is_success: Annotated[bool, Field(default=True)]
        _payload: T | None = PrivateAttr(default=None)
        error: Annotated[str | None, Field(default=None)]
        error_code: Annotated[str | None, Field(default=None)]
        error_data: Annotated[t.ConfigMap | None, Field(default=None)]

        _exception: BaseException | None = PrivateAttr(default=None)
        _result_logger: p.Logger | None = PrivateAttr(default=None)

        @override
        def __repr__(self) -> str:
            """String representation using short alias 'r' for brevity."""
            if self.is_success:
                return f"r[T].ok({self.value!r})"
            return f"r[T].fail({self.error!r})"

        def __bool__(self) -> bool:
            """Boolean conversion based on success state."""
            return self.is_success

        def __enter__(self) -> Self:
            """Context manager entry."""
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_val: BaseException | None,
            exc_tb: TracebackType | None,
        ) -> None:
            """Context manager exit."""

        def __or__(self, default: T) -> T:
            """Operator overload for default values."""
            return self.unwrap_or(default)

        @property
        def exception(self) -> BaseException | None:
            """Get the exception if one was captured."""
            return self._exception

        @property
        def is_failure(self) -> bool:
            """Check if result is a failure."""
            return not self.is_success

        @property
        def result_logger(self) -> p.Logger:
            """Logger for RuntimeResult."""
            logger = self._result_logger
            if logger is None:
                logger = structlog.get_logger(__name__)
                self._result_logger = logger
            return logger

        @property
        def value(self) -> T:
            """Result value — returns _payload directly on success.

            None IS a valid payload when T includes None (e.g. r[str | None]).
            DO NOT add None checks, asserts, or invariant guards here.
            The only guard is is_success — if the result is a failure,
            accessing .value raises RuntimeError. cast() narrows T | None to T.
            """
            if not self.is_success:
                msg = f"Cannot access value of failed result: {self.error}"
                raise RuntimeError(msg)
            if self._payload is not None:
                return self._payload
            return cast("T", self._payload)

        @classmethod
        def fail(
            cls,
            error: str | None,
            error_code: str | None = None,
            error_data: t.ResultErrorData | t.ConfigModelInput | None = None,
        ) -> Self:
            """Create failed result with error message.

            Business Rule: Creates failed RuntimeResult with error message, optional error
            code, and optional error metadata. Converts None error to empty string for
            consistency. This matches the API of r.fail() for compatibility.

            Args:
                error: Error message (None will be converted to empty string)
                error_code: Optional error code for categorization
                error_data: Optional error metadata

            Returns:
                Failed RuntimeResult instance

            """
            error_msg = error if error is not None else ""
            validated_error_data: t.ConfigMap
            if error_data is None:
                validated_error_data = t.ConfigMap(root={})
            elif isinstance(error_data, t.ConfigMap):
                validated_error_data = error_data
            elif isinstance(error_data, BaseModel):
                dump = error_data.model_dump()
                validated_error_data = t.ConfigMap(dump)
            else:
                validated_error_data = t.ConfigMap(dict(error_data))

            return cls(
                is_success=False,
                error=error_msg,
                error_code=error_code,
                error_data=validated_error_data,
            )

        @classmethod
        def ok(cls, value: T) -> FlextModelsResult.RuntimeResult[T]:
            """Create successful result wrapping data.

            Business Rule: Creates successful RuntimeResult wrapping value. Raises ValueError
            if value is None (None values are not allowed in success results). This enforces
            the same invariant as r.ok() at the base class level.

            Args:
                value: Value to wrap in success result (must not be None)

            Returns:
                Successful RuntimeResult instance

            """
            instance = cls(
                is_success=True,
                error=None,
                error_code=None,
                error_data=t.ConfigMap(root={}),
            )
            instance._payload = value
            return instance

        def filter(
            self,
            predicate: Callable[[T], bool],
        ) -> FlextModelsResult.RuntimeResult[T]:
            """Filter success value using predicate."""
            if self.is_success and (not predicate(self.value)):
                return FlextModelsResult.RuntimeResult[T].fail(
                    error="Filter predicate failed",
                )
            return self

        def flat_map[U](
            self,
            func: Callable[[T], FlextModelsResult.RuntimeResult[U]],
        ) -> FlextModelsResult.RuntimeResult[U]:
            """Chain operations returning RuntimeResult."""
            if self.is_success:
                return func(self.value)
            return FlextModelsResult.RuntimeResult[U].fail(
                error=self.error,
                error_code=self.error_code,
                error_data=self.error_data,
            )

        def flow_through[U](
            self,
            *funcs: Callable[[T | U], FlextModelsResult.RuntimeResult[U]],
        ) -> FlextModelsResult.RuntimeResult[T] | FlextModelsResult.RuntimeResult[U]:
            """Chain multiple operations in sequence.

            Returns:
                RuntimeResult[T] if no funcs provided, value is None, or chain
                short-circuits on failure. RuntimeResult[U] if all funcs applied.

            """
            if self.is_failure or not funcs:
                return self
            current: (
                FlextModelsResult.RuntimeResult[T] | FlextModelsResult.RuntimeResult[U]
            ) = self
            for func in funcs:
                if current.is_success:
                    result_value = current.value
                    if result_value is not None:
                        current = func(result_value)
                    else:
                        break
                else:
                    break
            return current

        def fold[U](
            self,
            on_failure: Callable[[str], U],
            on_success: Callable[[T], U],
        ) -> U:
            """Fold result into single value (catamorphism)."""
            if self.is_success:
                return on_success(self.value)
            return on_failure(self.error or "")

        def lash(
            self,
            func: Callable[[str], FlextModelsResult.RuntimeResult[T]],
        ) -> FlextModelsResult.RuntimeResult[T]:
            """Apply recovery function on failure."""
            if not self.is_success:
                return func(self.error or "")
            return self

        def map[U](self, func: Callable[[T], U]) -> FlextModelsResult.RuntimeResult[U]:
            """Transform success value using function."""
            if self.is_success:
                try:
                    return FlextModelsResult.RuntimeResult[U].ok(value=func(self.value))
                except (
                    ValueError,
                    TypeError,
                    KeyError,
                    AttributeError,
                    RuntimeError,
                ) as e:
                    return FlextModelsResult.RuntimeResult[U].fail(error=str(e))
            return FlextModelsResult.RuntimeResult[U].fail(
                error=self.error,
                error_code=self.error_code,
                error_data=self.error_data,
            )

        def map_error(
            self,
            func: Callable[[str], str],
        ) -> FlextModelsResult.RuntimeResult[T]:
            """Transform error message."""
            if not self.is_success:
                return FlextModelsResult.RuntimeResult[T].fail(
                    error=func(self.error or ""),
                    error_code=self.error_code,
                    error_data=self.error_data,
                )
            return self

        def recover(
            self,
            func: Callable[[str], T],
        ) -> FlextModelsResult.RuntimeResult[T]:
            """Recover from failure with fallback value."""
            if not self.is_success:
                fallback_value = func(self.error or "")
                return FlextModelsResult.RuntimeResult[T].ok(value=fallback_value)
            return self

        def tap(self, func: Callable[[T], None]) -> FlextModelsResult.RuntimeResult[T]:
            """Apply side effect to success value, return unchanged."""
            if self.is_success and self._payload is not None:
                func(self._payload)
            return self

        def tap_error(
            self,
            func: Callable[[str], None],
        ) -> FlextModelsResult.RuntimeResult[T]:
            """Apply side effect to error, return unchanged."""
            if not self.is_success:
                func(self.error or "")
            return self

        def unwrap(self) -> T:
            """Unwrap the success value or raise RuntimeError."""
            if not self.is_success:
                msg = f"Cannot unwrap failed result: {self.error}"
                raise RuntimeError(msg)
            return self.value

        def unwrap_or(self, default: T) -> T:
            """Return the success value or the default if failed."""
            if self.is_success:
                return self.value
            return default

        def unwrap_or_else(self, func: Callable[[], T]) -> T:
            """Return the success value or call func if failed."""
            if self.is_success:
                return self.value
            return func()
