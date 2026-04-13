"""FlextProtocolsResult - result and model-dump protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from types import TracebackType
from typing import (
    TYPE_CHECKING,
    Protocol,
    Self,
    TypeIs,
    overload,
    override,
    runtime_checkable,
)

from pydantic import TypeAdapter

from flext_core._protocols.base import FlextProtocolsBase
from flext_core._typings.base import FlextTypingBase
from flext_core._typings.containers import FlextTypingContainers

if TYPE_CHECKING:
    from flext_core._typings.services import FlextTypesServices
    from flext_core.result import FlextResult


class FlextProtocolsResult:
    """Protocols for railway result contracts and model dump shape."""

    @runtime_checkable
    class ResultLike[T](Protocol):
        """Minimal observable result contract for automatic narrowing."""

        @property
        def error(self) -> str | None:
            """Error message (available on failure, None on success)."""
            ...

        @property
        def error_code(self) -> str | None:
            """Structured error code when available."""
            ...

        @property
        def error_data(self) -> FlextTypingContainers.ConfigMap | None:
            """Structured error metadata when available."""
            ...

        @property
        def exception(self) -> BaseException | None:
            """Captured exception when available."""
            ...

        @property
        def failure(self) -> bool:
            """Failure status (strict: not success)."""
            ...

        @property
        def success(self) -> bool:
            """Success status (strict: True only when operation succeeded)."""
            ...

        @property
        def value(self) -> T:
            """Result value (available on success, strictly typed as T)."""
            ...

        def unwrap(self) -> T:
            """Unwrap success value (raises on failure)."""
            ...

        def unwrap_or(self, default: T) -> T:
            """Return success value or the provided default."""
            ...

        def unwrap_or_else(self, func: Callable[[], T]) -> T:
            """Return success value or the result of the fallback callable."""
            ...

    @runtime_checkable
    class Result[T](FlextProtocolsBase.Base, Protocol):
        """Observable result contract for structural interop across FLEXT.

        ``FlextResult`` is a ``BaseModel`` carrier, so this protocol intentionally
        models the stable public DSL consumed across layers.
        """

        def __bool__(self) -> bool:
            """Boolean conversion based on success state."""
            ...

        @override
        def __repr__(self) -> str:
            """Human-readable representation."""
            ...

        def __enter__(self) -> Self:
            """Context manager entry."""
            ...

        def __exit__(
            self,
            _exc_type: type[BaseException] | None,
            _exc_val: BaseException | None,
            _exc_tb: TracebackType | None,
        ) -> None:
            """Context manager exit."""
            ...

        def __or__(self, default: T) -> T:
            """Return success value or the provided default."""
            ...

        @property
        def error(self) -> str | None:
            """Error message (available on failure, None on success)."""
            ...

        @property
        def error_code(self) -> str | None:
            """Error code for categorization (structured error support)."""
            ...

        @property
        def error_data(self) -> FlextTypingContainers.ConfigMap | None:
            """Error metadata with structured error context (optional)."""
            ...

        @property
        def exception(self) -> BaseException | None:
            """Exception captured during operation (if any)."""
            ...

        @property
        def failure(self) -> bool:
            """Failure status (strict: not success)."""
            ...

        @property
        def success(self) -> bool:
            """Success status (strict: True only when operation succeeded)."""
            ...

        @property
        def value(self) -> T:
            """Result value (available on success, strictly typed as T)."""
            ...

        def unwrap(self) -> T:
            """Unwrap success value (raises RuntimeError on failure)."""
            ...

        def unwrap_or(self, default: T) -> T:
            """Return success value or the provided default."""
            ...

        def unwrap_or_else(self, func: Callable[[], T]) -> T:
            """Return success value or the result of the fallback callable."""
            ...

        def unwrap_model[U: FlextTypesServices.ModelCarrier](
            self,
            model: FlextTypesServices.ModelClass[U],
        ) -> U:
            """Unwrap success value after model conversion."""
            ...

        def unwrap_type[U](self, adapter: TypeAdapter[U]) -> U:
            """Unwrap success value after TypeAdapter conversion."""
            ...

        def filter(self, predicate: Callable[[T], bool]) -> Self:
            """Filter success value by predicate."""
            ...

        def flat_map[U](
            self,
            func: Callable[[T], FlextProtocolsResult.Result[U]],
        ) -> FlextResult[U]:
            """Chain operations that return structural FLEXT results."""
            ...

        def flow_through[U](
            self,
            *funcs: Callable[[T | U], FlextProtocolsResult.Result[U]],
        ) -> FlextResult[T] | FlextResult[U]:
            """Apply multiple Result-returning steps in sequence."""
            ...

        def fold[U](
            self,
            on_failure: Callable[[str], U],
            on_success: Callable[[T], U],
        ) -> U:
            """Reduce result into a single value."""
            ...

        def lash(
            self,
            func: Callable[[str], FlextProtocolsResult.Result[T]],
        ) -> FlextResult[T]:
            """Recover from failure using another structural result."""
            ...

        def map[U](self, func: Callable[[T], U]) -> FlextResult[U]:
            """Transform the success value."""
            ...

        def map_error(self, func: Callable[[str], str]) -> FlextResult[T]:
            """Transform the failure message."""
            ...

        @overload
        def map_or(self, default: None, func: None = None) -> T | None: ...

        @overload
        def map_or[U](self, default: U, func: None = None) -> T | U: ...

        @overload
        def map_or[U](self, default: U, func: Callable[[T], U]) -> U: ...

        def map_or[U](
            self,
            default: U,
            func: Callable[[T], U] | None = None,
        ) -> U | T:
            """Map success value or return default."""
            ...

        def recover(self, func: Callable[[str], T]) -> FlextResult[T]:
            """Recover from failure with a fallback value."""
            ...

        def tap(self, func: Callable[[T], None]) -> FlextResult[T]:
            """Apply a side effect to the success value."""
            ...

        def tap_error(self, func: Callable[[str], None]) -> Self:
            """Apply a side effect to the failure value."""
            ...

        def to_model[U: FlextTypesServices.ModelCarrier](
            self,
            model: FlextTypesServices.ModelClass[U],
        ) -> FlextResult[U]:
            """Convert success value to a model."""
            ...

        def to_type[U](self, adapter: TypeAdapter[U]) -> FlextResult[U]:
            """Convert success value using a TypeAdapter."""
            ...

        @classmethod
        def accumulate_errors[ValueT](
            cls,
            *results: FlextProtocolsResult.Result[ValueT],
        ) -> FlextProtocolsResult.Result[Sequence[ValueT]]:
            """Collect successes or combine failures."""
            ...

        @classmethod
        def create_from_callable[V](
            cls,
            func: Callable[[], V | None],
            error_code: str | None = None,
        ) -> FlextProtocolsResult.Result[V]:
            """Create result from callable, catching exceptions."""
            ...

        @classmethod
        def fail(
            cls,
            error: str | None,
            error_code: str | None = None,
            error_data: FlextTypesServices.ResultErrorData
            | FlextTypesServices.ConfigModelInput
            | None = None,
            *,
            exception: BaseException | None = None,
        ) -> Self:
            """Create failed result."""
            ...

        @classmethod
        def from_exception(
            cls,
            exception: BaseException,
            *,
            error: str | None = None,
            error_code: str | None = None,
            error_data: FlextTypesServices.ResultErrorData
            | FlextTypesServices.ConfigModelInput
            | None = None,
        ) -> Self:
            """Create failed result from exception."""
            ...

        @classmethod
        def fail_exc(cls, exc: BaseException) -> Self:
            """Create failed result from BaseException."""
            ...

        @classmethod
        def fail_op(
            cls,
            operation: str,
            exc: Exception | str | None = None,
            *,
            error_code: str | None = None,
        ) -> Self:
            """Create failed result for an operation."""
            ...

        @classmethod
        def from_result[V](
            cls,
            source: FlextProtocolsResult.Result[V],
        ) -> FlextProtocolsResult.Result[V]:
            """Normalize any structural ResultLike into Result."""
            ...

        @classmethod
        def from_validation[ModelT: FlextTypesServices.ModelCarrier](
            cls,
            data: FlextTypesServices.ModelInput,
            model: FlextTypesServices.ModelClass[ModelT],
        ) -> FlextProtocolsResult.Result[ModelT]:
            """Create result from model validation."""
            ...

        @classmethod
        def ok(cls, value: T) -> Self:
            """Create successful result."""
            ...

        @classmethod
        def traverse[V, U](
            cls,
            items: Sequence[V],
            func: Callable[[V], FlextProtocolsResult.Result[U]],
            *,
            fail_fast: bool = True,
        ) -> FlextProtocolsResult.Result[Sequence[U]]:
            """Traverse a sequence with Result-returning function."""
            ...

        @classmethod
        def with_resource[R, U](
            cls,
            factory: Callable[[], R],
            op: Callable[[R], FlextProtocolsResult.Result[U]],
            cleanup: Callable[[R], None] | None = None,
        ) -> FlextProtocolsResult.Result[U]:
            """Manage resource with optional cleanup."""
            ...

        @staticmethod
        def failed_result(
            value: FlextTypesServices.ProtocolSubject,
        ) -> TypeIs[FlextProtocolsResult.Result[FlextTypingBase.RecursiveContainer]]:
            """Return True when value is a failed result."""
            ...

        @staticmethod
        def successful_result(
            value: FlextTypesServices.ProtocolSubject,
        ) -> TypeIs[FlextProtocolsResult.Result[FlextTypingBase.RecursiveContainer]]:
            """Return True when value is a successful result."""
            ...

        @staticmethod
        def safe[U, **PFunc](
            func: Callable[PFunc, U],
        ) -> Callable[PFunc, FlextProtocolsResult.Result[U]]:
            """Wrap callable, returning Result on exceptions."""
            ...

    @runtime_checkable
    class HasModelDump(Protocol):
        """Protocol for items that can dump model data.

        Used for Pydantic model compatibility and serialization.
        """

        def model_dump(self) -> FlextTypingBase.ScalarMapping:
            """Dump model data to dictionary."""
            ...

    @runtime_checkable
    class StructuredError(Protocol):
        """Protocol for structured error handling in Results."""

        @property
        def error_domain(self) -> str | None:
            """Error domain category (e.g., 'VALIDATION', 'NETWORK', 'AUTH')."""
            ...

        @property
        def error_code(self) -> str | None:
            """Specific error code for routing and categorization."""
            ...

        @property
        def error_message(self) -> str | None:
            """Human-readable error message."""
            ...

        def matches_error_domain(self, domain: str) -> bool:
            """Whether the error belongs to a specific domain."""
            ...

    @runtime_checkable
    class SuccessCheckable(Protocol):
        """Protocol for any model with success/failure outcome semantics.

        Lighter than Result[T] — requires only success/failure status properties.
        Satisfied by RuntimeResult, FlextResult, BatchResult, HTTP response models,
        and any domain model that reports pass/fail status.
        """

        @property
        def success(self) -> bool:
            """True when the operation succeeded."""
            ...

        @property
        def failure(self) -> bool:
            """True when the operation failed."""
            ...

    @runtime_checkable
    class ErrorDomainProtocol(Protocol):
        """Protocol for error domain enumeration.

        Defines standard error categories for structured error handling
        across FLEXT. Enables strict error routing and categorization.
        """

        value: str  # e.g., "VALIDATION", "NETWORK", "AUTH"
        name: str  # e.g., "ValidationError", "NetworkError"


__all__: list[str] = ["FlextProtocolsResult"]
