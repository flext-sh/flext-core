"""FlextProtocolsResult - result and model-dump protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from types import TracebackType
from typing import Protocol, override, runtime_checkable

from flext_core import FlextTypes as t


class FlextProtocolsResult:
    """Protocols for railway result contracts and model dump shape."""

    @runtime_checkable
    class Result[T](FlextProtocolsBase.Base, Protocol):
        """Result type interface for railway-oriented programming.

        Used extensively for all operations that can fail. Provides
        structural typing interface for r without circular imports.
        Fully compatible with r and FlextRuntime usage patterns.

        Defined at root level to allow self-referencing in method signatures
        (e.g., `def map[U](...) -> FlextProtocolsResult.Result[U]`).

        All methods enforce strict typing with explicit return types and
        comprehensive error handling via structured error support.
        """

        @classmethod
        @override
        def __subclasshook__(cls, cls_: type) -> bool:
            """Enable isinstance() for Pydantic-backed implementations.

            Python 3.12+ Protocol isinstance checks use class __dict__ lookup,
            which misses Pydantic v2 model fields (stored in __pydantic_fields__,
            not __dict__). This hook uses class-level attrs that ARE in __dict__.
            """
            if cls is FlextProtocolsResult.Result:
                required = frozenset({"is_failure", "value", "flat_map", "lash"})
                if all(any(a in B.__dict__ for B in cls_.__mro__) for a in required):
                    return True
            return NotImplemented

        @override
        def __repr__(self) -> str:
            """String representation."""
            ...

        def __bool__(self) -> bool:
            """Boolean conversion based on success state."""
            ...

        def __enter__(self) -> FlextProtocolsResult.Result[T]:
            """Context manager entry."""
            ...

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_val: BaseException | None,
            exc_tb: TracebackType | None,
        ) -> None:
            """Context manager exit."""
            ...

        def __or__(self, default: T) -> T:
            """Operator overload for default values."""
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
        def error_data(self) -> t.ConfigMap | None:
            """Error metadata with structured error context (optional)."""
            ...

        @property
        def exception(self) -> BaseException | None:
            """Exception captured during operation (if any)."""
            ...

        @property
        def is_failure(self) -> bool:
            """Failure status (strict: not is_success)."""
            ...

        @property
        def is_success(self) -> bool:
            """Success status (strict: True only when operation succeeded)."""
            ...

        @property
        def result(self) -> FlextProtocolsResult.Result[T]:
            """Access internal Result for advanced operations."""
            ...

        @property
        def value(self) -> T:
            """Result value (available on success, strictly typed as T)."""
            ...

        @classmethod
        def accumulate_errors[TItem](
            cls,
            *results: FlextProtocolsResult.Result[TItem],
        ) -> FlextProtocolsResult.Result[Sequence[TItem]]:
            """Collect all successes, fail if any failure with combined errors."""
            ...

        @classmethod
        def create_from_callable(
            cls,
            func: Callable[[], T],
            error_code: str | None = None,
        ) -> FlextProtocolsResult.Result[T]:
            """Create result from callable, catching exceptions and returning structured errors."""
            ...

        @classmethod
        def fail(
            cls,
            error: str | None,
            error_code: str | None = None,
            error_data: t.ConfigMap | None = None,
            *,
            exception: BaseException | None = None,
        ) -> FlextProtocolsResult.Result[T]:
            """Create failed result with error message and structured error support."""
            ...

        @classmethod
        def ok(cls, value: T) -> FlextProtocolsResult.Result[T]:
            """Create successful result wrapping value (strict: value must be non-None unless T includes None)."""
            ...

        @classmethod
        def traverse[TItem, UResult](
            cls,
            items: Sequence[TItem],
            func: Callable[[TItem], FlextProtocolsResult.Result[UResult]],
            *,
            fail_fast: bool = True,
        ) -> FlextProtocolsResult.Result[Sequence[UResult]]:
            """Map over sequence with configurable failure handling."""
            ...

        def filter(
            self,
            predicate: Callable[[T], bool],
        ) -> FlextProtocolsResult.Result[T]:
            """Filter success value using predicate.

            Returns self if predicate passes or result is failure,
            new failed Result if predicate fails.
            """
            ...

        def flat_map[U](
            self,
            func: Callable[[T], FlextProtocolsResult.Result[U]],
        ) -> FlextProtocolsResult.Result[U]:
            """Chain operations returning Result (monadic bind)."""
            ...

        def flow_through[U](
            self,
            *funcs: Callable[[T | U], FlextProtocolsResult.Result[U]],
        ) -> FlextProtocolsResult.Result[U]:
            """Chain multiple operations in a pipeline (sequential composition)."""
            ...

        def fold[U](
            self,
            on_failure: Callable[[str], U],
            on_success: Callable[[T], U],
        ) -> U:
            """Catamorphism - reduce result to single value (strict typing)."""
            ...

        def lash(
            self,
            func: Callable[[str], FlextProtocolsResult.Result[T]],
        ) -> FlextProtocolsResult.Result[T]:
            """Apply recovery function on failure (monadic catch)."""
            ...

        def map[U](self, func: Callable[[T], U]) -> FlextProtocolsResult.Result[U]:
            """Transform success value using function (monadic map)."""
            ...

        def map_error(
            self,
            func: Callable[[str], str],
        ) -> FlextProtocolsResult.Result[T]:
            """Transform error message on failure.

            Returns self on success, new Result with transformed error on failure.
            """
            ...

        def recover(
            self,
            func: Callable[[str], T],
        ) -> FlextProtocolsResult.Result[T]:
            """Recover from failure with fallback value (safe alternative to lash)."""
            ...

        def tap(self, func: Callable[[T], None]) -> FlextProtocolsResult.Result[T]:
            """Apply side effect to success value, return unchanged."""
            ...

        def tap_error(
            self,
            func: Callable[[str], None],
        ) -> FlextProtocolsResult.Result[T]:
            """Apply side effect to error on failure, return unchanged."""
            ...

        def unwrap(self) -> T:
            """Unwrap success value (raises RuntimeError on failure)."""
            ...

        def unwrap_or(self, default: T) -> T:
            """Unwrap success value or return default on failure (safe alternative to unwrap)."""
            ...

    @runtime_checkable
    class ResultLike[T_co](Protocol):
        """Result-like protocol for compatibility with r operations.

        Used for type compatibility when working with result-like items.
        """

        @property
        def error(self) -> str | None:
            """Error message."""
            ...

        @property
        def error_code(self) -> str | None:
            """Error code."""
            ...

        @property
        def error_data(self) -> t.ConfigMap | None:
            """Error data."""
            ...

        @property
        def exception(self) -> BaseException | None:
            """Exception captured during operation (if any)."""
            ...

        @property
        def is_failure(self) -> bool:
            """Failure status."""
            ...

        @property
        def is_success(self) -> bool:
            """Success status."""
            ...

        @property
        def value(self) -> T_co:
            """Result value."""
            ...

        def unwrap(self) -> T_co:
            """Unwrap value."""
            ...

    @runtime_checkable
    class HasModelDump(Protocol):
        """Protocol for items that can dump model data.

        Used for Pydantic model compatibility and serialization.
        """

        def model_dump(self) -> Mapping[str, t.Scalar]:
            """Dump model data to dictionary."""
            ...

    @runtime_checkable
    class StructuredError(Protocol):
        """Protocol for structured error handling in Results.

        Enables categorized error handling with error domains instead of
        free-form error strings. Supports error routing and categorization
        across the FLEXT ecosystem.

        Usage:
            result = r[User].fail(
                "User not found",
                error_code="NOT_FOUND",
                error_data={"user_id": "123"},
            )
            if result.error_code == "NOT_FOUND":
                return 404
        """

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

        def is_error_domain(self, domain: str) -> bool:
            """Check if error belongs to a specific domain."""
            ...

    @runtime_checkable
    class ErrorDomain(Protocol):
        """Protocol for error domain enumeration.

        Defines standard error categories for structured error handling
        across FLEXT. Enables strict error routing and categorization.
        """

        value: str  # e.g., "VALIDATION", "NETWORK", "AUTH"
        name: str  # e.g., "ValidationError", "NetworkError"


__all__ = ["FlextProtocolsResult"]
