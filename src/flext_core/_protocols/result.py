"""FlextProtocolsResult - result and model-dump protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from types import TracebackType
from typing import Protocol, override, runtime_checkable

from flext_core._protocols.base import FlextProtocolsBase
from flext_core.typings import FlextTypes as t


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
            """Error code for categorization."""
            ...

        @property
        def error_data(self) -> t.ConfigMap | None:
            """Error metadata (optional)."""
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
        def result(self) -> FlextProtocolsResult.Result[T]:
            """Access internal Result for advanced operations."""
            ...

        @property
        def value(self) -> T:
            """Result value (available on success, never None)."""
            ...

        @classmethod
        def accumulate_errors[TItem](
            cls, *results: FlextProtocolsResult.Result[TItem]
        ) -> FlextProtocolsResult.Result[Sequence[TItem]]:
            """Collect all successes, fail if any failure."""
            ...

        @classmethod
        def create_from_callable(
            cls, func: Callable[[], T], error_code: str | None = None
        ) -> FlextProtocolsResult.Result[T]:
            """Create result from callable, catching exceptions."""
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
            self, predicate: Callable[[T], bool]
        ) -> FlextProtocolsResult.Result[T]:
            """Filter success value using predicate.

            Returns self if predicate passes or result is failure,
            new failed Result if predicate fails.
            """
            ...

        def flat_map[U](
            self, func: Callable[[T], FlextProtocolsResult.Result[U]]
        ) -> FlextProtocolsResult.Result[U]:
            """Chain operations returning Result."""
            ...

        def flow_through[U](
            self, *funcs: Callable[[T | U], FlextProtocolsResult.Result[U]]
        ) -> FlextProtocolsResult.Result[U]:
            """Chain multiple operations in a pipeline."""
            ...

        def lash(
            self, func: Callable[[str], FlextProtocolsResult.Result[T]]
        ) -> FlextProtocolsResult.Result[T]:
            """Apply recovery function on failure."""
            ...

        def map[U](self, func: Callable[[T], U]) -> FlextProtocolsResult.Result[U]:
            """Transform success value using function."""
            ...

        def map_error(
            self, func: Callable[[str], str]
        ) -> FlextProtocolsResult.Result[T]:
            """Transform error message on failure.

            Returns self on success, new Result with transformed error on failure.
            """
            ...

        def unwrap(self) -> T:
            """Unwrap success value (raises on failure)."""
            ...

        def unwrap_or(self, default: T) -> T:
            """Unwrap success value or return default on failure."""
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
    class HasModelFields(HasModelDump, Protocol):
        """Protocol for items with model fields.

        Extends HasModelDump with model fields access.
        Used for Pydantic model introspection.
        """

        @property
        def model_fields(self) -> Mapping[str, t.Scalar]:
            """Model fields mapping."""
            ...


__all__ = ["FlextProtocolsResult"]
