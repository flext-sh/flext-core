"""FlextProtocolsResult - result and model-dump contracts.

The public ``p.Result`` contract is nominal for direct static typing, while
auxiliary structural protocols segment the instance API by concern. Today only
``ResultLike`` has a direct structural consumer in the workspace, but the other
protocols still document and organize the full public result surface.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import (
    Callable,
)
from types import TracebackType
from typing import TYPE_CHECKING, Protocol, Self, overload, override, runtime_checkable

from flext_core._models.pydantic import FlextModelsPydantic as mp

if TYPE_CHECKING:
    from flext_core.typings import FlextTypes as t


class FlextProtocolsResult:
    """Protocols for railway result contracts and model dump shape."""

    # ------------------------------------------------------------------
    # Minimal interop contract
    # ------------------------------------------------------------------

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
        def error_data(self) -> t.JsonMapping | None:
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

        @overload
        def unwrap_or(self, default: T) -> T: ...
        @overload
        def unwrap_or[D](self, default: D) -> T | D: ...
        def unwrap_or[D](self, default: D) -> T | D:
            """Return success value or the provided default."""
            ...

        @overload
        def unwrap_or_else(self, func: Callable[[], T]) -> T: ...
        @overload
        def unwrap_or_else[D](self, func: Callable[[], D]) -> T | D: ...
        def unwrap_or_else[D](self, func: Callable[[], D]) -> T | D:
            """Return success value or the result of the fallback callable."""
            ...

    # ------------------------------------------------------------------
    # ISP Sub-Protocol 1: Observable (read-only state inspection)
    # ------------------------------------------------------------------

    @runtime_checkable
    class ResultObservable[T](Protocol):
        """Read-only observation of result state.

        Provides success/failure inspection and access to value, error,
        error_code, error_data, and exception properties.
        T variance is automatically inferred as covariant (output-only).
        """

        @property
        def error(self) -> str | None:
            """Error message (available on failure, None on success)."""
            ...

        @property
        def error_code(self) -> str | None:
            """Error code for categorization (structured error support)."""
            ...

        @property
        def error_data(self) -> t.JsonMapping | None:
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

    # ------------------------------------------------------------------
    # ISP Sub-Protocol 2: Unwrappable (value extraction)
    # ------------------------------------------------------------------

    @runtime_checkable
    class ResultUnwrappable[T](Protocol):
        """Value extraction from result.

        Provides unwrap, unwrap_or, and unwrap_or_else for
        extracting the success value with different failure strategies.
        T variance is automatically inferred as invariant (input + output).
        """

        def unwrap(self) -> T:
            """Unwrap success value (raises materialized exception on failure)."""
            ...

        @overload
        def unwrap_or(self, default: T) -> T: ...
        @overload
        def unwrap_or[D](self, default: D) -> T | D: ...
        def unwrap_or[D](self, default: D) -> T | D:
            """Return success value or the provided default."""
            ...

        @overload
        def unwrap_or_else(self, func: Callable[[], T]) -> T: ...
        @overload
        def unwrap_or_else[D](self, func: Callable[[], D]) -> T | D: ...
        def unwrap_or_else[D](self, func: Callable[[], D]) -> T | D:
            """Return success value or the result of the fallback callable."""
            ...

    # ------------------------------------------------------------------
    # ISP Sub-Protocol 3: Monadic (chaining operations)
    # ------------------------------------------------------------------

    @runtime_checkable
    class ResultMonadic[T](Protocol):
        """Monadic chaining operations on result.

        Provides map, flat_map, fold, and lash for composing
        result-returning operations in a railway-oriented style.
        All return types use the protocol (never the concrete carrier).
        T variance is automatically inferred as invariant (consumed via callbacks).
        """

        def flat_map[U](
            self,
            func: Callable[[T], FlextProtocolsResult.Result[U]],
        ) -> FlextProtocolsResult.Result[U]:
            """Chain operations that return structural FLEXT results."""
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
        ) -> FlextProtocolsResult.Result[T]:
            """Recover from failure using another structural result."""
            ...

        def map[U](
            self,
            func: Callable[[T], U],
        ) -> FlextProtocolsResult.Result[U]:
            """Transform the success value."""
            ...

    # ------------------------------------------------------------------
    # ISP Sub-Protocol 4: Tappable (side-effects & error transforms)
    # ------------------------------------------------------------------

    @runtime_checkable
    class ResultTappable[T](Protocol):
        """Side-effect and error transform operations.

        Provides tap, tap_error, flow_through, map_error, and map_or
        for observing or transforming result state without consuming it.
        T variance is automatically inferred as invariant (consumed via callbacks).
        """

        def flow_through(
            self,
            *funcs: Callable[[T], FlextProtocolsResult.Result[T]],
        ) -> FlextProtocolsResult.Result[T]:
            """Apply multiple Result-returning steps in sequence."""
            ...

        def map_error(
            self,
            func: Callable[[str], str],
        ) -> FlextProtocolsResult.Result[T]:
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

        def tap(
            self,
            func: Callable[[T], None],
        ) -> FlextProtocolsResult.Result[T]:
            """Apply a side effect to the success value."""
            ...

        def tap_error(self, func: Callable[[str], None]) -> Self:
            """Apply a side effect to the failure value."""
            ...

    # ------------------------------------------------------------------
    # ISP Sub-Protocol 5: Recoverable (fallback and predicate handling)
    # ------------------------------------------------------------------

    @runtime_checkable
    class ResultRecoverable[T](Protocol):
        """Fallback and predicate-based result operations."""

        def filter(
            self,
            predicate: Callable[[T], bool],
        ) -> FlextProtocolsResult.Result[T]:
            """Keep the value only when the predicate passes."""
            ...

        def recover[U](
            self,
            func: Callable[[str], U],
        ) -> FlextProtocolsResult.Result[T | U]:
            """Recover a failure into a success value."""
            ...

    # ------------------------------------------------------------------
    # ISP Sub-Protocol 6: Convertible (model and type adaptation)
    # ------------------------------------------------------------------

    @runtime_checkable
    class ResultConvertible[T](Protocol):
        """Conversion helpers from raw result payloads to validated types."""

        def to_model[U: mp.BaseModel](
            self,
            model: type[U],
        ) -> FlextProtocolsResult.Result[U]:
            """Convert the success payload into a validated model."""
            ...

    # ------------------------------------------------------------------
    # Facade: Result nominal contract (direct typing)
    # ------------------------------------------------------------------

    class Result[T_co](ABC):
        """Nominal public result contract for direct static typing across FLEXT.

        Pure ABC (not inheriting ResultLike Protocol) so Pydantic BaseModel
        subclasses (FlextResult) can inherit without metaclass conflict
        between ``_ProtocolMeta`` and ``ModelMetaclass``. ResultLike remains
        available as a @runtime_checkable Protocol for structural isinstance.
        """

        @property
        @abstractmethod
        def error(self) -> str | None: ...

        @property
        @abstractmethod
        def error_code(self) -> str | None: ...

        @property
        @abstractmethod
        def error_data(self) -> t.JsonMapping | None: ...

        @property
        @abstractmethod
        def success(self) -> bool: ...

        @property
        @abstractmethod
        def exception(self) -> BaseException | None: ...

        @property
        @abstractmethod
        def failure(self) -> bool: ...

        @property
        @abstractmethod
        def value(self) -> T_co: ...

        @abstractmethod
        def __enter__(self) -> Self: ...

        @abstractmethod
        def __exit__(
            self,
            _exc_type: type[BaseException] | None,
            _exc_val: BaseException | None,
            _exc_tb: TracebackType | None,
        ) -> None: ...

        @overload
        def __or__(self, default: T_co) -> T_co: ...

        @overload
        def __or__[D](self, default: D) -> T_co | D: ...

        @abstractmethod
        def __or__[D](self, default: D) -> T_co | D: ...

        @abstractmethod
        def unwrap(self) -> T_co: ...

        @abstractmethod
        def unwrap_or[D](self, default: D) -> T_co | D: ...

        @abstractmethod
        def unwrap_or_else[D](self, func: Callable[[], D]) -> T_co | D: ...

        @abstractmethod
        def flat_map[U](
            self,
            func: Callable[[T_co], FlextProtocolsResult.Result[U]],
        ) -> FlextProtocolsResult.Result[U]: ...

        @abstractmethod
        def fold[U](
            self,
            on_failure: Callable[[str], U],
            on_success: Callable[[T_co], U],
        ) -> U: ...

        @abstractmethod
        def lash(
            self,
            func: Callable[[str], FlextProtocolsResult.Result[T_co]],
        ) -> FlextProtocolsResult.Result[T_co]: ...

        @abstractmethod
        def map[U](
            self,
            func: Callable[[T_co], U],
        ) -> FlextProtocolsResult.Result[U]: ...

        @abstractmethod
        def flow_through(
            self,
            *funcs: Callable[[T_co], FlextProtocolsResult.Result[T_co]],
        ) -> FlextProtocolsResult.Result[T_co]: ...

        @abstractmethod
        def map_error(
            self,
            func: Callable[[str], str],
        ) -> FlextProtocolsResult.Result[T_co]: ...

        @overload
        def map_or(self, default: None, func: None = None) -> T_co | None: ...
        @overload
        def map_or[U](self, default: U, func: None = None) -> T_co | U: ...
        @overload
        def map_or[U](self, default: U, func: Callable[[T_co], U]) -> U: ...

        @abstractmethod
        def map_or[U](
            self,
            default: U,
            func: Callable[[T_co], U] | None = None,
        ) -> U | T_co: ...

        @abstractmethod
        def tap(
            self,
            func: Callable[[T_co], None],
        ) -> FlextProtocolsResult.Result[T_co]: ...

        @abstractmethod
        def tap_error(self, func: Callable[[str], None]) -> Self: ...

        @abstractmethod
        def filter(
            self,
            predicate: Callable[[T_co], bool],
        ) -> FlextProtocolsResult.Result[T_co]: ...

        @abstractmethod
        def recover[U](
            self,
            func: Callable[[str], U],
        ) -> FlextProtocolsResult.Result[T_co | U]: ...

        @abstractmethod
        def to_model[U: mp.BaseModel](
            self,
            model: type[U],
        ) -> FlextProtocolsResult.Result[U]: ...

        @abstractmethod
        def __bool__(self) -> bool: ...

        @override
        def __repr__(self) -> str:
            """Default repr for Result ABC."""
            return f"{type(self).__name__}(success={self.success})"

    # ------------------------------------------------------------------
    # Auxiliary protocols (unchanged)
    # ------------------------------------------------------------------

    @runtime_checkable
    class HasModelDump(Protocol):
        """Protocol for items that can dump model data.

        Used for Pydantic model compatibility and serialization.
        """

        def model_dump(self) -> t.MappingKV[str, t.JsonPayload | None]:
            """Dump model data to a mapping that runtime helpers can normalize."""
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

        Lighter than Result — requires only success/failure status properties.
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
