"""FlextProtocolsResult - result and model-dump protocols.

ISP-compliant split: 4 sub-protocols composed via MRO into Result facade.
PEP 695 type parameters with automatic variance inference per protocol:
- ResultObservable[T]: covariant (T in output positions only)
- ResultUnwrappable[T]: invariant (T in both output and input positions)
- ResultMonadic[T]: invariant (T consumed via callbacks)
- ResultTappable[T]: invariant (T consumed via callbacks)
- Result[T]: invariant (composed facade)

Factories, type guards, context manager, and low-usage methods live ONLY
on the concrete carrier (FlextResult), never on the protocol.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable
from typing import (
    Protocol,
    Self,
    overload,
    override,
    runtime_checkable,
)

from flext_core._protocols.base import FlextProtocolsBase
from flext_core._typings.base import FlextTypingBase
from flext_core._typings.containers import FlextTypingContainers


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
    # Facade: Result composed via MRO (ISP-compliant)
    # ------------------------------------------------------------------

    @runtime_checkable
    class Result[T](
        ResultObservable[T],
        ResultUnwrappable[T],
        ResultMonadic[T],
        ResultTappable[T],
        FlextProtocolsBase.Base,
        Protocol,
    ):
        """Observable result contract for structural interop across FLEXT.

        Composed via MRO from 4 ISP sub-protocols:
        - ``ResultObservable``: read-only state inspection (covariant T)
        - ``ResultUnwrappable``: value extraction (invariant T)
        - ``ResultMonadic``: map / flat_map / fold / lash (invariant T)
        - ``ResultTappable``: tap / tap_error / flow_through / map_error / map_or (invariant T)

        T variance is automatically inferred as invariant (union of all sub-protocols).

        Factories (ok, fail, fail_exc, fail_op, ...), context manager,
        ``__or__``, type transforms, type guards, filter, recover, and safe
        live ONLY on the concrete carrier ``FlextResult[T]``.
        """

        def __bool__(self) -> bool:
            """Boolean conversion based on success state."""
            ...

        @override
        def __repr__(self) -> str:
            """Human-readable representation."""
            ...

    # ------------------------------------------------------------------
    # Auxiliary protocols (unchanged)
    # ------------------------------------------------------------------

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
