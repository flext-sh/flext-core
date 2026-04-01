"""FlextProtocolsResult - result and model-dump protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import TracebackType
from typing import TYPE_CHECKING, Protocol, Self, runtime_checkable

from flext_core._protocols.base import FlextProtocolsBase
from flext_core.typings import t

if TYPE_CHECKING:
    from flext_core._protocols.logging import FlextProtocolsLogging


class FlextProtocolsResult:
    """Protocols for railway result contracts and model dump shape."""

    @runtime_checkable
    class Result[T_co](FlextProtocolsBase.Base, Protocol):
        """Observable result contract for structural interop across FLEXT.

        ``FlextResult`` is a ``BaseModel`` carrier, so this protocol intentionally
        models only the stable observation/extraction surface consumed across
        layers. Recursive monadic combinators stay on the concrete carrier to
        avoid unsound higher-kinded structural recursion in type checkers.
        """

        def __bool__(self) -> bool:
            """Boolean conversion based on success state."""
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
        def value(self) -> T_co:
            """Result value (available on success, strictly typed as T)."""
            ...

        @property
        def result_logger(self) -> FlextProtocolsLogging.Logger:
            """Logger used by the result implementation."""
            ...

        def unwrap(self) -> T_co:
            """Unwrap success value (raises RuntimeError on failure)."""
            ...

    @runtime_checkable
    class HasModelDump(Protocol):
        """Protocol for items that can dump model data.

        Used for Pydantic model compatibility and serialization.
        """

        def model_dump(self) -> t.ScalarMapping:
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

        def is_error_domain(self, domain: str) -> bool:
            """Check if error belongs to a specific domain."""
            ...

    @runtime_checkable
    class SuccessCheckable(Protocol):
        """Protocol for any model with success/failure outcome semantics.

        Lighter than Result[T] — requires only success/failure status properties.
        Satisfied by RuntimeResult, FlextResult, BatchResult, HTTP response models,
        and any domain model that reports pass/fail status.
        """

        @property
        def is_success(self) -> bool:
            """True when the operation succeeded."""
            ...

        @property
        def is_failure(self) -> bool:
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


__all__ = ["FlextProtocolsResult"]
