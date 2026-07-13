"""FlextProtocolsResult - result and model-dump contracts.

The public ``p.Result`` contract is nominal for direct static typing, while
auxiliary structural protocols segment the instance API by concern. Today only
``ResultLike`` has a direct structural consumer in the workspace, but the other
protocols still document and organize the full public result surface.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, overload, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
    )

    from flext_core import FlextTypes as t


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


__all__: list[str] = ["FlextProtocolsResult"]
