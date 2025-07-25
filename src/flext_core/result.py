"""FlextResult - Enterprise Result Pattern Implementation.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

Implementation of the Result pattern for type-safe error handling.
Provides the foundational error handling mechanism for FLEXT ecosystem
projects with functional programming paradigms.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import cast
from typing import final

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator

if TYPE_CHECKING:
    from collections.abc import Callable

# Generic type variable for result data payload
T = TypeVar("T")
U = TypeVar("U")


@final
class FlextResult[T](BaseModel):
    """Result pattern for type-safe error handling.

    Represents the outcome of operations that may succeed or fail,
    encapsulating either a success value or error information.
    Serves as the foundational error handling mechanism throughout
    the FLEXT ecosystem.
    """

    model_config = ConfigDict(
        frozen=True,
        str_strip_whitespace=True,
        validate_assignment=True,
        arbitrary_types_allowed=False,
        extra="forbid",
    )

    success: bool = Field(
        description="Indicates whether the operation completed successfully",
    )
    data: T | None = Field(
        default=None,
        description="The operation result data on success",
    )
    error: str | None = Field(
        default=None,
        description="Detailed error message on failure",
    )

    @field_validator("success", "data", "error")
    @classmethod
    def validate_result_consistency(
        cls,
        value: object,
        info: object,
    ) -> object:
        """Validate result state consistency and data integrity."""
        # Handle ValidationInfo type safely
        if hasattr(info, "data"):
            values = info.data
        else:
            return value

        min_required_fields = 2
        if len(values) < min_required_fields:
            return value

        if all(field in values for field in ["success", "data", "error"]):
            success = values.get("success")
            data = values.get("data")
            error = values.get("error")

            if success and error is not None:
                msg = "Success result cannot contain error message"
                raise ValueError(msg)
            if not success and data is not None:
                msg = "Failure result cannot contain data payload"
                raise ValueError(msg)
            if not success and not error:
                msg = "Failure result must contain descriptive error message"
                raise ValueError(msg)

        return value

    @classmethod
    def ok(cls, data: T) -> FlextResult[T]:
        """Create a successful result with operation data."""
        return cls(success=True, data=data, error=None)

    @classmethod
    def fail(cls, error: str) -> FlextResult[T]:
        """Create a failure result with descriptive error message."""
        if not error or not error.strip():
            error = "Unknown error occurred"
        return cls(success=False, data=None, error=error.strip())

    # Note: Compatibility creation functions available as module functions

    def __bool__(self) -> bool:
        """Enable Pythonic boolean evaluation for result checking."""
        return self.success

    @property
    def is_success(self) -> bool:
        """Check if the operation completed successfully."""
        return self.success

    @property
    def is_failure(self) -> bool:
        """Check if the operation failed."""
        return not self.success

    # ===== COMPATIBILITY ALIASES =====
    # Multiple ways to check success/failure for library compatibility

    @property
    def is_ok(self) -> bool:
        """Alias for is_success - compatibility."""
        return self.success

    @property
    def is_fail(self) -> bool:
        """Alias for is_failure - compatibility."""
        return not self.success

    @property
    def is_err(self) -> bool:
        """Rust-style error check - compatibility."""
        return not self.success

    @property
    def successful(self) -> bool:
        """Property-style success check - compatibility."""
        return self.success

    @property
    def failed(self) -> bool:
        """Property-style failure check - compatibility."""
        return not self.success

    @property
    def failure(self) -> bool:
        """Property-style failure check - compatibility."""
        return not self.success

    @property
    def ok_value(self) -> T | None:
        """Direct access to success value or None - Rust style."""
        return self.data if self.success else None

    @property
    def err_value(self) -> str | None:
        """Direct access to error value or None - Rust style."""
        return self.error if not self.success else None

    def unwrap(self) -> T:
        """Extract the success value or raise an exception."""
        if not self.success:
            msg = f"Cannot unwrap failure result: {self.error}"
            raise ValueError(msg)
        return self.data  # type: ignore[return-value]

    def unwrap_or(self, default: T) -> T:
        """Extract the success value or return a provided default."""
        if self.success:
            return self.data  # type: ignore[return-value]
        return default

    def map(self, func: Callable[[T], U]) -> FlextResult[U]:
        """Transform success value using a provided function."""
        if self.success and self.data is not None:
            try:
                return FlextResult.ok(func(self.data))
            except (TypeError, AttributeError, ValueError, IndexError) as e:
                return FlextResult.fail(f"Transformation failed: {e}")
            except Exception as e:
                return FlextResult.fail(
                    f"Unexpected transformation error: {e}",
                )
        # Return failure as FlextResult[U] (type ignore since failure has no
        # data)
        return FlextResult.fail(self.error or "Unknown error")

    def flat_map(
        self,
        func: Callable[[T], FlextResult[U]],
    ) -> FlextResult[U]:
        """Chain operations that return FlextResult instances."""
        if self.success and self.data is not None:
            try:
                return func(self.data)
            except (TypeError, AttributeError, ValueError, IndexError) as e:
                return FlextResult.fail(f"Chained operation failed: {e}")
            except Exception as e:
                return FlextResult.fail(f"Unexpected chaining error: {e}")
        # Return failure as FlextResult[U]
        return FlextResult.fail(self.error or "Unknown error")

    def then(self, func: Callable[[T], FlextResult[U]]) -> FlextResult[U]:
        """Alias for flat_map with better naming for chaining operations."""
        return self.flat_map(func)

    def bind(self, func: Callable[[T], FlextResult[U]]) -> FlextResult[U]:
        """Monadic bind operation for functional programming style."""
        return self.flat_map(func)

    def or_else(self, default_result: FlextResult[T]) -> FlextResult[T]:
        """Return this result if success, otherwise return default result."""
        return self if self.success else default_result

    def or_else_get(
        self,
        func: Callable[[], FlextResult[T]],
    ) -> FlextResult[T]:
        """Return this result if success, otherwise call function."""
        return self if self.success else func()

    def recover(self, func: Callable[[str], T]) -> FlextResult[T]:
        """Transform error into success using recovery function."""
        if self.success:
            return self
        try:
            return FlextResult.ok(func(self.error or "Unknown error"))
        except Exception as e:
            return FlextResult.fail(f"Recovery failed: {e}")

    def recover_with(
        self,
        func: Callable[[str], FlextResult[T]],
    ) -> FlextResult[T]:
        """Transform error into another result using recovery function."""
        if self.success:
            return self
        try:
            return func(self.error or "Unknown error")
        except Exception as e:
            return FlextResult.fail(f"Recovery failed: {e}")

    def tap(self, func: Callable[[T], None]) -> FlextResult[T]:
        """Execute side effect on success value without changing result."""
        if self.success and self.data is not None:
            with contextlib.suppress(Exception):
                func(self.data)
        return self

    def tap_error(self, func: Callable[[str], None]) -> FlextResult[T]:
        """Execute side effect on error without changing result."""
        if not self.success and self.error:
            with contextlib.suppress(Exception):
                func(self.error)
        return self

    def filter(self, predicate: Callable[[T], bool]) -> FlextResult[T]:
        """Filter success result based on predicate."""
        if self.success and self.data is not None:
            try:
                if predicate(self.data):
                    return self
                return FlextResult.fail("Filter predicate failed")
            except Exception as e:
                return FlextResult.fail(f"Filter error: {e}")
        return self

    def zip_with(
        self,
        other: FlextResult[U],
        func: Callable[[T, U], object],
    ) -> FlextResult[object]:
        """Combine two results using a function."""
        if (
            self.success
            and other.success
            and self.data is not None
            and other.data is not None
        ):
            try:
                return FlextResult.ok(func(self.data, other.data))
            except Exception as e:
                return FlextResult.fail(f"Zip operation failed: {e}")

        # Return the first error encountered
        if not self.success:
            return FlextResult.fail(self.error or "Unknown error")
        return FlextResult.fail(other.error or "Unknown error")

    def to_either(self) -> tuple[T | None, str | None]:
        """Convert to tuple format (data, error)."""
        return (self.data, self.error)

    def to_exception(self) -> Exception | None:
        """Convert error to exception or return None for success."""
        if self.success:
            return None
        return ValueError(self.error or "Unknown error")

    @classmethod
    def from_exception(cls, func: Callable[[], T]) -> FlextResult[T]:
        """Create result by catching exceptions from function."""
        try:
            return cls.ok(func())
        except Exception as e:
            return cls.fail(str(e))

    @classmethod
    def combine(
        cls,
        *results: FlextResult[object],
    ) -> FlextResult[list[object]]:
        """Combine multiple results into a single result with list."""
        if not results:
            # Create empty list result
            empty_result: FlextResult[list[object]] = cast(
                "FlextResult[list[object]]",
                cls(success=True, data=[], error=None),
            )
            return empty_result

        values: list[object] = []
        for result in results:
            if not result.success:
                failure_result: FlextResult[list[object]] = cast(
                    "FlextResult[list[object]]",
                    cls(
                        success=False,
                        data=None,
                        error=result.error or "Unknown error",
                    ),
                )
                return failure_result
            values.append(result.data)

        success_result: FlextResult[list[object]] = cast(
            "FlextResult[list[object]]",
            cls(success=True, data=values, error=None),
        )
        return success_result

    @classmethod
    def all_success(cls, *results: FlextResult[object]) -> bool:
        """Check if all results are successful."""
        return all(result.success for result in results)

    @classmethod
    def any_success(cls, *results: FlextResult[object]) -> bool:
        """Check if any result is successful."""
        return any(result.success for result in results)

    @classmethod
    def first_success(cls, *results: FlextResult[T]) -> FlextResult[T]:
        """Return the first successful result, or last error if all fail."""
        for result in results:
            if result.success:
                return result
        # Return last error if all failed
        return results[-1] if results else cls.fail("No results provided")

    @classmethod
    def try_all(cls, *funcs: Callable[[], T]) -> FlextResult[T]:
        """Try functions in order until one succeeds."""
        if not funcs:
            return cls.fail("No functions provided")

        last_error = "All functions failed"
        for func in funcs:
            try:
                return cls.ok(func())
            except Exception as e:
                last_error = str(e)

        return cls.fail(last_error)


# =====================================================================
# COMPATIBILITY FUNCTIONS - Alternative naming conventions
# =====================================================================


def success[T](data: T) -> FlextResult[T]:
    """Create successful result - compatibility alias for FlextResult.ok()."""
    return FlextResult.ok(data)


def successful[T](data: T) -> FlextResult[T]:
    """Create successful result - compatibility alias for FlextResult.ok()."""
    return FlextResult.ok(data)


def failure[T](error: str) -> FlextResult[T]:
    """Create failure result - compatibility alias for FlextResult.fail()."""
    return FlextResult.fail(error)


def failed[T](error: str) -> FlextResult[T]:
    """Create failure result - compatibility alias for FlextResult.fail()."""
    return FlextResult.fail(error)


def error[T](error_msg: str) -> FlextResult[T]:
    """Create failure result - compatibility alias for FlextResult.fail()."""
    return FlextResult.fail(error_msg)


__all__ = [
    "FlextResult",
    "error",
    "failed",
    "failure",
    "success",
    "successful",
]
