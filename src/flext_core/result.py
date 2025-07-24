"""FlextResult - Enterprise Result Pattern Implementation.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

Implementation of the Result pattern for type-safe error handling.
Provides the foundational error handling mechanism for FLEXT ecosystem
projects with functional programming paradigms.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import TypeVar
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
            except Exception as e:  # noqa: BLE001
                return FlextResult.fail(
                    f"Unexpected transformation error: {e}",
                )
        # Return failure as FlextResult[U] (type ignore since failure has no
        # data)
        return FlextResult.fail(self.error or "Unknown error")

    def flat_map(
        self,
        func: Callable[[T], FlextResult[T]],
    ) -> FlextResult[T]:
        """Chain operations that return FlextResult instances."""
        if self.success and self.data is not None:
            try:
                return func(self.data)
            except (TypeError, AttributeError, ValueError, IndexError) as e:
                return FlextResult.fail(f"Chained operation failed: {e}")
            except Exception as e:  # noqa: BLE001
                return FlextResult.fail(f"Unexpected chaining error: {e}")
        return self


__all__ = ["FlextResult"]
