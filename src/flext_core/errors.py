"""Structured error handling for Result types.

Provides ErrorDomain enum and FlextError dataclass for categorized error handling
with proper error routing and metadata support across the FLEXT ecosystem.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from enum import StrEnum
from typing import override

from pydantic import BaseModel, ConfigDict, Field

from flext_core import t


class ErrorDomain(StrEnum):
    """Standard error domain categories for structured error routing.

    Enables consistent error handling across FLEXT projects by categorizing
    errors into domains. Each domain has standard error codes that can be
    routed to specific handlers.

    Usage:
        result = r[User].fail(
            "Validation failed: email is invalid",
            error_code="INVALID_EMAIL",
            error_data={"field": "email", "domain": ErrorDomain.VALIDATION.value},
        )
        if result.error_code == "INVALID_EMAIL":
            return 400  # Bad request
    """

    #: Validation errors (input validation, schema validation, constraints)
    VALIDATION = "VALIDATION"

    #: Network errors (connection, timeout, DNS, protocol)
    NETWORK = "NETWORK"

    #: Authentication/Authorization errors (invalid credentials, access denied)
    AUTH = "AUTH"

    #: Resource not found errors (missing user, missing file, missing record)
    NOT_FOUND = "NOT_FOUND"

    #: Operation timeout errors (request timeout, operation timeout)
    TIMEOUT = "TIMEOUT"

    #: Internal errors (unexpected state, invariant violation, internal bug)
    INTERNAL = "INTERNAL"

    #: Unknown error category (when error doesn't fit other domains)
    UNKNOWN = "UNKNOWN"

    @override
    def __str__(self) -> str:
        """Return the domain value (not the enum name)."""
        return self.value


class FlextError(BaseModel):
    """Structured error with domain, code, and metadata.

    Replaces free-form error strings with structured error objects for
    proper error routing, categorization, and handling across FLEXT.

    Attributes:
        domain: Error domain category (e.g., VALIDATION, NETWORK, AUTH)
        code: Specific error code for routing (e.g., INVALID_EMAIL, NO_CONNECTION)
        message: Human-readable error message
        details: Additional error metadata (field errors, nested errors, etc.)
        source: Original exception or error source (if applicable)

    Example:
        error = FlextError(
            domain=ErrorDomain.VALIDATION,
            code="INVALID_EMAIL",
            message="Email address is invalid",
            details={"field": "email", "value": "not-an-email"},
        )
        result = r[User].fail(
            error.message,
            error_code=error.code,
            error_data={"domain": error.domain.value, **error.details},
        )

    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    domain: ErrorDomain = Field(
        default=ErrorDomain.UNKNOWN,
        description="Error domain category for routing",
    )

    code: str = Field(
        description="Specific error code (e.g., INVALID_EMAIL, NO_CONNECTION)",
    )

    message: str = Field(
        description="Human-readable error message",
    )

    details: t.ConfigMap = Field(
        default_factory=lambda: t.ConfigMap(root={}),
        description="Additional error metadata and context",
    )

    source: BaseException | None = Field(
        default=None,
        description="Original exception if error was caught from an exception",
    )

    @classmethod
    def from_exception(
        cls,
        exc: BaseException,
        domain: ErrorDomain = ErrorDomain.INTERNAL,
        code: str | None = None,
    ) -> FlextError:
        """Create FlextError from caught exception.

        Args:
            exc: The exception that was caught
            domain: Error domain to categorize the exception
            code: Optional error code (defaults to exception class name)

        Returns:
            FlextError with exception details

        """
        error_code = code or exc.__class__.__name__
        error_message = str(exc)
        return cls(
            domain=domain,
            code=error_code,
            message=error_message,
            source=exc,
        )

    def to_dict(self) -> t.ConfigMap:
        """Convert error to ConfigMap for Result error_data.

        Returns:
            ConfigMap with domain, code, message, and details

        """
        return t.ConfigMap(
            root={
                "domain": self.domain.value,
                "code": self.code,
                "message": self.message,
            }
            | dict(self.details)
        )

    @override
    def __str__(self) -> str:
        """String representation (full message with code)."""
        return f"{self.code}: {self.message}"


__all__ = ["ErrorDomain", "FlextError"]
