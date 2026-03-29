"""FlextModelsErrors - structured error model for Result types.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Annotated, override

from pydantic import Field

from flext_core._constants.errors import FlextConstantsErrors
from flext_core._models.base import FlextModelFoundation
from flext_core.typings import t


class FlextModelsErrors:
    """Error models for structured error handling (Layer 2)."""

    class Error(FlextModelFoundation.ArbitraryTypesModel):
        """Structured error with domain, code, and metadata.

        Replaces free-form error strings with structured error objects for
        proper error routing, categorization, and handling across FLEXT.

        Attributes:
            domain: Error domain category (e.g., VALIDATION, NETWORK, AUTH)
            code: Specific error code for routing (e.g., INVALID_EMAIL, NO_CONNECTION)
            message: Human-readable error message
            details: Additional error metadata (field errors, nested errors, etc.)
            source: Original exception or error source (if applicable)

        """

        domain: Annotated[
            FlextConstantsErrors.ErrorDomain,
            Field(
                default=FlextConstantsErrors.ErrorDomain.UNKNOWN,
                description="Error domain category for routing",
            ),
        ]

        code: Annotated[
            str,
            Field(
                description="Specific error code (e.g., INVALID_EMAIL, NO_CONNECTION)",
            ),
        ]

        message: Annotated[
            str,
            Field(
                description="Human-readable error message",
            ),
        ]

        details: Annotated[
            t.ConfigMap,
            Field(
                description="Additional error metadata and context",
            ),
        ] = Field(default_factory=lambda: t.ConfigMap(root={}))

        source: Annotated[
            BaseException | None,
            Field(
                default=None,
                description="Original exception if error was caught from an exception",
            ),
        ]

        @classmethod
        def from_exception(
            cls,
            exc: BaseException,
            domain: FlextConstantsErrors.ErrorDomain = FlextConstantsErrors.ErrorDomain.INTERNAL,
            code: str | None = None,
        ) -> FlextModelsErrors.Error:
            """Create Error from caught exception.

            Args:
                exc: The exception that was caught
                domain: Error domain to categorize the exception
                code: Optional error code (defaults to exception class name)

            Returns:
                Error with exception details

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
                | dict(self.details),
            )

        @override
        def __str__(self) -> str:
            """String representation (full message with code)."""
            return f"{self.code}: {self.message}"
