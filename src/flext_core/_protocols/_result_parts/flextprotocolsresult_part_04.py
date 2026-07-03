"""FlextProtocolsResult - result and model-dump contracts.

The public ``p.Result`` contract is nominal for direct static typing, while
auxiliary structural protocols segment the instance API by concern. Today only
``ResultLike`` has a direct structural consumer in the workspace, but the other
protocols still document and organize the full public result surface.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from flext_core.models import FlextModels as m
    from flext_core.typings import FlextTypes as t
from flext_core._protocols._result_parts.flextprotocolsresult_part_03 import (
    FlextProtocolsResult as FlextProtocolsResultPart03,
)


class FlextProtocolsResult(FlextProtocolsResultPart03):
    @runtime_checkable
    class HasModelDump(Protocol):
        """Protocol for items that can dump model data.

        Used for Pydantic model compatibility and serialization.
        """

        def model_dump(
            self,
            *,
            mode: str = "python",
        ) -> t.MappingKV[str, t.JsonPayload | None]:
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

        @property
        def message(self) -> str:
            """Canonical exception message."""
            ...

        @property
        def metadata(self) -> m.Metadata:
            """Structured metadata attached to the error."""
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
