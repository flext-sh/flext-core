"""Context token models for variable reset operations.

Token models track context variable state changes and enable rollback
to previous values in context managers and error handlers.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from flext_core import FlextModelsEntity, FlextUtilitiesPydantic, c, t


class FlextModelsContextTokens:
    """Namespace for context token models."""

    class StructlogProxyToken(FlextModelsEntity.Value):
        """Token for resetting structlog context variables.

        Used by StructlogProxyContextVar to track previous values and enable
        rollback to previous context state.

        Attributes:
            key: The context variable key being tracked
            previous_value: The value before the set operation (None if unset)

        """

        key: Annotated[
            t.NonEmptyStr,
            FlextUtilitiesPydantic.Field(
                pattern=c.PATTERN_IDENTIFIER_WITH_UNDERSCORE,
                description="Context variable key (alphanumeric, underscore)",
                examples=["correlation_id", "service_name", "user_id"],
            ),
        ]
        previous_value: Annotated[
            t.RuntimeData | datetime | None,
            FlextUtilitiesPydantic.Field(
                default=None, description="Previous value before set operation"
            ),
        ] = None

    class Token(FlextModelsEntity.Value):
        """Token for context variable reset operations.

        Used by FlextContext to track context variable changes and enable
        rollback to previous values.

        Attributes:
            key: The context variable key being tracked
            old_value: The value before the set operation (None if unset)

        """

        key: Annotated[
            t.NonEmptyStr,
            FlextUtilitiesPydantic.Field(
                description="Unique key for the context variable",
                examples=["user_id", "request_id", "session_id"],
            ),
        ]
        old_value: Annotated[
            t.RuntimeData | None,
            FlextUtilitiesPydantic.Field(
                default=None, description="Previous value before set operation"
            ),
        ]


__all__: list[str] = ["FlextModelsContextTokens"]
