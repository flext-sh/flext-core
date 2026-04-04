"""Registry tracking models extracted from FlextRegistry.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import MutableSequence
from typing import Annotated

from pydantic import Field, computed_field

from flext_core import t
from flext_core._models.entity import FlextModelsEntity
from flext_core._models.handler import FlextModelsHandler


class FlextModelsRegistry:
    """Registry model namespace for handler registration aggregates."""

    class RegistrySummary(FlextModelsEntity.Value):
        """Aggregated outcome for batch handler registration tracking."""

        registered: Annotated[
            MutableSequence[FlextModelsHandler.RegistrationDetails],
            Field(
                description="Successfully registered handlers with registration details.",
            ),
        ] = Field(default_factory=list[FlextModelsHandler.RegistrationDetails])
        skipped: Annotated[
            t.StrSequence,
            Field(
                description="Handler identifiers that were skipped (already registered)",
                examples=[["CreateUserCommand", "UpdateUserCommand"]],
            ),
        ] = Field(default_factory=list[str])
        errors: Annotated[
            MutableSequence[str],
            Field(
                description="Error messages for failed registrations",
                examples=[["Handler validation failed", "Duplicate registration"]],
            ),
        ] = Field(default_factory=list[str])

        @computed_field
        def is_failure(self) -> bool:
            """Indicate whether the batch registration had errors."""
            return bool(self.errors)

        @computed_field
        def is_success(self) -> bool:
            """Indicate whether the batch registration fully succeeded."""
            return not self.errors


__all__ = ["FlextModelsRegistry"]
