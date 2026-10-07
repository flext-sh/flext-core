"""Base Pydantic models - Foundation for FLEXT ecosystem.

TIER 0: Uses only stdlib, pydantic, and Tier 0 modules (constants, typings).

This module provides the fundamental base classes for all Pydantic models
in the FLEXT ecosystem. All classes are nested inside FlextModelsBase
following the namespace pattern.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, ClassVar, Self

from flext_core._models._base_parts.flextmodelsbase_part_02 import (
    FlextModelsBase as FlextModelsBasePart02,
)
from flext_core._models.pydantic import FlextModelsPydantic as mp
from flext_core._runtime._metadata_validation import (
    FlextRuntimeMetadataValidation as ur,
)
from flext_core._typings.base import FlextTypingBase as t
from flext_core._utilities.generators import FlextUtilitiesGenerators as ug
from flext_core._utilities.pydantic import FlextUtilitiesPydantic as up
from flext_core.constants import c


class FlextModelsBase(FlextModelsBasePart02):
    class TimestampableMixin(FlextModelsBasePart02.MutableConfiguredMixin):
        """Mixin for timestamps with Pydantic v2 validation and serialization."""

        created_at: Annotated[
            datetime,
            mp.AfterValidator(ur.ensure_utc_datetime),
            mp.Field(
                description="Creation timestamp (configured timezone)",
                frozen=True,
            ),
        ] = mp.Field(default_factory=ug.now)
        updated_at: Annotated[
            datetime | None,
            mp.AfterValidator(ur.ensure_utc_datetime),
            mp.Field(
                default=None,
                description="Last update timestamp (configured timezone)",
            ),
        ] = None

        # No ``staticmethod`` wrapper: pydantic 2.13 only registers serializers
        # declared on plain class-body functions, and the value parameter keeps
        # a value-oriented name: pydantic 2.13.5's signature contract for
        # ``mode=plain`` rejects a single ``self``-named parameter (it reads it
        # as a bound method); wrapping the marked function in ``staticmethod``
        # silently drops it from ``__pydantic_decorators__`` and the default
        # datetime serializer takes over (``Z`` instead of ``isoformat()``).
        @up.field_serializer("created_at", "updated_at", when_used="json")
        def serialize_timestamps(
            value: datetime | None,  # ruff: ignore[invalid-first-argument-name-for-method] -- pydantic 2.13.5's plain field_serializer contract requires the value-named parameter (see the block comment above); the self-named form is rejected at import time.
        ) -> str | None:
            """Serialize timestamps to ISO 8601 for JSON.

            Returns:
                The resulting ``str | None``.
            """
            return value.isoformat() if value else None

        @mp.model_validator(mode="after")
        def validate_timestamp_consistency(self) -> Self:
            """Validate timestamp consistency.

            Returns:
                The resulting ``Self``.

            Raises:
                ValueError: If ``self.updated_at is not None and self.updated_at <
                    self.created_at``.
            """
            if self.updated_at is not None and self.updated_at < self.created_at:
                raise ValueError(c.ERR_MODEL_UPDATED_AT_BEFORE_CREATED_AT)
            return self

    class VersionableMixin(FlextModelsBasePart02.MutableConfiguredMixin):
        """Mixin for versioning with optimistic locking."""

        version: Annotated[
            t.NonNegativeInt,
            mp.Field(
                default=c.DEFAULT_RETRY_DELAY_SECONDS,
                description="Version number for optimistic locking",
                frozen=False,
            ),
        ] = c.DEFAULT_RETRY_DELAY_SECONDS

        @mp.model_validator(mode="after")
        def validate_version_consistency(self) -> Self:
            """Ensure version consistency.

            Returns:
                The resulting ``Self``.

            Raises:
                ValueError: If ``self.version < c.DEFAULT_RETRY_DELAY_SECONDS``.
            """
            if self.version < c.DEFAULT_RETRY_DELAY_SECONDS:
                raise ValueError(
                    c.ERR_MODEL_VERSION_BELOW_MINIMUM.format(
                        version=self.version,
                        minimum=c.DEFAULT_RETRY_DELAY_SECONDS,
                    ),
                )
            return self

    class RetryConfigurationMixin(mp.BaseModel):
        """Mixin for shared retry configuration properties."""

        model_config: ClassVar[mp.ConfigDict] = mp.ConfigDict(populate_by_name=True)
        max_retries: Annotated[
            t.NonNegativeInt,
            mp.Field(
                default=c.MAX_RETRY_ATTEMPTS,
                alias="max_attempts",
                description="Maximum retry attempts",
            ),
        ] = c.MAX_RETRY_ATTEMPTS
        initial_delay_seconds: Annotated[
            t.PositiveFloat,
            mp.Field(
                default=c.DEFAULT_RETRY_DELAY_SECONDS,
                description="Initial delay between retries",
            ),
        ] = c.DEFAULT_RETRY_DELAY_SECONDS
        max_delay_seconds: Annotated[
            t.PositiveFloat,
            mp.Field(
                default=c.DEFAULT_MAX_DELAY_SECONDS,
                description="Maximum delay between retries",
            ),
        ] = c.DEFAULT_MAX_DELAY_SECONDS

    class TimestampedModel(
        FlextModelsBasePart02.ArbitraryTypesModel,
        TimestampableMixin,
    ):
        """Model with timestamp fields."""


__all__: list[str] = ["FlextModelsBase"]
