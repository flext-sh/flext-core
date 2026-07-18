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
from flext_core.constants import FlextConstants as c


class FlextModelsBase(FlextModelsBasePart02):
    ArbitraryTypesModel = FlextModelsBasePart02.ArbitraryTypesModel
    MutableConfiguredMixin = FlextModelsBasePart02.MutableConfiguredMixin

    class TimestampableMixin(MutableConfiguredMixin):
        """Mixin for timestamps with Pydantic v2 validation and serialization."""

        created_at: Annotated[
            datetime,
            mp.AfterValidator(ur.ensure_utc_datetime),
            mp.Field(
                description="Creation timestamp (configured timezone)", frozen=True
            ),
        ] = mp.Field(default_factory=ug.now)
        updated_at: Annotated[
            datetime | None,
            mp.AfterValidator(ur.ensure_utc_datetime),
            mp.Field(
                default=None, description="Last update timestamp (configured timezone)"
            ),
        ] = None

        @up.field_serializer("created_at", "updated_at", when_used="json")
        def serialize_timestamps(self, value: datetime | None) -> str | None:
            """Serialize timestamps to ISO 8601 for JSON."""
            return value.isoformat() if value else None

        @up.model_validator(mode="after")
        def validate_timestamp_consistency(self) -> Self:
            """Validate timestamp consistency."""
            if self.updated_at is not None and self.updated_at < self.created_at:
                raise ValueError(c.ERR_MODEL_UPDATED_AT_BEFORE_CREATED_AT)
            return self

    class VersionableMixin(MutableConfiguredMixin):
        """Mixin for versioning with optimistic locking."""

        version: Annotated[
            t.NonNegativeInt,
            mp.Field(
                default=c.DEFAULT_RETRY_DELAY_SECONDS,
                description="Version number for optimistic locking",
                frozen=False,
            ),
        ] = c.DEFAULT_RETRY_DELAY_SECONDS

        @up.model_validator(mode="after")
        def validate_version_consistency(self) -> Self:
            """Ensure version consistency."""
            if self.version < c.DEFAULT_RETRY_DELAY_SECONDS:
                raise ValueError(
                    c.ERR_MODEL_VERSION_BELOW_MINIMUM.format(
                        version=self.version, minimum=c.DEFAULT_RETRY_DELAY_SECONDS
                    )
                )
            return self

    class RetryConfigurationMixin(mp.BaseModel):
        """Mixin for shared retry configuration properties."""

        model_config: ClassVar[t.ConfigDict] = t.ConfigDict(populate_by_name=True)
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

    class TimestampedModel(ArbitraryTypesModel, TimestampableMixin):
        """Model with timestamp fields."""


__all__: list[str] = ["FlextModelsBase"]
