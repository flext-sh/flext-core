"""Base Pydantic models - Foundation for FLEXT ecosystem.

TIER 0: Uses only stdlib, pydantic, and Tier 0 modules (constants, typings).

This module provides the fundamental base classes for all Pydantic models
in the FLEXT ecosystem. All classes are nested inside FlextModelsBase
following the namespace pattern.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
)

from datetime import UTC, datetime
from typing import Annotated, ClassVar, Self

from pydantic import ConfigDict

from flext_core import c, t
from flext_core._models import FlextModelsPydantic
from flext_core._models._base import FlextModelsBasePart02


class FlextModelsBasePart03(FlextModelsBasePart02):
    class TimestampableMixin(FlextModelsBasePart02.MutableConfiguredMixin):
        """Mixin for timestamps with Pydantic v2 validation and serialization."""

        created_at: Annotated[
            datetime,
            FlextModelsPydantic.Field(
                description="Creation timestamp (configured timezone)",
                frozen=True,
            ),
        ] = FlextModelsPydantic.Field(default_factory=lambda: datetime.now(UTC))
        updated_at: Annotated[
            datetime | None,
            FlextModelsPydantic.Field(
                default=None,
                description="Last update timestamp (configured timezone)",
            ),
        ] = None

        @FlextModelsPydantic.field_validator("created_at", "updated_at", mode="after")
        @classmethod
        def ensure_utc_datetime(cls, value: datetime | None) -> datetime | None:
            """Attach UTC only to naive inputs; retain an aware input's timezone.

            Returns:
                The timezone-aware timestamp or None.

            """
            return (
                value.replace(tzinfo=UTC)
                if value is not None and value.tzinfo is None
                else value
            )

        # No ``staticmethod`` wrapper: pydantic 2.13 silently drops a
        # ``staticmethod``-wrapped serializer from ``__pydantic_decorators__``
        # and the default datetime serializer takes over (``Z`` instead of
        # ``isoformat()``). The 2.13.5 plain-signature contract accepts the
        # classmethod ``(cls, value)`` shape; the instance-method single
        # ``self``-named parameter (the value annotated as ``self``) is
        # rejected by ``inspect_field_serializer``, and the self-less
        # ``(value)`` shape fails the fleet method-naming rule. The
        # classmethod keeps the value mapping pure (no instance state).
        @FlextModelsPydantic.field_serializer(
            "created_at", "updated_at", when_used="json"
        )
        @classmethod
        def serialize_timestamps(cls, value: datetime | None) -> str | None:
            """Serialize timestamps to ISO 8601 for JSON.

            Returns:
                The resulting ``str | None``.

            """
            return value.isoformat() if value else None

        @FlextModelsPydantic.model_validator(mode="after")
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
            FlextModelsPydantic.Field(
                default=c.DEFAULT_RETRY_DELAY_SECONDS,
                description="Version number for optimistic locking",
                frozen=False,
            ),
        ] = c.DEFAULT_RETRY_DELAY_SECONDS

        @FlextModelsPydantic.model_validator(mode="after")
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

    class RetryConfigurationMixin(FlextModelsPydantic.BaseModel):
        """Mixin for shared retry configuration properties."""

        model_config: ClassVar[ConfigDict] = ConfigDict(populate_by_name=True)
        max_retries: Annotated[
            t.NonNegativeInt,
            FlextModelsPydantic.Field(
                default=c.MAX_RETRY_ATTEMPTS,
                alias="max_attempts",
                description="Maximum retry attempts",
            ),
        ] = c.MAX_RETRY_ATTEMPTS
        initial_delay_seconds: Annotated[
            t.PositiveFloat,
            FlextModelsPydantic.Field(
                default=c.DEFAULT_RETRY_DELAY_SECONDS,
                description="Initial delay between retries",
            ),
        ] = c.DEFAULT_RETRY_DELAY_SECONDS
        max_delay_seconds: Annotated[
            t.PositiveFloat,
            FlextModelsPydantic.Field(
                default=c.DEFAULT_MAX_DELAY_SECONDS,
                description="Maximum delay between retries",
            ),
        ] = c.DEFAULT_MAX_DELAY_SECONDS

    class TimestampedModel(
        FlextModelsBasePart02.ArbitraryTypesModel,
        TimestampableMixin,
    ):
        """Model with timestamp fields."""


__all__: list[str] = ["FlextModelsBasePart03"]
