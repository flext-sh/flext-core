"""Base Pydantic models - Foundation for FLEXT ecosystem.

TIER 0: Uses only stdlib, pydantic, and Tier 0 modules (constants, typings).

This module provides the fundamental base classes for all Pydantic models
in the FLEXT ecosystem. All classes are nested inside FlextModelsBase
following the namespace pattern.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import uuid
from collections.abc import Hashable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Annotated, ClassVar, Self, dataclass_transform, override

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator

from flext_core import c, p, t
from flext_core._models import FlextModelsContainers, FlextModelsPydantic


class FlextModelsBase:
    """Public facade for FlextModelsBase."""

    class EnforcedModel(FlextModelsPydantic.BaseModel):
        """Base model that enforces architectural rules on subclasses."""

        model_config: ClassVar[ConfigDict] = ConfigDict(
            extra=c.EXTRA_CONFIG_FORBID,
            use_enum_values=c.MODEL_USE_ENUM_VALUES,
            validate_default=c.MODEL_VALIDATE_DEFAULT,
        )

    class ManagedModel(EnforcedModel):
        """Shared preset for assignment validation with forbidden extra fields."""

        model_config: ClassVar[ConfigDict] = ConfigDict(
            validate_assignment=True,
        )

    class EnumManagedModel(ManagedModel):
        """Managed model preserving enum members through the shared preset."""

    class NormalizedModel(EnumManagedModel):
        """Shared preset for managed models with whitespace normalization."""

        model_config: ClassVar[ConfigDict] = ConfigDict(str_strip_whitespace=True)

    class StrictManagedModel(NormalizedModel):
        """Shared preset for strict managed validation boundaries."""

        model_config: ClassVar[ConfigDict] = ConfigDict(strict=True)

    class StrictModel(StrictManagedModel):
        """Reusable strict model preset for validated domain boundaries."""

    class FrozenModel(StrictModel):
        """Immutable strict domain model preset."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    class ArbitraryTypesModel(ManagedModel):
        """Managed preset for models carrying runtime collaborators.

        Every field is described to Pydantic: a runtime dependency is declared
        ``t.Port[p.Protocol]`` (validated by the port schema), never accepted
        through ``arbitrary_types_allowed``.
        """

    class StrictBoundaryModel(FrozenModel):
        """Strict boundary model for API/external boundaries."""

    class FlexibleInternalModel(NormalizedModel):
        """Flexible internal model for domain logic."""

        model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore", strict=False)

    class ImmutableValueModel(ManagedModel):
        """Immutable value model for value objects."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    class TaggedModel(EnforcedModel):
        """Base pattern for tagged discriminated unions."""

        model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")
        tag: ClassVar[str]

    class FlexibleModel(ArbitraryTypesModel):
        """Model for dynamic configuration - allows extra fields."""

        model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore", strict=False)

    class DynamicModel(FlexibleModel):
        """Dynamic domain model preset with string whitespace normalization."""

        model_config: ClassVar[ConfigDict] = ConfigDict(str_strip_whitespace=True)

    class FrozenDynamicModel(DynamicModel):
        """Immutable dynamic domain model preset."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    class Metadata(BaseModel):
        """Standard metadata model with timestamps, audit info, tags, attributes."""

        model_config: ClassVar[ConfigDict] = ConfigDict(
            extra=c.EXTRA_CONFIG_FORBID,
            frozen=True,
            validate_assignment=True,
            populate_by_name=True,
        )
        # Validate attributes with the explicit mapping type because
        # the checker does not accept TypeAliasType as a type form.
        MAPPING_VALIDATOR: ClassVar[TypeAdapter[t.JsonMapping]] = TypeAdapter(
            Mapping[str, t.JsonValue]
        )
        # Renders JSON-able attribute values (for example ``Path``) in JSON mode;
        # an unknown type fails serialization loudly.
        JSONABLE_MAPPING: ClassVar[TypeAdapter[Mapping[str, object]]] = TypeAdapter(
            Mapping[str, object]
        )
        created_at: Annotated[
            datetime,
            Field(
                description=(
                    "Timestamp when the metadata record was first created (configured "
                    "timezone)."
                ),
                title="Created At",
                examples=["2026-03-03T10:00:00+00:00"],
            ),
        ] = Field(default_factory=lambda: datetime.now(UTC))
        updated_at: Annotated[
            datetime,
            Field(
                description=(
                    "Timestamp of the most recent metadata update (configured "
                    "timezone)."
                ),
                title="Updated At",
                examples=["2026-03-03T10:05:00+00:00"],
            ),
        ] = Field(default_factory=lambda: datetime.now(UTC))
        version: Annotated[
            str,
            Field(
                default=c.METADATA_SCHEMA_VERSION_DEFAULT,
                description=(
                    "Semantic version string representing the metadata schema revision."
                ),
                title="Metadata Version",
                examples=["1.0.0", "1.2.3"],
            ),
        ] = c.METADATA_SCHEMA_VERSION_DEFAULT
        created_by: Annotated[
            str | None,
            Field(
                default=None,
                description=(
                    "Identifier of the actor that originally created this metadata."
                ),
                title="Created By",
                examples=["system", "user-123"],
            ),
        ] = None
        modified_by: Annotated[
            str | None,
            Field(
                default=None,
                description="Identifier of the actor that last modified this metadata.",
                title="Modified By",
                examples=["system", "user-456"],
            ),
        ] = None
        tags: Annotated[
            Sequence[str],
            Field(
                description=(
                    "Normalized labels used to classify and filter this metadata."
                ),
                title="Tags",
                examples=[["billing", "critical"]],
            ),
        ] = Field(
            default=(),
            validate_default=True,
        )
        attributes: Annotated[
            t.JsonMapping,
            Field(
                description="Arbitrary metadata attributes stored as key-value pairs.",
                title="Attributes",
                examples=[{"source": "api", "priority": "high"}],
            ),
        ] = Field(
            default_factory=dict,
            validate_default=True,
        )
        metadata_value: Annotated[
            t.Scalar | None,
            Field(default=None, description="Scalar metadata value."),
        ] = None

        @field_validator("attributes", mode="before")
        @classmethod
        def validate_metadata_attributes(
            cls, value: p.MetadataInput
        ) -> t.JsonMapping | p.MetadataInput:
            """Validate attributes through the canonical metadata boundary.

            Returns:
                The validated JSON metadata attributes.

            Raises:
                ValueError: If ``key.startswith('_')``.

            """
            payload = (
                value.model_dump(mode="json")
                if isinstance(value, p.HasModelDump)
                else value
            )
            if payload is None:
                return {}
            if not isinstance(payload, Mapping):
                # Any other shape is the field's own mapping validation to
                # reject (dict_type at ``attributes``), never iterated here.
                return payload
            # Native SDK serialization renders path-like and other JSON-able
            # values before the strict JSON container admits them.
            normalized = FlextModelsContainers.ConfigMap.model_validate(
                FlextModelsBase.Metadata.JSONABLE_MAPPING.dump_python(
                    dict(payload), mode="json"
                )
            ).root
            for key in normalized:
                if key.startswith("_"):
                    message = c.ERR_RUNTIME_KEYS_WITH_UNDERSCORE_RESERVED.format(
                        key=key
                    )
                    raise ValueError(message)
            return FlextModelsBase.Metadata.MAPPING_VALIDATOR.validate_python(
                {k: v for k, v in normalized.items() if v is not None}, strict=True
            )

    class ContractModel(StrictModel):
        """Immutable strict contract using its inherited validated enum values."""

        model_config: ClassVar[ConfigDict] = ConfigDict(
            validate_return=True,
            ser_json_timedelta=c.SERIALIZATION_ISO8601,
            ser_json_bytes=c.SERIALIZATION_BASE64,
            val_json_bytes=c.SERIALIZATION_BASE64,
            hide_input_in_errors=True,
            frozen=True,
        )

    class ExternalContractModel(ContractModel):
        """Strict immutable parser of a subset of an external tool's payload.

        Declared fields validate strictly; keys the external producer adds
        beyond the consumed subset are ignored, never stored.
        """

        model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore")

    class FrozenValueModel(ContractModel, Hashable):
        """Value model with equality/hash by value."""

        @override
        def __eq__(self, other: object) -> bool:
            if not isinstance(other, type(self)):
                return NotImplemented
            self_dump: t.JsonMapping = self.model_dump()
            other_dump: t.JsonMapping = other.model_dump()
            return self_dump == other_dump

        def __hash__(self) -> int:
            data = self.model_dump()
            return hash(tuple(sorted(((k, str(v)) for k, v in data.items()))))

    @dataclass_transform(kw_only_default=True)
    class MutableConfiguredMixin:
        """Shared preset for mutable mixins with assignment validation."""

        model_config: ClassVar[ConfigDict] = ConfigDict(
            validate_assignment=True,
        )

    class NormalizedMutableConfiguredMixin(MutableConfiguredMixin):
        """Shared preset for mutable mixins with whitespace normalization."""

        model_config: ClassVar[ConfigDict] = ConfigDict(str_strip_whitespace=True)

    class IdentifiableMixin(NormalizedMutableConfiguredMixin):
        """Mixin for unique identifiers."""

        unique_id: Annotated[
            t.NonEmptyStr,
            Field(description="Unique identifier", frozen=False),
        ] = Field(default_factory=lambda: str(uuid.uuid4()))

    class TimestampableMixin(MutableConfiguredMixin):
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

    class VersionableMixin(MutableConfiguredMixin):
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
        ArbitraryTypesModel,
        TimestampableMixin,
    ):
        """Model with timestamp fields."""


__all__: list[str] = ["FlextModelsBase"]
