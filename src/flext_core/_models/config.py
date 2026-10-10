"""FlextModelsConfig - declarative config record models (ADR-005).

Frozen Pydantic v2 record for a loaded config document: its parsed data plus
optional schema and source-path references. flext-core owns only this minimal
record; flext-cli returns instances of it from its advanced loader.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, ClassVar

from flext_core._models.base import FlextModelsBase
from flext_core._models.pydantic import FlextModelsPydantic
from flext_core._protocols._logging_parts.flextprotocolslogging_part_01 import (
    FlextProtocolsLogging,
)
from flext_core._typings.base import FlextTypingBase
from flext_core._typings.services import FlextTypesServices


class FlextModelsConfig:
    """Container for declarative config record models (ADR-005)."""

    class StructlogOptions(FlextModelsBase.ArbitraryTypesModel):
        """Knob options for the structlog runtime configuration."""

        model_config: ClassVar[FlextModelsPydantic.ConfigDict] = (
            FlextModelsPydantic.ConfigDict(
                extra="forbid",
                validate_assignment=True,
                arbitrary_types_allowed=True,
            )
        )

        log_level: Annotated[
            int | None,
            FlextModelsPydantic.Field(
                default=None,
                description="Effective structlog level number.",
            ),
        ] = None
        console_renderer: Annotated[
            bool,
            FlextModelsPydantic.Field(
                default=True,
                description="Use the console renderer.",
            ),
        ] = True
        processing_stages: Annotated[
            FlextTypingBase.SequenceOf[FlextProtocolsLogging.LoggingStage],
            FlextModelsPydantic.Field(
                description="Ordered typed event stages executed before rendering.",
            ),
        ] = ()
        wrapper_class_factory: Annotated[
            FlextTypesServices.LoggerWrapperFactory | None,
            FlextModelsPydantic.Field(
                default=None,
                description="Factory building the bound-logger wrapper class.",
            ),
        ] = None
        logger_factory: Annotated[
            FlextTypesServices.LoggerFactory | None,
            FlextModelsPydantic.Field(
                default=None,
                description="Factory building the logger.",
            ),
        ] = None
        cache_logger_on_first_use: Annotated[
            bool,
            FlextModelsPydantic.Field(
                default=True,
                description="Cache the logger on first use.",
            ),
        ] = True
        async_logging: Annotated[
            bool,
            FlextModelsPydantic.Field(
                description=(
                    "Write rendered log events through the asynchronous writer."
                ),
            ),
        ] = True

    class ConfigDocument(FlextModelsBase.FrozenModel):
        """A loaded, parsed config document with optional schema/source refs."""

        model_config: ClassVar[FlextModelsPydantic.ConfigDict] = (
            FlextModelsPydantic.ConfigDict(
                frozen=True,
                arbitrary_types_allowed=True,
            )
        )

        data: Annotated[
            FlextTypingBase.JsonMapping,
            FlextModelsPydantic.Field(
                description="Parsed config mapping (execution parametrization).",
            ),
        ]
        source_path: Annotated[
            str | None,
            FlextModelsPydantic.Field(
                default=None,
                description="Absolute path of the config source.",
            ),
        ] = None
        schema_ref: Annotated[
            str | None,
            FlextModelsPydantic.Field(
                default=None,
                description="Path of the JSON Schema validating this document.",
            ),
        ] = None

    class ModelDumpOptions(FlextModelsBase.FlexibleInternalModel):
        """Options controlling Pydantic model_dump() serialization behavior."""

        by_alias: Annotated[
            bool | None,
            FlextModelsPydantic.Field(
                description="Serialize using field aliases",
                validate_default=True,
            ),
        ] = None
        exclude_none: Annotated[
            bool | None,
            FlextModelsPydantic.Field(
                description="Exclude None-valued fields",
                validate_default=True,
            ),
        ] = None
        exclude_unset: Annotated[
            bool | None,
            FlextModelsPydantic.Field(
                description="Exclude fields not explicitly set",
                validate_default=True,
            ),
        ] = None
        exclude_defaults: Annotated[
            bool | None,
            FlextModelsPydantic.Field(
                description="Exclude fields matching defaults",
                validate_default=True,
            ),
        ] = None
        include: Annotated[
            set[str] | None,
            FlextModelsPydantic.Field(
                description="Whitelist of field names to include",
                validate_default=True,
            ),
        ] = None
        exclude: Annotated[
            set[str] | None,
            FlextModelsPydantic.Field(
                description="Blacklist of field names to exclude",
                validate_default=True,
            ),
        ] = None

    class ParseOptions[T](FlextModelsBase.FlexibleInternalModel):
        """Options controlling parsing behavior for string-to-type conversion."""

        strict: Annotated[
            bool | None,
            FlextModelsPydantic.Field(
                validate_default=True,
                description="Reject coercions; fail on type mismatch",
            ),
        ] = None
        case_insensitive: Annotated[
            bool | None,
            FlextModelsPydantic.Field(
                validate_default=True,
                description="Normalize case before parsing",
            ),
        ] = None
        default: Annotated[
            T | None,
            FlextModelsPydantic.Field(
                validate_default=True,
                description="Fallback value when parsing fails",
            ),
        ] = None
        default_factory: Annotated[
            Callable[[], T] | None,
            FlextModelsPydantic.Field(
                validate_default=True,
                description="Factory producing fallback value",
            ),
        ] = None
        field_name: Annotated[
            str | None,
            FlextModelsPydantic.Field(
                validate_default=True,
                description="Source field name for error context",
            ),
        ] = None

    class RetryOptions(FlextModelsBase.FlexibleInternalModel):
        """Configuration options for retry logic."""

        max_attempts: Annotated[
            int | None,
            FlextModelsPydantic.Field(
                ge=1,
                description="Maximum number of retry attempts",
            ),
        ] = None
        delay_seconds: Annotated[
            float | None,
            FlextModelsPydantic.Field(
                ge=0,
                description="Initial delay between retries in seconds",
            ),
        ] = None

__all__: list[str] = ["FlextModelsConfig"]
