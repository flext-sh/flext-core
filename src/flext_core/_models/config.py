"""FlextModelsConfig - declarative config record models (ADR-005).

Frozen Pydantic v2 record for a loaded config document: its parsed data plus
optional schema and source-path references. flext-core owns only this minimal
record; flext-cli returns instances of it from its advanced loader.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

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


__all__: list[str] = ["FlextModelsConfig"]
