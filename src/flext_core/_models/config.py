"""FlextModelsConfig - declarative config record models (ADR-005).

Frozen Pydantic v2 record for a loaded config document: its parsed data plus
optional schema and source-path references. flext-core owns only this minimal
record; flext-cli returns instances of it from its advanced loader.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Annotated, ClassVar

from structlog.types import Processor

from flext_core import t
from flext_core._models.base import FlextModelsBase as m
from flext_core._models.pydantic import FlextModelsPydantic as mp


class FlextModelsConfig:
    """Container for declarative config record models (ADR-005)."""

    class StructlogOptions(m.ArbitraryTypesModel):
        """Knob options for the structlog runtime configuration."""

        model_config: ClassVar[mp.ConfigDict] = mp.ConfigDict(
            extra="forbid",
            validate_assignment=True,
            arbitrary_types_allowed=True,
        )

        log_level: Annotated[
            int | None,
            mp.Field(default=None, description="Effective structlog level number."),
        ] = None
        console_renderer: Annotated[
            bool,
            mp.Field(default=True, description="Use the console renderer."),
        ] = True
        additional_processors: Annotated[
            t.SequenceOf[Processor] | None,
            mp.Field(
                default=None,
                description="Extra structlog processors appended to the chain.",
            ),
        ] = None
        wrapper_class_factory: Annotated[
            t.LoggerWrapperFactory | None,
            mp.Field(
                default=None,
                description="Factory building the bound-logger wrapper class.",
            ),
        ] = None
        logger_factory: Annotated[
            t.LoggerFactory | None,
            mp.Field(default=None, description="Factory building the logger."),
        ] = None
        cache_logger_on_first_use: Annotated[
            bool,
            mp.Field(default=True, description="Cache the logger on first use."),
        ] = True

    class ConfigDocument(m.FrozenModel):
        """A loaded, parsed config document with optional schema/source refs."""

        model_config: ClassVar[mp.ConfigDict] = mp.ConfigDict(
            frozen=True,
            arbitrary_types_allowed=True,
        )

        data: Annotated[
            t.JsonMapping,
            mp.Field(description="Parsed config mapping (execution parametrization)."),
        ]
        source_path: Annotated[
            str | None,
            mp.Field(default=None, description="Absolute path of the config source."),
        ] = None
        schema_ref: Annotated[
            str | None,
            mp.Field(
                default=None,
                description="Path of the JSON Schema validating this document.",
            ),
        ] = None


__all__: list[str] = ["FlextModelsConfig"]
