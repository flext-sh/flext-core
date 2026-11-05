"""Settings patterns extracted from FlextModels.

This module contains the FlextModelsSettings class with all settings-related patterns
as nested classes. It should NOT be imported directly - use
FlextModels.Settings instead.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import MutableSequence, Sequence
from typing import TYPE_CHECKING, Annotated, ClassVar, Self

from pydantic import AliasChoices, ConfigDict, model_validator

import flext_core

if TYPE_CHECKING:
    pass
from flext_core._models import FlextModelsBase, FlextModelsPydantic


class FlextModelsSettings:
    """Settings pattern container class.

    This class acts as a namespace container for settings patterns.
    All nested classes are accessed via FlextModels.Settings.* in the main
    models.py.
    """

    class SettingsValue(FlextModelsBase.ImmutableValueModel):
        """Frozen settings branch model that preserves Pydantic env coercion."""

        model_config: ClassVar[ConfigDict] = ConfigDict(
            str_strip_whitespace=True,
            validate_default=True,
            validate_return=True,
        )

    class RetryConfiguration(
        FlextModelsBase.RetryConfigurationMixin,
    ):
        """Retry configuration with advanced validation."""

        max_retries: Annotated[
            flext_core.t.PositiveInt,
            FlextModelsPydantic.Field(
                default=flext_core.c.MAX_RETRY_ATTEMPTS,
                le=flext_core.c.MAX_RETRY_ATTEMPTS,
                alias="max_attempts",
                validation_alias=AliasChoices("max_attempts", "max_retries"),
                serialization_alias="max_attempts",
                description="Maximum retry attempts from c (Constants default)",
            ),
        ] = flext_core.c.MAX_RETRY_ATTEMPTS
        exponential_backoff: Annotated[
            bool,
            FlextModelsPydantic.Field(
                default=True,
                description=(
                    "Whether to use exponential backoff between retry attempts."
                ),
            ),
        ] = True
        backoff_multiplier: Annotated[
            flext_core.t.BackoffMultiplier,
            FlextModelsPydantic.Field(
                default=flext_core.c.DEFAULT_BACKOFF_MULTIPLIER,
                description="Backoff multiplier for exponential backoff",
            ),
        ] = flext_core.c.DEFAULT_BACKOFF_MULTIPLIER
        retry_on_exceptions: Annotated[
            Sequence[type[BaseException]],
            FlextModelsPydantic.Field(description="Exception types to retry on"),
        ] = FlextModelsPydantic.Field(
            default=(),
            validate_default=True,
        )
        retry_on_status_codes: Annotated[
            Sequence[int],
            FlextModelsPydantic.Field(
                max_length=flext_core.c.HTTP_STATUS_MIN,
                description="HTTP status codes to retry on",
            ),
        ] = FlextModelsPydantic.Field(
            default=(),
            validate_default=True,
        )

        @model_validator(mode="after")
        def validate_delay_consistency(self) -> Self:
            """Validate delay configuration consistency.

            Returns:
                The resulting ``Self``.

            Raises:
                ValueError: If ``self.max_delay_seconds < self.initial_delay_seconds``.

            """
            if self.max_delay_seconds < self.initial_delay_seconds:
                raise ValueError(flext_core.c.ERR_MODEL_MAX_DELAY_LESS_THAN_INITIAL)
            return self


__all__: MutableSequence[str] = ["FlextModelsSettings"]
