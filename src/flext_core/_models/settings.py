"""Settings patterns extracted from FlextModels.

This module contains the FlextModelsSettings class with all settings-related patterns
as nested classes. It should NOT be imported directly - use FlextModels.Settings instead.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Annotated, ClassVar, Self

from pydantic import AliasChoices, ConfigDict, model_validator

from flext_core import c, t
from flext_core._models.base import FlextModelsBase
from flext_core._models.pydantic import FlextModelsPydantic


class FlextModelsSettings:
    """Settings pattern container class.

    This class acts as a namespace container for settings patterns.
    All nested classes are accessed via FlextModels.Settings.* in the main
    models.py.
    """

    class SettingsValue(FlextModelsBase.ImmutableValueModel):
        """Frozen settings branch model that preserves Pydantic env coercion."""

        model_config: ClassVar[ConfigDict] = ConfigDict(
            use_enum_values=True,
            str_strip_whitespace=True,
            validate_default=True,
            validate_return=True,
        )

    class RetryConfiguration(
        FlextModelsBase.ArbitraryTypesModel,
        FlextModelsBase.RetryConfigurationMixin,
    ):
        """Retry configuration with advanced validation."""

        max_retries: Annotated[
            t.PositiveInt,
            FlextModelsPydantic.Field(
                default=c.MAX_RETRY_ATTEMPTS,
                le=c.MAX_RETRY_ATTEMPTS,
                alias="max_attempts",
                validation_alias=AliasChoices("max_attempts", "max_retries"),
                serialization_alias="max_attempts",
                description="Maximum retry attempts from c (Constants default)",
            ),
        ] = c.MAX_RETRY_ATTEMPTS
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
            t.BackoffMultiplier,
            FlextModelsPydantic.Field(
                default=c.DEFAULT_BACKOFF_MULTIPLIER,
                description="Backoff multiplier for exponential backoff",
            ),
        ] = c.DEFAULT_BACKOFF_MULTIPLIER
        retry_on_exceptions: Annotated[
            t.SequenceOf[type[BaseException]],
            FlextModelsPydantic.Field(description="Exception types to retry on"),
        ] = FlextModelsPydantic.Field(default_factory=tuple)
        retry_on_status_codes: Annotated[
            t.SequenceOf[int],
            FlextModelsPydantic.Field(
                max_length=c.HTTP_STATUS_MIN,
                description="HTTP status codes to retry on",
            ),
        ] = FlextModelsPydantic.Field(default_factory=tuple)

        @model_validator(mode="after")
        def validate_delay_consistency(self) -> Self:
            """Validate delay configuration consistency.

            Returns:
                The resulting ``Self``.

            Raises:
                ValueError: If ``self.max_delay_seconds < self.initial_delay_seconds``.

            """
            if self.max_delay_seconds < self.initial_delay_seconds:
                raise ValueError(c.ERR_MODEL_MAX_DELAY_LESS_THAN_INITIAL)
            return self


__all__: t.MutableSequenceOf[str] = ["FlextModelsSettings"]
