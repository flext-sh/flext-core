"""Settings patterns extracted from FlextModels.

This module contains the FlextModelsSettings class with all settings-related patterns
as nested classes. It should NOT be imported directly - use FlextModels.Settings instead.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, ClassVar, Final, Self

from pydantic import (
    AliasChoices,
    ConfigDict,
    Field,
    model_validator,
)

from flext_core import FlextModelsBase as m, c, p, t


class FlextModelsSettings:
    """Settings pattern container class.

    This class acts as a namespace container for settings patterns.
    All nested classes are accessed via FlextModels.Settings.* in the main models.py.
    """

    class AutoSettings(m.ArbitraryTypesModel):
        """Automatic settings wrapper for canonical FLEXT settings classes."""

        model_config: ClassVar[ConfigDict] = ConfigDict(
            frozen=True,
            arbitrary_types_allowed=True,
        )

        settings_class: Annotated[
            type[p.SettingsType],
            Field(description="Settings class to instantiate"),
        ]
        env_prefix: Annotated[
            str,
            Field(
                default=c.ENV_PREFIX,
                description="Environment variable prefix for settings resolution",
            ),
        ] = c.ENV_PREFIX
        env_file: Annotated[
            str | None,
            Field(
                default=None,
                description="Path to .env file for environment variable loading",
            ),
        ] = None

        def create_settings(self) -> p.Settings:
            return self.settings_class.fetch_global()

    class RetryConfiguration(
        m.ArbitraryTypesModel,
        m.RetryConfigurationMixin,
    ):
        """Retry configuration with advanced validation."""

        max_retries: Annotated[
            t.PositiveInt,
            Field(
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
            Field(
                default=True,
                description="Whether to use exponential backoff between retry attempts.",
            ),
        ] = True
        backoff_multiplier: Annotated[
            t.BackoffMultiplier,
            Field(
                default=c.DEFAULT_BACKOFF_MULTIPLIER,
                description="Backoff multiplier for exponential backoff",
            ),
        ] = c.DEFAULT_BACKOFF_MULTIPLIER
        retry_on_exceptions: Annotated[
            Sequence[type[BaseException]],
            Field(
                description="Exception types to retry on",
            ),
        ] = Field(default_factory=list[type[BaseException]])
        retry_on_status_codes: Annotated[
            Sequence[int],
            Field(
                max_length=c.HTTP_STATUS_MIN,
                description="HTTP status codes to retry on",
            ),
        ] = Field(default_factory=list[int])

        @model_validator(mode="after")
        def validate_delay_consistency(self) -> Self:
            """Validate delay configuration consistency."""
            if self.max_delay_seconds < self.initial_delay_seconds:
                raise ValueError(c.ERR_MODEL_MAX_DELAY_LESS_THAN_INITIAL)
            return self

    DOMAIN_MODEL_CONFIG: Final[ConfigDict] = ConfigDict(
        use_enum_values=True,
        validate_assignment=True,
        validate_return=True,
        validate_default=True,
        str_strip_whitespace=True,
        arbitrary_types_allowed=False,
        extra="forbid",
    )
    "Domain model configuration defaults.\n\n    Moved from FlextConstants.DOMAIN_MODEL_CONFIG because\n    constants.py cannot import ConfigDict from pydantic.\n\n    Use m.DOMAIN_MODEL_CONFIG instead of c.DOMAIN_MODEL_CONFIG.\n    "


__all__ = ["FlextModelsSettings"]
