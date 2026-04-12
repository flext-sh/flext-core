"""Example 04 settings models."""

from __future__ import annotations

from pydantic import Field, field_validator

from examples import c
from examples._models.errors import ExamplesFlextCoreModelsErrors as _err
from flext_core import FlextSettings, t


class ExSettingsAppSettings(FlextSettings):
    """Application settings model for settings examples."""

    database_url: str = Field(
        default=f"postgresql://{c.LOCALHOST}:5432/testdb",
        min_length=1,
    )
    api_timeout: float = Field(default=c.DEFAULT_TIMEOUT_SECONDS, gt=0)
    debug: bool = Field(default=False)
    max_workers: int = Field(default=4, ge=1)
    log_level: c.LogLevel = Field(default=c.LogLevel.INFO)

    @field_validator("database_url", mode="before")
    @classmethod
    def normalize_database_url(cls, value: t.RuntimeData) -> str:
        """Normalize and validate database URL."""
        if not isinstance(value, str):
            raise TypeError(_err.Examples.ErrorMessages.DB_URL_MUST_BE_TEXT)
        normalized = value.strip()
        if not normalized:
            raise ValueError(_err.Examples.ErrorMessages.DB_URL_CANNOT_BE_EMPTY)
        return normalized
