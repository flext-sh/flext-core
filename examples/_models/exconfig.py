"""Example 04 configuration models."""

from __future__ import annotations

from pydantic import Field, field_validator

from examples import c
from flext_core import FlextSettings


class ExConfigAppConfig(FlextSettings):
    """Application settings model for configuration examples."""

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
    def normalize_database_url(cls, value: object) -> str:
        """Normalize and validate database URL."""
        if not isinstance(value, str):
            msg = "Database URL must be text"
            raise TypeError(msg)
        normalized = value.strip()
        if not normalized:
            msg = "Database URL cannot be empty"
            raise ValueError(msg)
        return normalized
