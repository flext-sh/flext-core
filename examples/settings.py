"""Example 04 settings models."""

from __future__ import annotations

from examples import ExamplesFlextModelsErrors as _err
from flext_core import FlextSettings, c, t, u


class ExamplesSettings(FlextSettings):
    """Application settings model for settings examples."""

    api_timeout: float = u.Field(
        default_factory=lambda: c.DEFAULT_TIMEOUT_SECONDS, gt=0
    )

    service_name: str = u.Field(
        default_factory=lambda: "example-service",
        description="Service name for application",
    )
    feature_enabled: bool = u.Field(
        default_factory=lambda: True, description="Feature enable flag"
    )

    @u.field_validator("database_url", mode="before")
    @classmethod
    def normalize_database_url(cls, value: t.JsonPayload) -> str:
        """Normalize and validate database URL."""
        if not isinstance(value, str):
            raise TypeError(_err.Examples.ErrorMessages.DB_URL_MUST_BE_TEXT)
        normalized = value.strip()
        if not normalized:
            raise ValueError(_err.Examples.ErrorMessages.DB_URL_CANNOT_BE_EMPTY)
        return normalized
