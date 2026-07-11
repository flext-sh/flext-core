"""Example 04 settings models."""

from __future__ import annotations

from flext_core import FlextSettings, c, u


class ExamplesSettings(FlextSettings):
    """Application settings model for settings examples."""

    api_timeout: float = u.Field(
        default_factory=lambda: c.DEFAULT_TIMEOUT_SECONDS,
        gt=0,
    )

    service_name: str = u.Field(
        default_factory=lambda: "example-service",
        description="Service name for application",
    )
    feature_enabled: bool = u.Field(
        default_factory=lambda: True,
        description="Feature enable flag",
    )
