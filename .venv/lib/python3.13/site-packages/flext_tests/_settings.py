"""Runtime settings namespace for FLEXT test services."""

from __future__ import annotations

from typing import ClassVar

from pydantic_settings import SettingsConfigDict

from flext_core import FlextSettings, m


class FlextTestsSettings(FlextSettings):
    """FLEXT settings extended with the test runtime namespace."""

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_prefix="FLEXT_TESTS_", extra="ignore"
    )

    class TestsSettings(m.BaseModel):
        """Namespaced test runtime settings branch."""

    Tests: TestsSettings = m.Field(
        default_factory=TestsSettings,
        description="Namespaced FLEXT tests settings branch.",
    )


settings: FlextTestsSettings = FlextTestsSettings.fetch_global()
"""Pre-instantiated project settings singleton — ``from flext_tests import settings``."""

__all__: list[str] = ["FlextTestsSettings", "settings"]
