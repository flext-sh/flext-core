"""Example 02 settings models."""

from __future__ import annotations

from pydantic import Field

from flext_core import FlextSettings


class Ex02TestConfig(FlextSettings):
    """Settings model used by Ex02 settings golden tests."""

    service_name: str = Field(default="example-service")
    feature_enabled: bool = Field(default=True)
