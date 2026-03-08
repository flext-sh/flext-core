"""Domain models for the release subpackage."""

from __future__ import annotations

from pydantic import Field

from flext_core import FlextModels


class FlextInfraReleaseModels:
    """Models for release management."""

    class ReleaseSpec(FlextModels.ArbitraryTypesModel):
        """Release descriptor with version, tag, and bump metadata."""

        version: str = Field(min_length=1, description="Semantic version string")
        tag: str = Field(min_length=1, description="Git tag for release")
        bump_type: str = Field(min_length=1, description="Release bump type")


__all__ = ["FlextInfraReleaseModels"]
