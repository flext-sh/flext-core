"""Domain models for the release subpackage."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated

from pydantic import Field

from flext_core import FlextModels


class FlextInfraReleaseModels:
    """Models for release management."""

    class BuildRecord(FlextModels.ArbitraryTypesModel):
        """Base model for build result data."""

        project: Annotated[str, Field(min_length=1, description="Project name")]
        path: Annotated[str, Field(min_length=1, description="Project absolute path")]
        exit_code: Annotated[
            int, Field(ge=0, description="Exit code returned by make build")
        ]
        log: Annotated[str, Field(min_length=1, description="Build log file path")]

    class ReleaseSpec(FlextModels.ArbitraryTypesModel):
        """Release descriptor with version, tag, and bump metadata."""

        version: Annotated[
            str, Field(min_length=1, description="Semantic version string")
        ]
        tag: Annotated[str, Field(min_length=1, description="Git tag for release")]
        bump_type: Annotated[str, Field(min_length=1, description="Release bump type")]

    class BuildReport(FlextModels.ArbitraryTypesModel):
        """Aggregated build report payload written to JSON."""

        version: Annotated[str, Field(min_length=1, description="Release version")]
        total: Annotated[int, Field(ge=0, description="Total projects attempted")]
        failures: Annotated[
            int, Field(ge=0, description="Total projects with non-zero exit")
        ]
        records: Annotated[
            Sequence[FlextInfraReleaseModels.BuildRecord],
            Field(
                default_factory=list,
                description="Per-project build records",
            ),
        ]


__all__ = ["FlextInfraReleaseModels"]
