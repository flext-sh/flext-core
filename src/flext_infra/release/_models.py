"""Domain models for the release subpackage."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import Field

from flext_core import FlextModels


class _BuildRecord(FlextModels.ArbitraryTypesModel):
    project: str = Field(min_length=1, description="Project name")
    path: str = Field(min_length=1, description="Project absolute path")
    exit_code: int = Field(ge=0, description="Exit code returned by make build")
    log: str = Field(min_length=1, description="Build log file path")


def _empty_build_records() -> list[_BuildRecord]:
    return []


class FlextInfraReleaseModels:
    """Models for release management."""

    class ReleaseSpec(FlextModels.ArbitraryTypesModel):
        """Release descriptor with version, tag, and bump metadata."""

        version: str = Field(min_length=1, description="Semantic version string")
        tag: str = Field(min_length=1, description="Git tag for release")
        bump_type: str = Field(min_length=1, description="Release bump type")

    class BuildRecord(_BuildRecord):
        """Single project build result entry."""

    class BuildReport(FlextModels.ArbitraryTypesModel):
        """Aggregated build report payload written to JSON."""

        version: str = Field(min_length=1, description="Release version")
        total: int = Field(ge=0, description="Total projects attempted")
        failures: int = Field(ge=0, description="Total projects with non-zero exit")
        records: Sequence[_BuildRecord] = Field(
            default_factory=_empty_build_records,
            description="Per-project build records",
        )


__all__ = ["FlextInfraReleaseModels"]
