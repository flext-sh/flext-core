"""CLI Pydantic domain models."""

from __future__ import annotations

from typing import Annotated, ClassVar

from flext_core import m


class FlextCliModelsBase:
    """Implementation part for FlextCliModelsBase."""

    class SettingsSnapshot(m.Value):
        """Snapshot of current CLI settings information."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True, extra="forbid")

        settings_dir: Annotated[str, m.Field(description="Settings directory path")] = (
            ""
        )

        settings_exists: Annotated[
            bool, m.Field(description="Whether settings directory exists")
        ] = False

        settings_readable: Annotated[
            bool, m.Field(description="Whether settings directory is readable")
        ] = False

        settings_writable: Annotated[
            bool, m.Field(description="Whether settings directory is writable")
        ] = False

        timestamp: Annotated[str, m.Field(description="Timestamp of snapshot")] = ""


__all__: list[str] = ["FlextCliModelsBase"]
