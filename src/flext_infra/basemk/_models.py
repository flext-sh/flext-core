"""Domain models for the basemk subpackage."""

from __future__ import annotations

from pydantic import Field

from flext_core import FlextModels
from flext_infra import c


class FlextInfraBasemkModels:
    """Models for base.mk template rendering."""

    class BaseMkConfig(FlextModels.ArbitraryTypesModel):
        """Configuration model used to render base.mk templates."""

        project_name: str = Field(min_length=1, description="Project identifier")
        python_version: str = Field(min_length=1, description="Target Python version")
        core_stack: str = Field(min_length=1, description="Core stack classification")
        package_manager: str = Field(
            default=c.Infra.Toml.POETRY, description="Dependency manager",
        )
        source_dir: str = Field(
            default=c.Infra.Paths.DEFAULT_SRC_DIR,
            description="Source directory path",
        )
        tests_dir: str = Field(
            default=c.Infra.Directories.TESTS,
            description="Tests directory path",
        )
        lint_gates: list[str] = Field(
            default_factory=list,
            description="Enabled quality gates",
        )
        test_command: str = Field(
            default=c.Infra.Toml.PYTEST, description="Default test command",
        )


__all__ = ["FlextInfraBasemkModels"]
