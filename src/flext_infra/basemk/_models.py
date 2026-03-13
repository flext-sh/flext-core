"""Domain models for the basemk subpackage."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from flext_core import FlextModels
from flext_infra import c


class FlextInfraBasemkModels:
    """Models for base.mk template rendering."""

    class BaseMkConfig(FlextModels.ArbitraryTypesModel):
        """Configuration model used to render base.mk templates."""

        project_name: Annotated[
            str, Field(min_length=1, description="Project identifier")
        ]
        python_version: Annotated[
            str, Field(min_length=1, description="Target Python version")
        ]
        core_stack: Annotated[
            str, Field(min_length=1, description="Core stack classification")
        ]
        package_manager: Annotated[
            str,
            Field(
                default=c.Infra.Toml.POETRY,
                description="Dependency manager",
            ),
        ]
        source_dir: Annotated[
            str,
            Field(
                default=c.Infra.Paths.DEFAULT_SRC_DIR,
                description="Source directory path",
            ),
        ]
        tests_dir: Annotated[
            str,
            Field(
                default=c.Infra.Directories.TESTS,
                description="Tests directory path",
            ),
        ]
        lint_gates: Annotated[
            list[str],
            Field(
                default_factory=list,
                description="Enabled quality gates",
            ),
        ]
        test_command: Annotated[
            str,
            Field(
                default=c.Infra.Toml.PYTEST,
                description="Default test command",
            ),
        ]


__all__ = ["FlextInfraBasemkModels"]
