"""Project metadata model parts.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import Field

from flext_core._models._project_metadata_parts.flextmodelsprojectmetadata_part_01 import (  # ruff: ignore[line-too-long] -- the dotted module path is a single identifier chain; it cannot be wrapped without renaming the module
    FlextModelsProjectMetadataContract,
)
from flext_core._models._project_metadata_parts.flextmodelsprojectmetadata_part_03 import (  # ruff: ignore[line-too-long] -- the dotted module path is a single identifier chain; it cannot be wrapped without renaming the module
    FlextModelsProjectMetadataAggregates,
)
from flext_core._models._project_metadata_parts.flextmodelsprojectmetadata_part_05 import (  # ruff: ignore[line-too-long] -- the dotted module path is a single identifier chain; it cannot be wrapped without renaming the module
    FlextModelsPyprojectIngressContract,
)


class FlextModelsProjectMetadataDocument(FlextModelsProjectMetadataAggregates):
    """Validated TOML document sub-tables and canonical domain aggregate."""

    class PyprojectTool(FlextModelsPyprojectIngressContract):
        """Owned subset of the top-level ``[tool]`` table."""

        flext: Annotated[
            FlextModelsProjectMetadataAggregates.ProjectToolFlext,
            Field(
                default_factory=FlextModelsProjectMetadataAggregates.ProjectToolFlext,
                description="Validated FLEXT project policy",
            ),
        ] = Field(default_factory=FlextModelsProjectMetadataAggregates.ProjectToolFlext)

    class ProjectMetadata(FlextModelsProjectMetadataContract):
        """Canonical project metadata retaining exact validated source objects."""

        root: Annotated[Path, Field(description="Project root")]
        package_name: Annotated[str, Field(min_length=1, description="Import package")]
        class_stem: Annotated[str, Field(min_length=1, description="Class stem")]
        project: Annotated[
            FlextModelsProjectMetadataAggregates.Project,
            Field(description="Exact validated PEP 621 project object"),
        ]
        flext: Annotated[
            FlextModelsProjectMetadataAggregates.ProjectToolFlext,
            Field(description="Exact validated tool.flext object"),
        ]
