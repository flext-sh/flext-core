"""Project metadata model parts.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import Field

from flext_core._models._project_metadata_parts import (
    flextmodelsprojectmetadata_part_01,
    flextmodelsprojectmetadata_part_03,
    flextmodelsprojectmetadata_part_05,
)



class FlextModelsProjectMetadataDocument(flextmodelsprojectmetadata_part_03.FlextModelsProjectMetadataAggregates):
    """Validated TOML document sub-tables and canonical domain aggregate."""

    class PyprojectTool(flextmodelsprojectmetadata_part_05.FlextModelsPyprojectIngressContract):
        """Owned subset of the top-level ``[tool]`` table."""

        flext: Annotated[
            flextmodelsprojectmetadata_part_03.FlextModelsProjectMetadataAggregates.ProjectToolFlext,
            Field(
                default_factory=flextmodelsprojectmetadata_part_03.FlextModelsProjectMetadataAggregates.ProjectToolFlext,
                description="Validated FLEXT project policy",
            ),
        ] = Field(default_factory=flextmodelsprojectmetadata_part_03.FlextModelsProjectMetadataAggregates.ProjectToolFlext)

    class ProjectMetadata(flextmodelsprojectmetadata_part_01.FlextModelsProjectMetadataContract):
        """Canonical project metadata retaining exact validated source objects."""

        root: Annotated[Path, Field(description="Project root")]
        package_name: Annotated[str, Field(min_length=1, description="Import package")]
        class_stem: Annotated[str, Field(min_length=1, description="Class stem")]
        project: Annotated[
            flextmodelsprojectmetadata_part_03.FlextModelsProjectMetadataAggregates.Project,
            Field(description="Exact validated PEP 621 project object"),
        ]
        flext: Annotated[
            flextmodelsprojectmetadata_part_03.FlextModelsProjectMetadataAggregates.ProjectToolFlext,
            Field(description="Exact validated tool.flext object"),
        ]
