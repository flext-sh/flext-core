"""Project metadata model parts.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import Field

from .flextmodelsprojectmetadata_part_01 import (
    PyprojectIngressContract,
    _ProjectMetadataContract,
)
from .flextmodelsprojectmetadata_part_03 import ProjectMetadataAggregates


class ProjectMetadataDocument(ProjectMetadataAggregates):
    """Validated TOML document sub-tables and canonical domain aggregate."""

    class PyprojectTool(PyprojectIngressContract):
        """Owned subset of the top-level ``[tool]`` table."""

        flext: Annotated[
            ProjectMetadataAggregates.ProjectToolFlext,
            Field(
                default_factory=ProjectMetadataAggregates.ProjectToolFlext,
                description="Validated FLEXT project policy",
            ),
        ] = Field(default_factory=ProjectMetadataAggregates.ProjectToolFlext)

    class ProjectMetadata(_ProjectMetadataContract):
        """Canonical project metadata retaining exact validated source objects."""

        root: Annotated[Path, Field(description="Project root")]
        package_name: Annotated[str, Field(min_length=1, description="Import package")]
        class_stem: Annotated[str, Field(min_length=1, description="Class stem")]
        project: Annotated[
            ProjectMetadataAggregates.Project,
            Field(description="Exact validated PEP 621 project object"),
        ]
        flext: Annotated[
            ProjectMetadataAggregates.ProjectToolFlext,
            Field(description="Exact validated tool.flext object"),
        ]
