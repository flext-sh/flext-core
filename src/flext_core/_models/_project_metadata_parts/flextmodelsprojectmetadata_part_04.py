"""Project metadata model parts.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import Field

from .flextmodelsprojectmetadata_part_01 import (
    _ProjectMetadataContract,
    _PyprojectIngressContract,
)
from .flextmodelsprojectmetadata_part_03 import _ProjectMetadataAggregates


class _ProjectMetadataDocument(_ProjectMetadataAggregates):
    """Validated TOML document sub-tables and canonical domain aggregate."""

    class PyprojectTool(_PyprojectIngressContract):
        """Owned subset of the top-level ``[tool]`` table."""

        flext: Annotated[
            _ProjectMetadataAggregates.ProjectToolFlext,
            Field(
                default_factory=_ProjectMetadataAggregates.ProjectToolFlext,
                description="Validated FLEXT project policy",
            ),
        ] = Field(default_factory=_ProjectMetadataAggregates.ProjectToolFlext)

    class ProjectMetadata(_ProjectMetadataContract):
        """Canonical project metadata retaining exact validated source objects."""

        root: Annotated[Path, Field(description="Project root")]
        package_name: Annotated[str, Field(min_length=1, description="Import package")]
        class_stem: Annotated[str, Field(min_length=1, description="Class stem")]
        project: Annotated[
            _ProjectMetadataAggregates.Project,
            Field(description="Exact validated PEP 621 project object"),
        ]
        flext: Annotated[
            _ProjectMetadataAggregates.ProjectToolFlext,
            Field(description="Exact validated tool.flext object"),
        ]
