"""Project metadata model parts.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import Field

from flext_core._models._project_metadata import aggregates, contract, pyproject_ingress


class FlextModelsProjectMetadataDocument(
    aggregates.FlextModelsProjectMetadataAggregates
):
    """Validated TOML document sub-tables and canonical domain aggregate."""

    class PyprojectTool(pyproject_ingress.FlextModelsPyprojectIngressContract):
        """Owned subset of the top-level ``[tool]`` table."""

        flext: Annotated[
            aggregates.FlextModelsProjectMetadataAggregates.ProjectToolFlext,
            Field(
                default_factory=aggregates.FlextModelsProjectMetadataAggregates.ProjectToolFlext,
                description="Validated FLEXT project policy",
            ),
        ] = Field(
            default_factory=aggregates.FlextModelsProjectMetadataAggregates.ProjectToolFlext
        )

    class ProjectMetadata(contract.FlextModelsProjectMetadataContract):
        """Canonical project metadata retaining exact validated source objects."""

        root: Annotated[Path, Field(description="Project root")]
        package_name: Annotated[str, Field(min_length=1, description="Import package")]
        class_stem: Annotated[str, Field(min_length=1, description="Class stem")]
        project: Annotated[
            aggregates.FlextModelsProjectMetadataAggregates.Project,
            Field(description="Exact validated PEP 621 project object"),
        ]
        flext: Annotated[
            aggregates.FlextModelsProjectMetadataAggregates.ProjectToolFlext,
            Field(description="Exact validated tool.flext object"),
        ]


__all__: list[str] = ["FlextModelsProjectMetadataDocument"]
