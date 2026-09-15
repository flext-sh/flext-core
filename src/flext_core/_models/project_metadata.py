"""Declaration-only Pydantic v2 project metadata contracts.

The ingress document retains the exact validated PEP 621 and ``tool.flext``
objects. Derived filesystem and naming values are added only by ``u`` when it
builds ``ProjectMetadata``; models never compute, normalize, or copy them.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from ._project_metadata_parts.flextmodelsprojectmetadata_part_01 import (
    PyprojectIngressContract,
)
from ._project_metadata_parts.flextmodelsprojectmetadata_part_03 import (
    ProjectMetadataAggregates,
)
from ._project_metadata_parts.flextmodelsprojectmetadata_part_04 import (
    ProjectMetadataDocument,
)


class FlextModelsProjectMetadata(ProjectMetadataDocument):
    """Public project metadata model facade."""

    class PyprojectDocument(PyprojectIngressContract):
        """Complete validated project document ingress."""

        project: Annotated[
            ProjectMetadataAggregates.Project | None,
            Field(default=None, description="Optional PEP 621 project table"),
        ] = None
        tool: Annotated[
            ProjectMetadataDocument.PyprojectTool,
            Field(
                default_factory=ProjectMetadataDocument.PyprojectTool,
                description="Owned tool tables",
            ),
        ] = Field(default_factory=ProjectMetadataDocument.PyprojectTool)


__all__: list[str] = ["FlextModelsProjectMetadata"]
