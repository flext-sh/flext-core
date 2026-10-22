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

from flext_core._models._project_metadata import (
    flextmodelsprojectmetadata_part_03,
    flextmodelsprojectmetadata_part_04,
    flextmodelsprojectmetadata_part_05,
)


class FlextModelsProjectMetadata(
    flextmodelsprojectmetadata_part_04.FlextModelsProjectMetadataDocument
):
    """Public project metadata model facade."""

    class PyprojectDocument(
        flextmodelsprojectmetadata_part_05.FlextModelsPyprojectIngressContract
    ):
        """Complete validated project document ingress."""

        project: Annotated[
            flextmodelsprojectmetadata_part_03.FlextModelsProjectMetadataAggregates.Project
            | None,
            Field(default=None, description="Optional PEP 621 project table"),
        ] = None
        tool: Annotated[
            flextmodelsprojectmetadata_part_04.FlextModelsProjectMetadataDocument.PyprojectTool,
            Field(
                default_factory=flextmodelsprojectmetadata_part_04.FlextModelsProjectMetadataDocument.PyprojectTool,
                description="Owned tool tables",
            ),
        ] = Field(
            default_factory=flextmodelsprojectmetadata_part_04.FlextModelsProjectMetadataDocument.PyprojectTool,
        )


__all__: list[str] = ["FlextModelsProjectMetadata"]
