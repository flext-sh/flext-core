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

from flext_core._models._project_metadata_parts import (
    flextmodelsprojectmetadata_part_03 as _03,
    flextmodelsprojectmetadata_part_04 as _04,
    flextmodelsprojectmetadata_part_05 as _05,
)

FlextModelsProjectMetadataAggregates = _03.FlextModelsProjectMetadataAggregates

FlextModelsProjectMetadataDocument = _04.FlextModelsProjectMetadataDocument

FlextModelsPyprojectIngressContract = _05.FlextModelsPyprojectIngressContract


class FlextModelsProjectMetadata(FlextModelsProjectMetadataDocument):
    """Public project metadata model facade."""

    class PyprojectDocument(FlextModelsPyprojectIngressContract):
        """Complete validated project document ingress."""

        project: Annotated[
            FlextModelsProjectMetadataAggregates.Project | None,
            Field(default=None, description="Optional PEP 621 project table"),
        ] = None
        tool: Annotated[
            FlextModelsProjectMetadataDocument.PyprojectTool,
            Field(
                default_factory=FlextModelsProjectMetadataDocument.PyprojectTool,
                description="Owned tool tables",
            ),
        ] = Field(default_factory=FlextModelsProjectMetadataDocument.PyprojectTool)


__all__: list[str] = ["FlextModelsProjectMetadata"]
