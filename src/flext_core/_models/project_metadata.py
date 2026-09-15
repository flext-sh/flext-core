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
    _PyprojectIngressContract,
)
from ._project_metadata_parts.flextmodelsprojectmetadata_part_03 import (
    _ProjectMetadataAggregates,
)
from ._project_metadata_parts.flextmodelsprojectmetadata_part_04 import (
    _ProjectMetadataDocument,
)


class FlextModelsProjectMetadata(_ProjectMetadataDocument):
    """Public project metadata model facade."""

    class PyprojectDocument(_PyprojectIngressContract):
        """Complete validated project document ingress."""

        project: Annotated[
            _ProjectMetadataAggregates.Project | None,
            Field(default=None, description="Optional PEP 621 project table"),
        ] = None
        tool: Annotated[
            _ProjectMetadataDocument.PyprojectTool,
            Field(
                default_factory=_ProjectMetadataDocument.PyprojectTool,
                description="Owned tool tables",
            ),
        ] = Field(default_factory=_ProjectMetadataDocument.PyprojectTool)


__all__: list[str] = ["FlextModelsProjectMetadata"]
