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

from flext_core._models._project_metadata_parts.flextmodelsprojectmetadata_part_03 import (  # ruff: ignore[line-too-long] -- the dotted module path is a single identifier chain; it cannot be wrapped without renaming the module
    FlextModelsProjectMetadataAggregates,
)
from flext_core._models._project_metadata_parts.flextmodelsprojectmetadata_part_04 import (  # ruff: ignore[line-too-long] -- the dotted module path is a single identifier chain; it cannot be wrapped without renaming the module
    FlextModelsProjectMetadataDocument,
)
from flext_core._models._project_metadata_parts.flextmodelsprojectmetadata_part_05 import (  # ruff: ignore[line-too-long] -- the dotted module path is a single identifier chain; it cannot be wrapped without renaming the module
    FlextModelsPyprojectIngressContract,
)


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
