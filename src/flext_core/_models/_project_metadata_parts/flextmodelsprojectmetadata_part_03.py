"""Project metadata model parts.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from flext_core._models._project_metadata_parts import (
    flextmodelsprojectmetadata_part_01,
    flextmodelsprojectmetadata_part_02,
    flextmodelsprojectmetadata_part_05,
)
from flext_core._typings.base import FlextTypingBase


class FlextModelsProjectMetadataAggregates(
    flextmodelsprojectmetadata_part_02.FlextModelsProjectMetadataFields
):
    """Validated PEP 621 and FLEXT aggregate declarations."""

    class Project(
        flextmodelsprojectmetadata_part_05.FlextModelsPyprojectIngressContract
    ):
        """Complete owned PEP 621 project metadata used by FLEXT."""

        name: Annotated[str, Field(min_length=1)]
        version: Annotated[str, Field(min_length=1)]
        description: str = ""
        authors: Annotated[
            tuple[
                flextmodelsprojectmetadata_part_02.FlextModelsProjectMetadataFields.ProjectAuthor,
                ...,
            ],
            Field(default=(), description="Project authors"),
        ] = ()
        urls: Annotated[
            flextmodelsprojectmetadata_part_02.FlextModelsProjectMetadataFields.ProjectUrls,
            Field(
                default_factory=flextmodelsprojectmetadata_part_02.FlextModelsProjectMetadataFields.ProjectUrls,
                description="Project URLs",
            ),
        ] = Field(
            default_factory=flextmodelsprojectmetadata_part_02.FlextModelsProjectMetadataFields.ProjectUrls
        )
        requires_python: Annotated[
            str,
            Field(default="", alias="requires-python", description="Python constraint"),
        ] = ""
        dependencies: Annotated[
            FlextTypingBase.StrTuple,
            Field(default=(), description="PEP 508 runtime dependency declarations"),
        ] = ()
        classifiers: Annotated[
            FlextTypingBase.StrTuple,
            Field(default=(), description="Trove classifiers"),
        ] = ()
        keywords: Annotated[
            FlextTypingBase.StrTuple,
            Field(default=(), description="Project search keywords"),
        ] = ()

    class ProjectToolFlext(
        flextmodelsprojectmetadata_part_01.FlextModelsProjectMetadataContract
    ):
        """Complete ``[tool.flext]`` contract."""

        project: Annotated[
            flextmodelsprojectmetadata_part_02.FlextModelsProjectMetadataFields.ProjectToolFlextProject,
            Field(
                default_factory=flextmodelsprojectmetadata_part_02.FlextModelsProjectMetadataFields.ProjectToolFlextProject,
                description="Project naming policy",
            ),
        ] = Field(
            default_factory=flextmodelsprojectmetadata_part_02.FlextModelsProjectMetadataFields.ProjectToolFlextProject,
        )
        docs: Annotated[
            flextmodelsprojectmetadata_part_02.FlextModelsProjectMetadataFields.ProjectToolFlextDocs,
            Field(
                default_factory=flextmodelsprojectmetadata_part_02.FlextModelsProjectMetadataFields.ProjectToolFlextDocs,
                description="Documentation policy",
            ),
        ] = Field(
            default_factory=flextmodelsprojectmetadata_part_02.FlextModelsProjectMetadataFields.ProjectToolFlextDocs
        )
        workspace: Annotated[
            flextmodelsprojectmetadata_part_02.FlextModelsProjectMetadataFields.ProjectToolFlextWorkspace,
            Field(
                default_factory=flextmodelsprojectmetadata_part_02.FlextModelsProjectMetadataFields.ProjectToolFlextWorkspace,
                description="Workspace attachment policy",
            ),
        ] = Field(
            default_factory=flextmodelsprojectmetadata_part_02.FlextModelsProjectMetadataFields.ProjectToolFlextWorkspace,
        )
        namespace: Annotated[
            flextmodelsprojectmetadata_part_02.FlextModelsProjectMetadataFields.ProjectToolFlextNamespace,
            Field(
                default_factory=flextmodelsprojectmetadata_part_02.FlextModelsProjectMetadataFields.ProjectToolFlextNamespace,
                description="Namespace enforcement policy",
            ),
        ] = Field(
            default_factory=flextmodelsprojectmetadata_part_02.FlextModelsProjectMetadataFields.ProjectToolFlextNamespace,
        )
