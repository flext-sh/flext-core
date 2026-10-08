"""Project metadata model parts.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from flext_core._models._project_metadata_parts import (
    flextmodelsprojectmetadata_part_01 as _01,
    flextmodelsprojectmetadata_part_02 as _02,
    flextmodelsprojectmetadata_part_05 as _05,
)
from flext_core._typings.base import FlextTypingBase as t

FlextModelsProjectMetadataContract = _01.FlextModelsProjectMetadataContract

FlextModelsProjectMetadataFields = _02.FlextModelsProjectMetadataFields

FlextModelsPyprojectIngressContract = _05.FlextModelsPyprojectIngressContract


class FlextModelsProjectMetadataAggregates(FlextModelsProjectMetadataFields):
    """Validated PEP 621 and FLEXT aggregate declarations."""

    class Project(FlextModelsPyprojectIngressContract):
        """Complete owned PEP 621 project metadata used by FLEXT."""

        name: Annotated[str, Field(min_length=1)]
        version: Annotated[str, Field(min_length=1)]
        description: str = ""
        authors: Annotated[
            tuple[FlextModelsProjectMetadataFields.ProjectAuthor, ...],
            Field(default=(), description="Project authors"),
        ] = ()
        urls: Annotated[
            FlextModelsProjectMetadataFields.ProjectUrls,
            Field(
                default_factory=FlextModelsProjectMetadataFields.ProjectUrls,
                description="Project URLs",
            ),
        ] = Field(default_factory=FlextModelsProjectMetadataFields.ProjectUrls)
        requires_python: Annotated[
            str,
            Field(default="", alias="requires-python", description="Python constraint"),
        ] = ""
        dependencies: Annotated[
            t.StrTuple,
            Field(default=(), description="PEP 508 runtime dependency declarations"),
        ] = ()
        classifiers: Annotated[
            t.StrTuple,
            Field(default=(), description="Trove classifiers"),
        ] = ()
        keywords: Annotated[
            t.StrTuple,
            Field(default=(), description="Project search keywords"),
        ] = ()

    class ProjectToolFlext(FlextModelsProjectMetadataContract):
        """Complete ``[tool.flext]`` contract."""

        project: Annotated[
            FlextModelsProjectMetadataFields.ProjectToolFlextProject,
            Field(
                default_factory=FlextModelsProjectMetadataFields.ProjectToolFlextProject,
                description="Project naming policy",
            ),
        ] = Field(
            default_factory=FlextModelsProjectMetadataFields.ProjectToolFlextProject,
        )
        docs: Annotated[
            FlextModelsProjectMetadataFields.ProjectToolFlextDocs,
            Field(
                default_factory=FlextModelsProjectMetadataFields.ProjectToolFlextDocs,
                description="Documentation policy",
            ),
        ] = Field(default_factory=FlextModelsProjectMetadataFields.ProjectToolFlextDocs)
        workspace: Annotated[
            FlextModelsProjectMetadataFields.ProjectToolFlextWorkspace,
            Field(
                default_factory=FlextModelsProjectMetadataFields.ProjectToolFlextWorkspace,
                description="Workspace attachment policy",
            ),
        ] = Field(
            default_factory=FlextModelsProjectMetadataFields.ProjectToolFlextWorkspace,
        )
        namespace: Annotated[
            FlextModelsProjectMetadataFields.ProjectToolFlextNamespace,
            Field(
                default_factory=FlextModelsProjectMetadataFields.ProjectToolFlextNamespace,
                description="Namespace enforcement policy",
            ),
        ] = Field(
            default_factory=FlextModelsProjectMetadataFields.ProjectToolFlextNamespace,
        )
