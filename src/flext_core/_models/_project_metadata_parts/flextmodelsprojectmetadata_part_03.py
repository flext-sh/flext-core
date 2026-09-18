"""Project metadata model parts.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from ..._typings.base import FlextTypingBase as t
from .flextmodelsprojectmetadata_part_01 import (
    ProjectMetadataContract,
    PyprojectIngressContract,
)
from .flextmodelsprojectmetadata_part_02 import ProjectMetadataFields


class ProjectMetadataAggregates(ProjectMetadataFields):
    """Validated PEP 621 and FLEXT aggregate declarations."""

    class Project(PyprojectIngressContract):
        """Complete owned PEP 621 project metadata used by FLEXT."""

        name: Annotated[str, Field(min_length=1)]
        version: Annotated[str, Field(min_length=1)]
        description: str = ""
        authors: Annotated[
            tuple[ProjectMetadataFields.ProjectAuthor, ...],
            Field(default=(), description="Project authors"),
        ] = ()
        urls: Annotated[
            ProjectMetadataFields.ProjectUrls,
            Field(
                default_factory=ProjectMetadataFields.ProjectUrls,
                description="Project URLs",
            ),
        ] = Field(default_factory=ProjectMetadataFields.ProjectUrls)
        requires_python: Annotated[
            str,
            Field(default="", alias="requires-python", description="Python constraint"),
        ] = ""
        dependencies: Annotated[
            t.StrTuple,
            Field(default=(), description="PEP 508 runtime dependency declarations"),
        ] = ()
        classifiers: Annotated[
            t.StrTuple, Field(default=(), description="Trove classifiers")
        ] = ()
        keywords: Annotated[
            t.StrTuple, Field(default=(), description="Project search keywords")
        ] = ()

    class ProjectToolFlext(ProjectMetadataContract):
        """Complete ``[tool.flext]`` contract."""

        project: Annotated[
            ProjectMetadataFields.ProjectToolFlextProject,
            Field(
                default_factory=ProjectMetadataFields.ProjectToolFlextProject,
                description="Project naming policy",
            ),
        ] = Field(default_factory=ProjectMetadataFields.ProjectToolFlextProject)
        docs: Annotated[
            ProjectMetadataFields.ProjectToolFlextDocs,
            Field(
                default_factory=ProjectMetadataFields.ProjectToolFlextDocs,
                description="Documentation policy",
            ),
        ] = Field(default_factory=ProjectMetadataFields.ProjectToolFlextDocs)
        workspace: Annotated[
            ProjectMetadataFields.ProjectToolFlextWorkspace,
            Field(
                default_factory=ProjectMetadataFields.ProjectToolFlextWorkspace,
                description="Workspace attachment policy",
            ),
        ] = Field(default_factory=ProjectMetadataFields.ProjectToolFlextWorkspace)
        namespace: Annotated[
            ProjectMetadataFields.ProjectToolFlextNamespace,
            Field(
                default_factory=ProjectMetadataFields.ProjectToolFlextNamespace,
                description="Namespace enforcement policy",
            ),
        ] = Field(default_factory=ProjectMetadataFields.ProjectToolFlextNamespace)
