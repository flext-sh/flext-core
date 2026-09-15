"""Project metadata model parts.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from ..._typings.base import FlextTypingBase as t
from .flextmodelsprojectmetadata_part_01 import (
    _ProjectMetadataContract,
    _PyprojectIngressContract,
)
from .flextmodelsprojectmetadata_part_02 import _ProjectMetadataFields


class _ProjectMetadataAggregates(_ProjectMetadataFields):
    """Validated PEP 621 and FLEXT aggregate declarations."""

    class Project(_PyprojectIngressContract):
        """Complete owned PEP 621 project metadata used by FLEXT."""

        name: Annotated[str, Field(min_length=1)]
        version: Annotated[str, Field(min_length=1)]
        description: str = ""
        authors: Annotated[
            tuple[_ProjectMetadataFields.ProjectAuthor, ...],
            Field(default=(), description="Project authors"),
        ] = ()
        urls: Annotated[
            _ProjectMetadataFields.ProjectUrls,
            Field(
                default_factory=_ProjectMetadataFields.ProjectUrls,
                description="Project URLs",
            ),
        ] = Field(default_factory=_ProjectMetadataFields.ProjectUrls)
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

    class ProjectToolFlext(_ProjectMetadataContract):
        """Complete ``[tool.flext]`` contract."""

        project: Annotated[
            _ProjectMetadataFields.ProjectToolFlextProject,
            Field(
                default_factory=_ProjectMetadataFields.ProjectToolFlextProject,
                description="Project naming policy",
            ),
        ] = Field(default_factory=_ProjectMetadataFields.ProjectToolFlextProject)
        docs: Annotated[
            _ProjectMetadataFields.ProjectToolFlextDocs,
            Field(
                default_factory=_ProjectMetadataFields.ProjectToolFlextDocs,
                description="Documentation policy",
            ),
        ] = Field(default_factory=_ProjectMetadataFields.ProjectToolFlextDocs)
        workspace: Annotated[
            _ProjectMetadataFields.ProjectToolFlextWorkspace,
            Field(
                default_factory=_ProjectMetadataFields.ProjectToolFlextWorkspace,
                description="Workspace attachment policy",
            ),
        ] = Field(default_factory=_ProjectMetadataFields.ProjectToolFlextWorkspace)
        namespace: Annotated[
            _ProjectMetadataFields.ProjectToolFlextNamespace,
            Field(
                default_factory=_ProjectMetadataFields.ProjectToolFlextNamespace,
                description="Namespace enforcement policy",
            ),
        ] = Field(default_factory=_ProjectMetadataFields.ProjectToolFlextNamespace)
