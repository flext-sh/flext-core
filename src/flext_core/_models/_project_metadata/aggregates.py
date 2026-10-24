"""Project metadata model parts.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from flext_core import t
from flext_core._models._project_metadata import contract, fields, pyproject_ingress


class FlextModelsProjectMetadataAggregates(fields.FlextModelsProjectMetadataFields):
    """Validated PEP 621 and FLEXT aggregate declarations."""

    class Project(pyproject_ingress.FlextModelsPyprojectIngressContract):
        """Complete owned PEP 621 project metadata used by FLEXT."""

        name: Annotated[str, Field(min_length=1)]
        version: Annotated[str, Field(min_length=1)]
        description: str = ""
        authors: Annotated[
            tuple[
                fields.FlextModelsProjectMetadataFields.ProjectAuthor,
                ...,
            ],
            Field(default=(), description="Project authors"),
        ] = ()
        urls: Annotated[
            fields.FlextModelsProjectMetadataFields.ProjectUrls,
            Field(
                default_factory=fields.FlextModelsProjectMetadataFields.ProjectUrls,
                description="Project URLs",
            ),
        ] = Field(default_factory=fields.FlextModelsProjectMetadataFields.ProjectUrls)
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

    class ProjectToolFlext(contract.FlextModelsProjectMetadataContract):
        """Complete ``[tool.flext]`` contract."""

        project: Annotated[
            fields.FlextModelsProjectMetadataFields.ProjectToolFlextProject,
            Field(
                default_factory=fields.FlextModelsProjectMetadataFields.ProjectToolFlextProject,
                description="Project naming policy",
            ),
        ] = Field(
            default_factory=fields.FlextModelsProjectMetadataFields.ProjectToolFlextProject,
        )
        docs: Annotated[
            fields.FlextModelsProjectMetadataFields.ProjectToolFlextDocs,
            Field(
                default_factory=fields.FlextModelsProjectMetadataFields.ProjectToolFlextDocs,
                description="Documentation policy",
            ),
        ] = Field(
            default_factory=fields.FlextModelsProjectMetadataFields.ProjectToolFlextDocs
        )
        workspace: Annotated[
            fields.FlextModelsProjectMetadataFields.ProjectToolFlextWorkspace,
            Field(
                default_factory=fields.FlextModelsProjectMetadataFields.ProjectToolFlextWorkspace,
                description="Workspace attachment policy",
            ),
        ] = Field(
            default_factory=fields.FlextModelsProjectMetadataFields.ProjectToolFlextWorkspace,
        )
        namespace: Annotated[
            fields.FlextModelsProjectMetadataFields.ProjectToolFlextNamespace,
            Field(
                default_factory=fields.FlextModelsProjectMetadataFields.ProjectToolFlextNamespace,
                description="Namespace enforcement policy",
            ),
        ] = Field(
            default_factory=fields.FlextModelsProjectMetadataFields.ProjectToolFlextNamespace,
        )


__all__: list[str] = ["FlextModelsProjectMetadataAggregates"]
