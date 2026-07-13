"""Declaration-only Pydantic v2 project metadata contracts.

The ingress document retains the exact validated PEP 621 and ``tool.flext``
objects. Derived filesystem and naming values are added only by ``u`` when it
builds ``ProjectMetadata``; models never compute, normalize, or copy them.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, ClassVar

from pydantic import AliasChoices, Field

from flext_core._models.pydantic import FlextModelsPydantic
from flext_core._typings.base import FlextTypingBase as t


class _ProjectMetadataContract(FlextModelsPydantic.BaseModel):
    """Frozen declaration base for owned project metadata."""

    model_config: ClassVar[FlextModelsPydantic.ConfigDict] = (
        FlextModelsPydantic.ConfigDict(frozen=True, extra="forbid")
    )


class _PyprojectIngressContract(_ProjectMetadataContract):
    """Frozen declaration base for standards-owned TOML tables."""

    model_config: ClassVar[FlextModelsPydantic.ConfigDict] = (
        FlextModelsPydantic.ConfigDict(
            frozen=True, extra="ignore", populate_by_name=True
        )
    )


# NOTE (multi-agent, mro-wkii.17.23 / agent: uv_overlay_owner): one non-part,
# declaration-only owner replaces the method-bearing split model hierarchy.
class _ProjectMetadataFields:
    """Leaf field contracts shared by the aggregate model layers."""

    class ProjectAuthor(_PyprojectIngressContract):
        """One PEP 621 project author."""

        name: Annotated[str, Field(default="", description="Author display name")] = ""
        email: Annotated[str, Field(default="", description="Author email address")] = (
            ""
        )

    class ProjectUrls(_PyprojectIngressContract):
        """Canonical project URL fields from the PEP 621 URL table."""

        homepage: Annotated[
            str,
            Field(
                default="",
                validation_alias=AliasChoices("Homepage", "homepage"),
                description="Project homepage URL",
            ),
        ] = ""
        documentation: Annotated[
            str,
            Field(
                default="",
                validation_alias=AliasChoices("Documentation", "documentation"),
                description="Published documentation URL",
            ),
        ] = ""
        repository: Annotated[
            str,
            Field(
                default="",
                validation_alias=AliasChoices("Repository", "repository"),
                description="Source repository URL",
            ),
        ] = ""

    class ProjectToolFlextProject(_ProjectMetadataContract):
        """``[tool.flext.project]`` contract."""

        class_stem_override: Annotated[
            str | None, Field(default=None, description="Explicit class stem override")
        ] = None

    class ProjectToolFlextDocs(_ProjectMetadataContract):
        """``[tool.flext.docs]`` contract."""

        package_name: Annotated[
            str | None,
            Field(default=None, description="Explicit import package override"),
        ] = None
        project_class: Annotated[
            str, Field(default="library", description="Documentation project class")
        ] = "library"
        site_title: Annotated[
            str | None,
            Field(default=None, description="Documentation site title override"),
        ] = None
        exclude_docs: Annotated[
            t.StrTuple,
            Field(default=(), description="Documentation exclusion patterns"),
        ] = ()

    class ProjectToolFlextWorkspace(_ProjectMetadataContract):
        """``[tool.flext.workspace]`` contract."""

        attached: Annotated[
            bool,
            Field(default=False, description="Attach project to its parent workspace"),
        ] = False


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


class _ProjectMetadataDocument(_ProjectMetadataAggregates):
    """Validated TOML document sub-tables and canonical domain aggregate."""

    class PyprojectTool(_PyprojectIngressContract):
        """Owned subset of the top-level ``[tool]`` table."""

        flext: Annotated[
            _ProjectMetadataAggregates.ProjectToolFlext,
            Field(
                default_factory=_ProjectMetadataAggregates.ProjectToolFlext,
                description="Validated FLEXT project policy",
            ),
        ] = Field(default_factory=_ProjectMetadataAggregates.ProjectToolFlext)

    class ProjectMetadata(_ProjectMetadataContract):
        """Canonical project metadata retaining exact validated source objects."""

        root: Annotated[Path, Field(description="Project root")]
        package_name: Annotated[str, Field(min_length=1, description="Import package")]
        class_stem: Annotated[str, Field(min_length=1, description="Class stem")]
        project: Annotated[
            _ProjectMetadataAggregates.Project,
            Field(description="Exact validated PEP 621 project object"),
        ]
        flext: Annotated[
            _ProjectMetadataAggregates.ProjectToolFlext,
            Field(description="Exact validated tool.flext object"),
        ]


class FlextModelsProjectMetadata(_ProjectMetadataDocument):
    """Public project metadata model facade."""

    class PyprojectDocument(_PyprojectIngressContract):
        """Complete validated project document ingress."""

        project: Annotated[
            _ProjectMetadataAggregates.Project,
            Field(description="PEP 621 project table"),
        ]
        tool: Annotated[
            _ProjectMetadataDocument.PyprojectTool,
            Field(
                default_factory=_ProjectMetadataDocument.PyprojectTool,
                description="Owned tool tables",
            ),
        ] = Field(default_factory=_ProjectMetadataDocument.PyprojectTool)


__all__: list[str] = ["FlextModelsProjectMetadata"]
