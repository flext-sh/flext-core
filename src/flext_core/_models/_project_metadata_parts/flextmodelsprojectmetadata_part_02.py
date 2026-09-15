"""Project metadata model parts.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import PurePosixPath, PureWindowsPath
from typing import Annotated, Self

from pydantic import AliasChoices, Field, field_validator, model_validator

from ..._constants.regex import FlextConstantsRegex as cr
from ..._typings.base import FlextTypingBase as t
from .flextmodelsprojectmetadata_part_01 import (
    _ProjectMetadataContract,
    PyprojectIngressContract,
)


class _ProjectMetadataFields:
    """Leaf field contracts shared by the aggregate model layers."""

    class ProjectAuthor(PyprojectIngressContract):
        """One PEP 621 project author."""

        name: Annotated[str, Field(default="", description="Author display name")] = ""
        email: Annotated[str, Field(default="", description="Author email address")] = (
            ""
        )

    class ProjectUrls(PyprojectIngressContract):
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
        budget: Annotated[
            t.JsonMapping | None,
            Field(
                default=None,
                description=(
                    "Per-gate resource budget table emitted by the flext-infra "
                    "budget-gate projection into the managed "
                    "``[tool.flext.project.budget]`` section. The ingress owns "
                    "the validated shape so the generator's own output always "
                    "round-trips through this contract."
                ),
            ),
        ] = None

    class ProjectToolFlextReadmeSection(_ProjectMetadataContract):
        """One ordered project README section declaration."""

        id: Annotated[
            str,
            Field(
                pattern=cr.PATTERN_IDENTIFIER_LOWERCASE,
                description="Stable project README section identifier",
            ),
        ]
        title: Annotated[
            str, Field(min_length=1, description="Project README section heading")
        ]
        content: Annotated[
            str | None,
            Field(default=None, min_length=1, description="Inline Markdown content"),
        ] = None
        include: Annotated[
            PurePosixPath | None,
            Field(
                default=None,
                description="Portable repository-relative Markdown include path",
            ),
        ] = None

        @field_validator("include", mode="before")
        @classmethod
        def _validate_include_path(cls, value: object) -> object:
            if value is None or not isinstance(value, (str, PurePosixPath)):
                return value
            rendered = str(value)
            path = PurePosixPath(rendered)
            windows_path = PureWindowsPath(rendered)
            if (
                not rendered
                or "\\\\" in rendered
                or path.is_absolute()
                or bool(windows_path.drive)
                or any(part in {"", ".", ".."} for part in rendered.split("/"))
            ):
                msg = "README include path must be a portable repository-relative path"
                raise ValueError(msg)
            return path

        @model_validator(mode="after")
        def _validate_content_source(self) -> Self:
            if (self.content is None) == (self.include is None):
                msg = "README section must declare exactly one of content or include"
                raise ValueError(msg)
            return self

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
        readme_sections: Annotated[
            tuple[_ProjectMetadataFields.ProjectToolFlextReadmeSection, ...],
            Field(default=(), description="Ordered project README sections"),
        ] = ()

        @model_validator(mode="after")
        def _validate_readme_section_ids(self) -> Self:
            section_ids = tuple(section.id for section in self.readme_sections)
            if len(section_ids) != len(set(section_ids)):
                msg = "project README section identifiers must be unique"
                raise ValueError(msg)
            return self

    class ProjectToolFlextWorkspace(_ProjectMetadataContract):
        """``[tool.flext.workspace]`` contract."""

        attached: Annotated[
            bool,
            Field(default=False, description="Attach project to its parent workspace"),
        ] = False

    class ProjectToolFlextNamespace(_ProjectMetadataContract):
        """``[tool.flext.namespace]`` contract."""

        enabled: Annotated[
            bool | None,
            Field(default=None, description="Explicit namespace-enforcement toggle"),
        ] = None
        scan_dirs: Annotated[
            t.StrTuple,
            Field(default=(), description="Explicit namespace scan directories"),
        ] = ()
        include_dynamic_dirs: Annotated[
            bool | None,
            Field(
                default=None,
                description="Whether non-canonical tracked dirs join the scan",
            ),
        ] = None
