"""Pydantic v2 project-metadata SSOT models (flat on ``m.*``).

Tier 3 domain models owned by flext-core. Consumed across the monorepo
as ``m.ProjectMetadata`` / ``m.ProjectNamespaceConfig`` /
``m.ProjectToolFlext`` / ``m.ProjectToolFlextProject`` / etc. Immutable
by construction (``frozen=True``, ``extra="forbid"``) so callers cannot
drift metadata after load.

Architecture: Tier 3 — imports Tier 0 (_constants/pydantic) only.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Annotated, ClassVar

from pydantic import (
    BeforeValidator,
    Field,
    field_validator,
)

from flext_core import FlextTypingBase as tb
from flext_core._models.pydantic import FlextModelsPydantic

from .flextmodelsprojectmetadata_part_01 import (
    FlextModelsProjectMetadata as FlextModelsProjectMetadataPart01,
)


class FlextModelsProjectMetadata(FlextModelsProjectMetadataPart01):
    class PyprojectProject(FlextModelsPydantic.BaseModel):
        """PEP 621 ``[project]`` table normalized for ``ProjectMetadata``."""

        model_config: ClassVar[FlextModelsPydantic.ConfigDict] = (
            FlextModelsPydantic.ConfigDict(
                frozen=True,
                extra="ignore",
                populate_by_name=True,
            )
        )

        name: Annotated[str, Field(min_length=1)]
        version: Annotated[str, Field(min_length=1)]
        license: Annotated[
            str,
            Field(default="UNLICENSED", min_length=1),
            BeforeValidator(
                lambda v: (
                    str(v.get("text") or "UNLICENSED")
                    if isinstance(v, dict)
                    else (str(v) if v is not None else "UNLICENSED")
                )
            ),
        ] = "UNLICENSED"
        description: str = ""
        authors: Annotated[
            tb.StrSequence,
            Field(default=()),
            BeforeValidator(
                lambda value: (
                    tuple(
                        str(entry.get("name", ""))
                        if isinstance(entry, dict)
                        else str(entry)
                        for entry in value
                    )
                    if isinstance(value, tb.SEQUENCE_PAIR_TYPES)
                    else ()
                )
            ),
        ] = ()
        urls: Annotated[
            tb.StrMapping,
            Field(default_factory=lambda: MappingProxyType({})),
            BeforeValidator(
                lambda v: (
                    {k: str(val) for k, val in v.items()}
                    if isinstance(v, dict)
                    else MappingProxyType({})
                )
            ),
        ] = Field(default_factory=lambda: MappingProxyType({}))
        requires_python: str = Field(default="", alias="requires-python")

        @field_validator("requires_python", mode="after")
        @classmethod
        def _normalize_requires_python(cls, value: str) -> str:
            return value.lstrip(">= ").split(",")[0].split("<")[0].strip()

        def to_metadata(self, root: Path) -> FlextModelsProjectMetadata.ProjectMetadata:
            """Convert the PEP 621 table into the canonical metadata model."""
            return FlextModelsProjectMetadata.ProjectMetadata(
                name=self.name,
                version=self.version,
                license=self.license,
                root=root,
                description=self.description,
                authors=self.authors,
                url=self.urls.get("Homepage", ""),
                requires_python=self.requires_python,
            )

    class ProjectNamespaceConfig(FlextModelsPydantic.BaseModel):
        """Effective namespace configuration for a project.

        Merges project-level overrides with the universal alias sources
        (``r``/``e``/``d``/``x`` always from ``flext_core``).
        """

        model_config: ClassVar[FlextModelsPydantic.ConfigDict] = (
            FlextModelsPydantic.ConfigDict(frozen=True, extra="forbid")
        )

        project_name: Annotated[
            str,
            Field(min_length=1, description="Owning project's kebab-case name."),
        ]
        enabled: Annotated[
            bool,
            Field(default=True, description="Whether namespace enforcement is active."),
        ] = True
        scan_dirs: Annotated[
            tb.StrSequence,
            Field(
                default=(),
                description="Top-level directories to scan for facades.",
            ),
        ] = ()
        include_dynamic_dirs: Annotated[
            bool,
            Field(
                default=False,
                description="Also scan dynamically created directories.",
            ),
        ] = False
        alias_parent_sources: Annotated[
            tb.StrMapping,
            Field(
                default_factory=lambda: MappingProxyType({}),
                description="Per-alias parent package source overrides.",
            ),
        ] = Field(default_factory=lambda: MappingProxyType({}))

    class ProjectToolFlextProject(FlextModelsPydantic.BaseModel):
        """``[tool.flext.project]`` table contract."""

        model_config: ClassVar[FlextModelsPydantic.ConfigDict] = (
            FlextModelsPydantic.ConfigDict(frozen=True, extra="forbid")
        )

        class_stem_override: Annotated[
            str | None,
            Field(
                default=None,
                description="Override the SSOT-derived class stem (rare).",
            ),
        ] = None
        project_class: Annotated[
            str,
            Field(
                default="library",
                description="Project archetype (library / platform / integration).",
            ),
        ] = "library"


__all__: list[str] = ["FlextModelsProjectMetadata"]
