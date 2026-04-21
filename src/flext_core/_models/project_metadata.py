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

from collections.abc import (
    Mapping,
)
from pathlib import Path
from typing import Annotated, Any, ClassVar

from flext_core import (
    FlextConstantsProjectMetadata as t,
    FlextModelsPydantic,
)
from pydantic import Field, model_validator


class FlextModelsProjectMetadata:
    """Namespace holder for project-metadata SSOT models.

    Each nested model bubbles up flat onto ``m.*`` via the MRO facade
    wiring in ``flext_core/models.py``.
    """

    class ProjectMetadata(FlextModelsPydantic.BaseModel):
        """Canonical per-project metadata (name, version, license, derived names)."""

        model_config: ClassVar[FlextModelsPydantic.ConfigDict] = (
            FlextModelsPydantic.ConfigDict(frozen=True, extra="forbid")
        )

        name: Annotated[
            str,
            Field(
                min_length=1,
                description="Kebab-case project name (e.g. flext-ldif).",
            ),
        ]
        version: Annotated[
            str,
            Field(min_length=1, description="Project version string."),
        ]
        license: Annotated[
            str,
            Field(min_length=1, description="SPDX license identifier or free text."),
        ]
        root: Annotated[
            Path,
            Field(description="Filesystem path to the project root."),
        ]
        description: Annotated[
            str,
            Field(default="", description="Free-text project description."),
        ] = ""
        authors: Annotated[
            tuple[str, ...],
            Field(default=(), description="Author/maintainer display names."),
        ] = ()
        url: Annotated[
            str,
            Field(default="", description="Project homepage URL."),
        ] = ""
        requires_python: Annotated[
            str,
            Field(
                default="",
                description="Extracted Python version from requires-python (e.g. '3.13').",
            ),
        ] = ""

        @property
        def package_name(self) -> str:
            """Return the Python package name (``flext-ldif`` → ``flext_ldif``)."""
            return self.name.replace("-", "_")

        @property
        def class_stem(self) -> str:
            """Return the canonical PascalCase class stem (SSOT-derived)."""
            override = t.SPECIAL_NAME_OVERRIDES.get(self.name)
            if override is not None:
                return override
            parts = self.name.replace("-", "_").split("_")
            return "".join(part[:1].upper() + part[1:] for part in parts if part)

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
            tuple[str, ...],
            Field(
                default=t.SCAN_DIRECTORIES,
                description="Top-level directories to scan for facades.",
            ),
        ] = t.SCAN_DIRECTORIES
        include_dynamic_dirs: Annotated[
            bool,
            Field(
                default=False,
                description="Also scan dynamically created directories.",
            ),
        ] = False
        alias_parent_sources: Annotated[
            Mapping[str, str],
            Field(
                default_factory=dict,
                description="Per-alias parent package source overrides.",
            ),
        ] = Field(default_factory=dict)

        @model_validator(mode="before")
        @classmethod
        def _merge_alias_sources(cls, data: Any) -> Any:
            """Reject unknown aliases; merge universal sources into user input."""
            if not isinstance(data, dict):
                return data
            sources = dict(data.get("alias_parent_sources") or {})
            unknown = set(sources) - set(t.RUNTIME_ALIAS_NAMES)
            if unknown:
                msg = f"unknown alias(es): {sorted(unknown)}"
                raise ValueError(msg)
            for alias, canonical in t.UNIVERSAL_ALIAS_PARENT_SOURCES.items():
                if alias in sources and sources[alias] != canonical:
                    msg = (
                        f"cannot override universal alias {alias!r}: "
                        f"must remain {canonical!r}"
                    )
                    raise ValueError(msg)
            merged = {**t.UNIVERSAL_ALIAS_PARENT_SOURCES, **sources}
            data["alias_parent_sources"] = dict(merged)
            return data

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

    class ProjectToolFlextNamespace(FlextModelsPydantic.BaseModel):
        """``[tool.flext.namespace]`` table contract."""

        model_config: ClassVar[FlextModelsPydantic.ConfigDict] = (
            FlextModelsPydantic.ConfigDict(frozen=True, extra="forbid")
        )

        enabled: Annotated[
            bool,
            Field(default=True, description="Enable namespace enforcement."),
        ] = True
        scan_dirs: Annotated[
            tuple[str, ...],
            Field(
                default=t.SCAN_DIRECTORIES,
                description="Top-level directories to scan.",
            ),
        ] = t.SCAN_DIRECTORIES
        include_dynamic_dirs: Annotated[
            bool,
            Field(
                default=False,
                description="Also scan dynamically created dirs.",
            ),
        ] = False
        alias_parent_sources: Annotated[
            Mapping[str, str],
            Field(
                default_factory=dict,
                description="Per-alias parent package overrides.",
            ),
        ] = Field(default_factory=dict)

    class ProjectToolFlextDocs(FlextModelsPydantic.BaseModel):
        """``[tool.flext.docs]`` table contract."""

        model_config: ClassVar[FlextModelsPydantic.ConfigDict] = (
            FlextModelsPydantic.ConfigDict(frozen=True, extra="forbid")
        )

        project_class: Annotated[
            str,
            Field(default="library", description="Docs project archetype."),
        ] = "library"
        site_title: Annotated[
            str | None,
            Field(
                default=None,
                description="Optional human-readable site title override.",
            ),
        ] = None

    class ProjectToolFlextAliases(FlextModelsPydantic.BaseModel):
        """``[tool.flext.aliases]`` table contract."""

        model_config: ClassVar[FlextModelsPydantic.ConfigDict] = (
            FlextModelsPydantic.ConfigDict(frozen=True, extra="forbid")
        )

        overrides: Annotated[
            Mapping[str, str],
            Field(
                default_factory=dict,
                description="Per-alias type override strings.",
            ),
        ] = Field(default_factory=dict)

    class ProjectToolFlext(FlextModelsPydantic.BaseModel):
        """``[tool.flext]`` root contract aggregating the sub-tables."""

        model_config: ClassVar[FlextModelsPydantic.ConfigDict] = (
            FlextModelsPydantic.ConfigDict(frozen=True, extra="forbid")
        )

        project: Annotated[
            FlextModelsProjectMetadata.ProjectToolFlextProject,
            Field(
                default_factory=lambda: (
                    FlextModelsProjectMetadata.ProjectToolFlextProject()
                ),
                description="[tool.flext.project] sub-table.",
            ),
        ] = Field(
            default_factory=lambda: FlextModelsProjectMetadata.ProjectToolFlextProject()
        )
        namespace: Annotated[
            FlextModelsProjectMetadata.ProjectToolFlextNamespace,
            Field(
                default_factory=lambda: (
                    FlextModelsProjectMetadata.ProjectToolFlextNamespace()
                ),
                description="[tool.flext.namespace] sub-table.",
            ),
        ] = Field(
            default_factory=lambda: (
                FlextModelsProjectMetadata.ProjectToolFlextNamespace()
            )
        )
        docs: Annotated[
            FlextModelsProjectMetadata.ProjectToolFlextDocs,
            Field(
                default_factory=lambda: (
                    FlextModelsProjectMetadata.ProjectToolFlextDocs()
                ),
                description="[tool.flext.docs] sub-table.",
            ),
        ] = Field(
            default_factory=lambda: FlextModelsProjectMetadata.ProjectToolFlextDocs()
        )
        aliases: Annotated[
            FlextModelsProjectMetadata.ProjectToolFlextAliases,
            Field(
                default_factory=lambda: (
                    FlextModelsProjectMetadata.ProjectToolFlextAliases()
                ),
                description="[tool.flext.aliases] sub-table.",
            ),
        ] = Field(
            default_factory=lambda: FlextModelsProjectMetadata.ProjectToolFlextAliases()
        )
