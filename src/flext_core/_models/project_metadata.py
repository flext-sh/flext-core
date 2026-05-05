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

from flext_core._constants.project_metadata import FlextConstantsProjectMetadata
from flext_core._models.pydantic import FlextModelsPydantic
from flext_core._typings.base import FlextTypingBase as tb


class FlextModelsProjectMetadata:
    """Namespace holder for project-metadata SSOT models.

    Each nested model bubbles up flat onto ``m.*`` via the MRO facade
    wiring in ``flext_core/models.py``.
    """

    @staticmethod
    def pascalize(slug: str) -> str:
        """Kebab/snake → PascalCase. SSOT for project-name derivation."""
        parts = slug.replace("-", "_").split("_")
        return "".join(part[:1].upper() + part[1:] for part in parts if part)

    @staticmethod
    def derive_class_stem(project_name: str) -> str:
        """Return the generic PascalCase class stem for metadata models."""
        if not project_name:
            msg = "empty project name"
            raise ValueError(msg)
        normalized = project_name.lower()
        override: str | None = FlextConstantsProjectMetadata.SPECIAL_NAME_OVERRIDES.get(
            normalized
        )
        if override is not None:
            return override
        return FlextModelsProjectMetadata.pascalize(normalized)

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
            return FlextModelsProjectMetadata.derive_class_stem(self.name)

    class ProjectConstants(FlextModelsPydantic.BaseModel):
        """Constants derived from installed package runtime metadata."""

        model_config: ClassVar[FlextModelsPydantic.ConfigDict] = (
            FlextModelsPydantic.ConfigDict(frozen=True, extra="forbid")
        )

        PACKAGE_NAME: Annotated[str, Field(min_length=1)]
        PACKAGE_VERSION: Annotated[str, Field(min_length=1)]
        PACKAGE_LICENSE: Annotated[str, Field(min_length=1)]
        PACKAGE_URL: str = ""
        PACKAGE_AUTHORS: tuple[str, ...] = ()
        PACKAGE_ROOT: Path
        PYTHON_PACKAGE_NAME: Annotated[str, Field(min_length=1)]
        CLASS_STEM: Annotated[str, Field(min_length=1)]
        ALIAS_TO_SUFFIX: tb.StrMapping
        RUNTIME_ALIAS_NAMES: frozenset[str]
        FACADE_ALIAS_NAMES: frozenset[str]
        FACADE_MODULE_NAMES: frozenset[str]
        UNIVERSAL_ALIAS_PARENT_SOURCES: tb.StrMapping
        TIER_FACADE_PREFIX: tb.StrMapping
        SCAN_DIRECTORIES: tuple[str, ...]
        TIER_SUB_NAMESPACE: tb.StrMapping
        PYPROJECT_FILENAME: Annotated[str, Field(min_length=1)]

    class LazyAliasMetadata(FlextModelsPydantic.BaseModel):
        """Normalized runtime alias metadata derived from generated lazy exports."""

        model_config: ClassVar[FlextModelsPydantic.ConfigDict] = (
            FlextModelsPydantic.ConfigDict(frozen=True, extra="forbid")
        )

        alias: Annotated[str, Field(min_length=1)]
        module_path: Annotated[str, Field(min_length=1)]
        parent_source: Annotated[str, Field(min_length=1)]
        suffix: Annotated[str, Field(min_length=1)]
        facade: bool

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
            tuple[str, ...],
            Field(default=()),
            BeforeValidator(
                lambda value: (
                    tuple(
                        str(entry.get("name", ""))
                        if isinstance(entry, dict)
                        else str(entry)
                        for entry in value
                    )
                    if isinstance(value, (list, tuple))
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
            tuple[str, ...],
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
                default=(),
                description="Top-level directories to scan.",
            ),
        ] = ()
        include_dynamic_dirs: Annotated[
            bool,
            Field(
                default=False,
                description="Also scan dynamically created dirs.",
            ),
        ] = False
        alias_parent_sources: Annotated[
            tb.StrMapping,
            Field(
                default_factory=lambda: MappingProxyType({}),
                description="Per-alias parent package overrides.",
            ),
        ] = Field(default_factory=lambda: MappingProxyType({}))

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
            tb.StrMapping,
            Field(
                default_factory=lambda: MappingProxyType({}),
                description="Per-alias type override strings.",
            ),
        ] = Field(default_factory=lambda: MappingProxyType({}))

    class ProjectToolFlextWorkspace(FlextModelsPydantic.BaseModel):
        """``[tool.flext.workspace]`` table contract."""

        model_config: ClassVar[FlextModelsPydantic.ConfigDict] = (
            FlextModelsPydantic.ConfigDict(frozen=True, extra="forbid")
        )

        attached: Annotated[
            bool,
            Field(
                default=False,
                description=(
                    "Opt this directory into workspace iteration as an attached "
                    "sub-repo. False (default) keeps the dir invisible to "
                    "workspace verbs unless the iterator passes "
                    "include_attached=True."
                ),
            ),
        ] = False

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
        workspace: Annotated[
            FlextModelsProjectMetadata.ProjectToolFlextWorkspace,
            Field(
                default_factory=lambda: (
                    FlextModelsProjectMetadata.ProjectToolFlextWorkspace()
                ),
                description="[tool.flext.workspace] sub-table.",
            ),
        ] = Field(
            default_factory=lambda: (
                FlextModelsProjectMetadata.ProjectToolFlextWorkspace()
            )
        )
