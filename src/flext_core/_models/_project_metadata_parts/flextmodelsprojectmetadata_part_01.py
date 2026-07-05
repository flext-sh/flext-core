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
from typing import Annotated, ClassVar

from pydantic import (
    Field,
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
        """The generic PascalCase class stem for metadata models."""
        if not project_name:
            msg = "empty project name"
            raise ValueError(msg)
        normalized = project_name.lower()
        override: str | None = FlextConstantsProjectMetadata.SPECIAL_NAME_OVERRIDES.get(
            normalized,
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
            tb.StrSequence,
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
            """The Python package name (``flext-ldif`` → ``flext_ldif``)."""
            return self.name.replace("-", "_")

        @property
        def class_stem(self) -> str:
            """The canonical PascalCase class stem (SSOT-derived)."""
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
        PACKAGE_AUTHORS: tb.StrSequence = ()
        PACKAGE_ROOT: Path
        PYTHON_PACKAGE_NAME: Annotated[str, Field(min_length=1)]
        CLASS_STEM: Annotated[str, Field(min_length=1)]
        ALIAS_TO_SUFFIX: tb.StrMapping
        RUNTIME_ALIAS_NAMES: frozenset[str]
        FACADE_ALIAS_NAMES: frozenset[str]
        FACADE_MODULE_NAMES: frozenset[str]
        UNIVERSAL_ALIAS_PARENT_SOURCES: tb.StrMapping
        TIER_FACADE_PREFIX: tb.StrMapping
        SCAN_DIRECTORIES: tb.StrSequence
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


__all__: list[str] = ["FlextModelsProjectMetadata"]
