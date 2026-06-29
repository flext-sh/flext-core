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

from types import MappingProxyType
from typing import Annotated, ClassVar

from pydantic import (
    Field,
)

from flext_core import FlextTypingBase as tb
from flext_core._models.pydantic import FlextModelsPydantic

from .flextmodelsprojectmetadata_part_02 import (
    FlextModelsProjectMetadata as FlextModelsProjectMetadataPart02,
)


class FlextModelsProjectMetadata(FlextModelsProjectMetadataPart02):
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
            tb.StrSequence,
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


__all__: list[str] = ["FlextModelsProjectMetadata"]
