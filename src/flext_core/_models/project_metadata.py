"""Pydantic v2 project-metadata SSOT models.

Tier 3 domain models owned by flext-core. Consumed across the monorepo
via ``m.Project.*`` (flext-infra generators, flext-tests fixtures,
per-project facades). Immutable by construction (``frozen=True``,
``extra="forbid"``) so callers cannot drift metadata after load.

Architecture: Tier 3 — imports Tier 0 (_constants/_typings/pydantic) only.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, ClassVar

from pydantic import Field, model_validator

from flext_core._constants.project_metadata import (
    FlextConstantsProjectMetadata as _k,
)
from flext_core._models.pydantic import FlextModelsPydantic


class FlextModelsProjectMetadata:
    """Namespace for project-metadata SSOT models.

    Access via ``m.Project.*`` once wired into the models facade. Do not
    reference these inner classes directly from consumer code; always
    route through the facade alias to preserve MRO semantics.
    """

    _flext_enforcement_exempt: ClassVar[bool] = True

    class Project(FlextModelsPydantic.BaseModel):
        """Canonical per-project metadata (name, version, license, derived names)."""

        _flext_enforcement_exempt: ClassVar[bool] = True

        model_config: ClassVar[FlextModelsPydantic.ConfigDict] = (
            FlextModelsPydantic.ConfigDict(frozen=True, extra="forbid")
        )

        name: Annotated[
            str,
            Field(min_length=1, description="Kebab-case project name (e.g. flext-ldif)."),
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

        @property
        def package_name(self) -> str:
            """Return the Python package name (``flext-ldif`` → ``flext_ldif``)."""
            return self.name.replace("-", "_")

        @property
        def class_stem(self) -> str:
            """Return the canonical PascalCase class stem (SSOT-derived)."""
            override = _k.SPECIAL_NAME_OVERRIDES.get(self.name)
            if override is not None:
                return override
            parts = self.name.replace("-", "_").split("_")
            return "".join(part[:1].upper() + part[1:] for part in parts if part)

        @property
        def src_facade_name(self) -> str:
            """Return the src-tier facade class name (``FlextLdif``)."""
            return self.tier_facade_name("src")

        @property
        def tests_facade_name(self) -> str:
            """Return the tests-tier facade class name (``TestsFlextLdif``)."""
            return self.tier_facade_name("tests")

        def tier_facade_name(self, tier: str) -> str:
            """Build the tier-specific facade class name."""
            prefix = _k.TIER_FACADE_PREFIX.get(tier)
            if prefix is None:
                msg = f"unknown tier: {tier!r}"
                raise ValueError(msg)
            stem = self.class_stem
            if stem.startswith("Flext") and prefix.endswith("Flext"):
                return prefix + stem[len("Flext") :]
            return prefix + stem

    class Namespace(FlextModelsPydantic.BaseModel):
        """Effective namespace configuration for a project.

        Merges project-level overrides with the universal alias sources
        (``r``/``e``/``d``/``x`` always from ``flext_core``).
        """

        _flext_enforcement_exempt: ClassVar[bool] = True

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
                default=_k.SCAN_DIRECTORIES,
                description="Top-level directories to scan for facades.",
            ),
        ] = _k.SCAN_DIRECTORIES
        include_dynamic_dirs: Annotated[
            bool,
            Field(
                default=False,
                description="Also scan dynamically created project directories.",
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
            """Reject unknown aliases; merge universal sources into user input.

            Runs BEFORE the frozen model is constructed so we do not need
            to mutate a frozen instance. Validates that universal aliases
            (r/e/d/x) are not overridden, then merges them into the user's
            ``alias_parent_sources`` mapping.
            """
            if not isinstance(data, dict):
                return data
            sources = dict(data.get("alias_parent_sources") or {})
            unknown = set(sources) - set(_k.RUNTIME_ALIAS_NAMES)
            if unknown:
                msg = f"unknown alias(es): {sorted(unknown)}"
                raise ValueError(msg)
            for alias, canonical in _k.UNIVERSAL_ALIAS_PARENT_SOURCES.items():
                if alias in sources and sources[alias] != canonical:
                    msg = (
                        f"cannot override universal alias {alias!r}: "
                        f"must remain {canonical!r}"
                    )
                    raise ValueError(msg)
            merged = {**_k.UNIVERSAL_ALIAS_PARENT_SOURCES, **sources}
            data["alias_parent_sources"] = dict(merged)
            return data

    class ToolFlextProject(FlextModelsPydantic.BaseModel):
        """``[tool.flext.project]`` table contract."""

        _flext_enforcement_exempt: ClassVar[bool] = True

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

    class ToolFlextNamespace(FlextModelsPydantic.BaseModel):
        """``[tool.flext.namespace]`` table contract."""

        _flext_enforcement_exempt: ClassVar[bool] = True

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
                default=_k.SCAN_DIRECTORIES,
                description="Top-level directories to scan.",
            ),
        ] = _k.SCAN_DIRECTORIES
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

    class ToolFlextDocs(FlextModelsPydantic.BaseModel):
        """``[tool.flext.docs]`` table contract."""

        _flext_enforcement_exempt: ClassVar[bool] = True

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

    class ToolFlextAliases(FlextModelsPydantic.BaseModel):
        """``[tool.flext.aliases]`` table contract."""

        _flext_enforcement_exempt: ClassVar[bool] = True

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

    class ToolFlext(FlextModelsPydantic.BaseModel):
        """``[tool.flext]`` root contract aggregating the sub-tables."""

        _flext_enforcement_exempt: ClassVar[bool] = True

        model_config: ClassVar[FlextModelsPydantic.ConfigDict] = (
            FlextModelsPydantic.ConfigDict(frozen=True, extra="forbid")
        )

        project: Annotated[
            FlextModelsProjectMetadata.ToolFlextProject,
            Field(
                default_factory=lambda: FlextModelsProjectMetadata.ToolFlextProject(),
                description="[tool.flext.project] sub-table.",
            ),
        ] = Field(default_factory=lambda: FlextModelsProjectMetadata.ToolFlextProject())
        namespace: Annotated[
            FlextModelsProjectMetadata.ToolFlextNamespace,
            Field(
                default_factory=lambda: FlextModelsProjectMetadata.ToolFlextNamespace(),
                description="[tool.flext.namespace] sub-table.",
            ),
        ] = Field(default_factory=lambda: FlextModelsProjectMetadata.ToolFlextNamespace())
        docs: Annotated[
            FlextModelsProjectMetadata.ToolFlextDocs,
            Field(
                default_factory=lambda: FlextModelsProjectMetadata.ToolFlextDocs(),
                description="[tool.flext.docs] sub-table.",
            ),
        ] = Field(default_factory=lambda: FlextModelsProjectMetadata.ToolFlextDocs())
        aliases: Annotated[
            FlextModelsProjectMetadata.ToolFlextAliases,
            Field(
                default_factory=lambda: FlextModelsProjectMetadata.ToolFlextAliases(),
                description="[tool.flext.aliases] sub-table.",
            ),
        ] = Field(default_factory=lambda: FlextModelsProjectMetadata.ToolFlextAliases())
