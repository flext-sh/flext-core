"""FlextUtilitiesProjectMetadata — Tier 4 SSOT utilities (flat on ``u.*``).

Static helpers for deriving project metadata and reading standardized
pyproject.toml tables. Every reader/writer across the monorepo routes
through this class via ``u.*`` (``u.derive_class_stem``,
``u.read_project_metadata``, etc.). Never hand-roll name derivation or
pyproject parsing — duplication is forbidden.

Architecture: Tier 4 — depends on Tier 0 (_constants) and Tier 3 (_models).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import tomllib
from collections.abc import (
    Mapping,
)
from pathlib import Path
from typing import Any, ClassVar

from flext_core import (
    FlextConstantsProjectMetadata as _k,
    FlextModelsProjectMetadata as _m,
)


class FlextUtilitiesProjectMetadata:
    """SSOT utilities for project-metadata derivation and pyproject reads.

    Inherited into ``FlextUtilities`` via MRO so each static method is
    callable as ``u.pascalize`` / ``u.derive_class_stem`` /
    ``u.read_project_metadata`` / etc. (flat access).
    """

    _flext_enforcement_exempt: ClassVar[bool] = True

    PYPROJECT_FILENAME: ClassVar[str] = "pyproject.toml"

    @staticmethod
    def derive_package_name(project_name: str) -> str:
        """Return the Python package name (``flext-ldif`` → ``flext_ldif``)."""
        if not project_name:
            msg = "empty project name"
            raise ValueError(msg)
        return project_name.replace("-", "_")

    @staticmethod
    def pascalize(slug: str) -> str:
        """Simple kebab/snake → PascalCase (no project-name override lookup).

        Use this when the input is a Python package segment or an
        arbitrary identifier — NOT a full kebab-case project name.
        For project names use ``derive_class_stem`` which additionally
        applies ``SPECIAL_NAME_OVERRIDES`` (flext → FlextRoot etc.).
        """
        parts = slug.replace("-", "_").split("_")
        return "".join(part[:1].upper() + part[1:] for part in parts if part)

    @staticmethod
    def derive_class_stem(project_name: str) -> str:
        """Return the canonical PascalCase class stem for a project name.

        Accepts either kebab-case (``flext-core``) or snake-case
        (``flext_core``) — inputs are normalized before the
        ``SPECIAL_NAME_OVERRIDES`` lookup so that the SSOT is robust to
        both Python package names and pyproject project names.
        """
        if not project_name:
            msg = "empty project name"
            raise ValueError(msg)
        normalized = project_name.replace("_", "-").lower()
        override = _k.SPECIAL_NAME_OVERRIDES.get(normalized)
        if override is not None:
            return override
        return FlextUtilitiesProjectMetadata.pascalize(normalized)

    @staticmethod
    def derive_tier_facade_name(project_name: str, tier: str) -> str:
        """Build the tier-specific facade class name (src/tests/examples/...)."""
        prefix = _k.TIER_FACADE_PREFIX.get(tier)
        if prefix is None:
            msg = f"unknown tier: {tier!r}"
            raise ValueError(msg)
        stem = FlextUtilitiesProjectMetadata.derive_class_stem(project_name)
        if stem.startswith("Flext") and prefix.endswith("Flext"):
            return prefix + stem[len("Flext") :]
        return prefix + stem

    @staticmethod
    def _load_pyproject_toml(root: Path) -> Mapping[str, Any]:
        """Load and return the parsed pyproject.toml under ``root``."""
        pyproject = root / FlextUtilitiesProjectMetadata.PYPROJECT_FILENAME
        if not pyproject.is_file():
            msg = (
                f"{FlextUtilitiesProjectMetadata.PYPROJECT_FILENAME} "
                f"not found under {root}"
            )
            raise FileNotFoundError(msg)
        with pyproject.open("rb") as stream:
            return tomllib.load(stream)

    @staticmethod
    def read_project_metadata(root: Path) -> _m.ProjectMetadata:
        """Read canonical project metadata from a project's pyproject.toml."""
        data = FlextUtilitiesProjectMetadata._load_pyproject_toml(root)
        project = dict(data.get("project") or {})
        if "name" not in project:
            msg = f"{root}: missing [project].name in pyproject.toml"
            raise ValueError(msg)
        if "version" not in project:
            msg = f"{root}: missing [project].version in pyproject.toml"
            raise ValueError(msg)
        license_field = project.get("license")
        if isinstance(license_field, Mapping):
            license_text = str(license_field.get("text", "UNLICENSED"))
        elif license_field is None:
            license_text = "UNLICENSED"
        else:
            license_text = str(license_field)
        authors_raw = project.get("authors") or ()
        authors = tuple(
            str(entry.get("name", "")) if isinstance(entry, Mapping) else str(entry)
            for entry in authors_raw
        )
        urls = project.get("urls") or {}
        url = str(urls.get("Homepage", "")) if isinstance(urls, Mapping) else ""
        requires_python_raw = project.get("requires-python", "")
        requires_python = (
            str(requires_python_raw).lstrip(">= ").split(",")[0].split("<")[0].strip()
            if requires_python_raw
            else ""
        )
        return _m.ProjectMetadata(
            name=str(project["name"]),
            version=str(project["version"]),
            license=license_text,
            root=root,
            description=str(project.get("description", "")),
            authors=authors,
            url=url,
            requires_python=requires_python,
        )

    @staticmethod
    def read_tool_flext_config(root: Path) -> _m.ProjectToolFlext:
        """Read ``[tool.flext.*]`` tables from pyproject.toml."""
        data = FlextUtilitiesProjectMetadata._load_pyproject_toml(root)
        tool = data.get("tool") or {}
        tool_flext = tool.get("flext") if isinstance(tool, Mapping) else None
        return _m.ProjectToolFlext.model_validate(tool_flext or {})

    @staticmethod
    def compose_namespace_config(root: Path) -> _m.ProjectNamespaceConfig:
        """Build the effective namespace config for a project."""
        meta = FlextUtilitiesProjectMetadata.read_project_metadata(root)
        cfg = FlextUtilitiesProjectMetadata.read_tool_flext_config(root)
        return _m.ProjectNamespaceConfig(
            project_name=meta.name,
            enabled=cfg.namespace.enabled,
            scan_dirs=cfg.namespace.scan_dirs,
            include_dynamic_dirs=cfg.namespace.include_dynamic_dirs,
            alias_parent_sources=dict(cfg.namespace.alias_parent_sources),
        )
