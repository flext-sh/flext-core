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
from functools import cache
from pathlib import Path
from typing import Any, ClassVar

from flext_core._constants.project_metadata import FlextConstantsProjectMetadata as cpm
from flext_core._models.project_metadata import FlextModelsProjectMetadata as mpm


class FlextUtilitiesProjectMetadata:
    """SSOT utilities for project-metadata derivation and pyproject reads.

    Inherited into ``FlextUtilities`` via MRO so each static method is
    callable as ``u.pascalize`` / ``u.derive_class_stem`` /
    ``u.read_project_metadata`` / etc. (flat access).
    """

    PYPROJECT_FILENAME: ClassVar[str] = "pyproject.toml"

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
        override = cpm.SPECIAL_NAME_OVERRIDES.get(normalized)
        if override is not None:
            return override
        return FlextUtilitiesProjectMetadata.pascalize(normalized)

    @staticmethod
    @cache
    def load_pyproject_toml(root: Path) -> Mapping[str, Any]:
        """Load and return the parsed pyproject.toml under ``root``.

        Result is cached by ``root`` (``Path`` is hashable). The cache is
        process-global so every workspace reader — typed config in
        ``read_tool_flext_config``/``read_project_metadata`` and the
        flext-infra ``_pyproject_payload`` validation layer — shares the
        same disk read; no parallel file I/O.
        """
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
    def read_project_metadata(root: Path) -> mpm.ProjectMetadata:
        """Read canonical project metadata from a project's pyproject.toml."""
        data = FlextUtilitiesProjectMetadata.load_pyproject_toml(root)
        project_raw = data.get("project")
        project: dict[str, Any] = (
            dict(project_raw) if isinstance(project_raw, Mapping) else {}
        )
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
        return mpm.ProjectMetadata(
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
    def read_tool_flext_config(root: Path) -> mpm.ProjectToolFlext:
        """Read ``[tool.flext.*]`` tables from pyproject.toml."""
        data = FlextUtilitiesProjectMetadata.load_pyproject_toml(root)
        tool = data.get("tool") or {}
        tool_flext = tool.get("flext") if isinstance(tool, Mapping) else None
        return mpm.ProjectToolFlext.model_validate(tool_flext or {})

    @staticmethod
    def compose_namespace_config(root: Path) -> mpm.ProjectNamespaceConfig:
        """Build the effective namespace config for a project."""
        meta = FlextUtilitiesProjectMetadata.read_project_metadata(root)
        cfg = FlextUtilitiesProjectMetadata.read_tool_flext_config(root)
        return mpm.ProjectNamespaceConfig(
            project_name=meta.name,
            enabled=cfg.namespace.enabled,
            scan_dirs=cfg.namespace.scan_dirs,
            include_dynamic_dirs=cfg.namespace.include_dynamic_dirs,
            alias_parent_sources=dict(cfg.namespace.alias_parent_sources),
        )
