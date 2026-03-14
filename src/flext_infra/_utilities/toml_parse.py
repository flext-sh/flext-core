"""TOML dependency and project parsing helpers for flext-infra.

Advanced parsing operations for pyproject.toml: dependency extraction,
deduplication, namespace discovery, and tool configuration management.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import tomlkit
from pydantic import TypeAdapter, ValidationError
from tomlkit.container import Container
from tomlkit.items import Item, Table

from flext_core import r
from flext_infra import c, t
from flext_infra._utilities.toml import FlextInfraUtilitiesToml


class FlextInfraUtilitiesTomlParse:
    """TOML parsing helpers — dependency extraction and project configuration.

    Usage::

        from flext_infra._utilities.toml_parse import FlextInfraUtilitiesTomlParse

        name = FlextInfraUtilitiesTomlParse.dep_name("requests>=2.0")
    """

    @staticmethod
    def dep_name(spec: str) -> str:
        """Extract normalized dependency name from requirement specification."""
        base = spec.strip().split("@", 1)[0].strip()
        match = c.Infra.Deps.DEP_NAME_RE.match(base)
        if match:
            return match.group(1).lower().replace("_", "-")
        return base.lower().replace("_", "-")

    @staticmethod
    def dedupe_specs(specs: list[str]) -> list[str]:
        """Deduplicate dependency specifications by normalized name, sorted by full spec string."""
        seen: dict[str, str] = {}
        for spec in specs:
            key = FlextInfraUtilitiesTomlParse.dep_name(spec)
            if key and key not in seen:
                seen[key] = spec
        return sorted(seen.values())

    @staticmethod
    def ensure_pyright_execution_envs(
        pyright: Table,
        expected: list[dict[str, str]],
        changes: list[str],
    ) -> None:
        """Ensure pyright executionEnvironments matches expected; append to changes if updated."""
        raw = FlextInfraUtilitiesToml.unwrap_item(
            FlextInfraUtilitiesToml.get(pyright, "executionEnvironments"),
        )
        current: list[dict[str, str]] = []
        if isinstance(raw, list):
            try:
                current = TypeAdapter(list[dict[str, str]]).validate_python(raw)
            except ValidationError:
                current = []
        if list(current) != expected:
            pyright["executionEnvironments"] = expected
            changes.append(
                "tool.pyright.executionEnvironments set with tests reportPrivateUsage=none",
            )

    @staticmethod
    def discover_first_party_namespaces(project_dir: Path) -> list[str]:
        """Discover first-party namespace packages from src/ for tool configuration."""
        src_dir = project_dir / c.Infra.Paths.DEFAULT_SRC_DIR
        if not src_dir.is_dir():
            return []
        namespaces: list[str] = []
        for entry in sorted(src_dir.iterdir()):
            if not entry.is_dir() or entry.name == "__pycache__":
                continue
            if not entry.name.isidentifier() or "-" in entry.name:
                continue
            namespaces.append(entry.name)
        return namespaces

    @staticmethod
    def project_dev_groups(doc: tomlkit.TOMLDocument) -> dict[str, list[str]]:
        """Extract optional-dependencies groups from project table."""
        project_raw: t.Infra.InfraValue | Item | Container | None = None
        if c.Infra.Toml.PROJECT in doc:
            project_raw = doc[c.Infra.Toml.PROJECT]
        if not isinstance(project_raw, (Table, dict)):
            return {}
        optional_raw: t.Infra.InfraValue | Item | None = None
        if c.Infra.Toml.OPTIONAL_DEPENDENCIES in project_raw:
            optional_raw = project_raw[c.Infra.Toml.OPTIONAL_DEPENDENCIES]
        if not isinstance(optional_raw, (Table, dict)):
            return {}
        opt_deps: Table | dict[str, t.Infra.InfraValue] = optional_raw

        def _group_values(group_key: str) -> list[str]:
            value: t.Infra.InfraValue | Item | None = None
            if group_key in opt_deps:
                value = opt_deps[group_key]
            return FlextInfraUtilitiesToml.as_string_list(value)

        return {
            c.Infra.Toml.DEV: _group_values(c.Infra.Toml.DEV),
            c.Infra.Directories.DOCS: _group_values(c.Infra.Toml.DOCS),
            c.Infra.Gates.SECURITY: _group_values(c.Infra.Toml.SECURITY),
            c.Infra.Toml.TEST: _group_values(c.Infra.Toml.TEST),
            c.Infra.Directories.TYPINGS: _group_values(c.Infra.Directories.TYPINGS),
        }

    @staticmethod
    def canonical_dev_dependencies(root_doc: tomlkit.TOMLDocument) -> list[str]:
        """Merge all dev dependency groups from root pyproject."""
        groups = FlextInfraUtilitiesTomlParse.project_dev_groups(root_doc)
        merged = [
            *groups.get(c.Infra.Toml.DEV, []),
            *groups.get(c.Infra.Directories.DOCS, []),
            *groups.get(c.Infra.Gates.SECURITY, []),
            *groups.get(c.Infra.Toml.TEST, []),
            *groups.get(c.Infra.Directories.TYPINGS, []),
        ]
        return FlextInfraUtilitiesTomlParse.dedupe_specs(merged)

    @staticmethod
    def read_plain(path: Path) -> r[t.Infra.TomlConfig]:
        """Read and parse a TOML file as a plain dict with r error handling."""
        if not path.exists():
            return r[t.Infra.TomlConfig].ok({})
        try:
            data_raw = tomllib.loads(
                path.read_text(encoding=c.Infra.Encoding.DEFAULT),
            )
            data: t.Infra.TomlConfig = data_raw
            return r[t.Infra.TomlConfig].ok(data)
        except (tomllib.TOMLDecodeError, OSError) as exc:
            return r[t.Infra.TomlConfig].fail(f"TOML read error: {exc}")


__all__ = ["FlextInfraUtilitiesTomlParse"]
