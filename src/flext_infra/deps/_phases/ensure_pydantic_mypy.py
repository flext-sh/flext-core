"""Phase: Ensure standard pydantic-mypy configuration for strict model typing."""

from __future__ import annotations

import tomlkit
from tomlkit.items import Table

from flext_infra import c
from flext_infra._utilities.toml import FlextInfraUtilitiesToml as _Toml
from flext_infra.deps.tool_config import FlextInfraToolConfigDocument

ensure_table = _Toml.ensure_table
toml_get = _Toml.get
unwrap_item = _Toml.unwrap_item


class EnsurePydanticMypyConfigPhase:
    """Ensure standard pydantic-mypy configuration for strict model typing."""

    def __init__(self, tool_config: FlextInfraToolConfigDocument) -> None:
        self._tool_config = tool_config

    def apply(self, doc: tomlkit.TOMLDocument) -> list[str]:
        changes: list[str] = []
        tool: object | None = None
        if c.Infra.Toml.TOOL in doc:
            tool = doc[c.Infra.Toml.TOOL]
        if not isinstance(tool, Table):
            tool = tomlkit.table()
            doc[c.Infra.Toml.TOOL] = tool
        pydantic_mypy = ensure_table(tool, "pydantic-mypy")
        for key, value in {
            "init_forbid_extra": self._tool_config.tools.pydantic_mypy.init_forbid_extra,
            "init_typed": self._tool_config.tools.pydantic_mypy.init_typed,
            "warn_required_dynamic_aliases": self._tool_config.tools.pydantic_mypy.warn_required_dynamic_aliases,
        }.items():
            if unwrap_item(toml_get(pydantic_mypy, key)) is not value:
                pydantic_mypy[key] = value
                changes.append(f"tool.pydantic-mypy.{key} set to {value}")
        return changes
