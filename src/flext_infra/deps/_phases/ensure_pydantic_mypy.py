"""Phase: Ensure standard pydantic-mypy configuration for strict model typing."""

from __future__ import annotations

import tomlkit
from tomlkit.container import Container
from tomlkit.items import Item, Table

from flext_infra import c, u
from flext_infra.deps._models import FlextInfraDepsModels


class EnsurePydanticMypyConfigPhase:
    """Ensure standard pydantic-mypy configuration for strict model typing."""

    def __init__(self, tool_config: FlextInfraDepsModels.ToolConfigDocument) -> None:
        self._tool_config = tool_config

    def apply(self, doc: tomlkit.TOMLDocument) -> list[str]:
        changes: list[str] = []
        tool: Item | Container | None = None
        if c.Infra.Toml.TOOL in doc:
            tool = doc[c.Infra.Toml.TOOL]
        if not isinstance(tool, Table):
            tool = tomlkit.table()
            doc[c.Infra.Toml.TOOL] = tool
        pydantic_mypy = u.Infra.ensure_table(tool, "pydantic-mypy")
        for key, value in {
            "init_forbid_extra": self._tool_config.tools.pydantic_mypy.init_forbid_extra,
            "init_typed": self._tool_config.tools.pydantic_mypy.init_typed,
            "warn_required_dynamic_aliases": self._tool_config.tools.pydantic_mypy.warn_required_dynamic_aliases,
        }.items():
            if u.Infra.unwrap_item(u.Infra.get(pydantic_mypy, key)) is not value:
                pydantic_mypy[key] = value
                changes.append(f"tool.pydantic-mypy.{key} set to {value}")
        return changes
