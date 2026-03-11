"""Phase: Ensure safe default config for TOML/YAML formatting tools."""

from __future__ import annotations

import tomlkit
from tomlkit.items import Table

from flext_infra import c
from flext_infra._utilities.toml import FlextInfraUtilitiesToml as _Toml
from flext_infra.deps.tool_config import FlextInfraToolConfigDocument

array = _Toml.array
ensure_table = _Toml.ensure_table
toml_get = _Toml.get
unwrap_item = _Toml.unwrap_item


class EnsureFormattingToolingPhase:
    """Ensure safe default config for TOML/YAML formatting tools."""

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
        tomlsort = ensure_table(tool, "tomlsort")
        for key, value in {
            "all": self._tool_config.tools.tomlsort.all,
            "in_place": self._tool_config.tools.tomlsort.in_place,
            "sort_first": self._tool_config.tools.tomlsort.sort_first,
        }.items():
            current = unwrap_item(toml_get(tomlsort, key))
            if isinstance(value, list) and isinstance(current, list):
                if sorted(str(i) for i in current) != sorted(str(i) for i in value):
                    tomlsort[key] = array(sorted(str(item) for item in value))
                    changes.append(f"tool.tomlsort.{key} set")
            elif current != value:
                if isinstance(value, list):
                    tomlsort[key] = array(sorted(str(item) for item in value))
                else:
                    tomlsort[key] = value
                changes.append(f"tool.tomlsort.{key} set")
        yamlfix = ensure_table(tool, "yamlfix")
        for key, value in {
            "line_length": self._tool_config.tools.yamlfix.line_length,
            "preserve_quotes": self._tool_config.tools.yamlfix.preserve_quotes,
            "whitelines": self._tool_config.tools.yamlfix.whitelines,
            "section_whitelines": self._tool_config.tools.yamlfix.section_whitelines,
            "explicit_start": self._tool_config.tools.yamlfix.explicit_start,
        }.items():
            if unwrap_item(toml_get(yamlfix, key)) != value:
                yamlfix[key] = value
                changes.append(f"tool.yamlfix.{key} set to {value}")
        return changes
