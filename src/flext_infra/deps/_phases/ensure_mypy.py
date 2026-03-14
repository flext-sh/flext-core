"""Phase: Ensure standard mypy configuration with pydantic plugin across all projects."""

from __future__ import annotations

import tomlkit
from tomlkit.container import Container
from tomlkit.items import Item, Table

from flext_infra import c, u
from flext_infra.deps._models import FlextInfraDepsModels


class EnsureMypyConfigPhase:
    """Ensure standard mypy configuration with pydantic plugin across all projects."""

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
        mypy = u.Infra.ensure_table(tool, c.Infra.Toml.MYPY)
        if (
            u.Infra.unwrap_item(
                u.Infra.get(mypy, c.Infra.Toml.PYTHON_VERSION_UNDERSCORE)
            )
            != "3.13"
        ):
            mypy[c.Infra.Toml.PYTHON_VERSION_UNDERSCORE] = "3.13"
            changes.append("tool.mypy.python_version set to 3.13")
        current_plugins = u.Infra.as_string_list(
            u.Infra.get(mypy, c.Infra.Toml.PLUGINS)
        )
        needed_plugins = [
            plugin
            for plugin in self._tool_config.tools.mypy.plugins
            if plugin not in current_plugins
        ]
        if needed_plugins:
            mypy[c.Infra.Toml.PLUGINS] = u.Infra.array(
                sorted(
                    set(current_plugins) | set(self._tool_config.tools.mypy.plugins)
                ),
            )
            changes.append(f"tool.mypy.plugins added {', '.join(needed_plugins)}")
        current_disabled = u.Infra.as_string_list(
            u.Infra.get(mypy, c.Infra.Toml.DISABLE_ERROR_CODE),
        )
        needed_disabled = [
            ec
            for ec in self._tool_config.tools.mypy.disabled_error_codes
            if ec not in current_disabled
        ]
        if needed_disabled:
            mypy[c.Infra.Toml.DISABLE_ERROR_CODE] = u.Infra.array(
                sorted(
                    set(current_disabled)
                    | set(self._tool_config.tools.mypy.disabled_error_codes),
                ),
            )
            changes.append(
                f"tool.mypy.disable_error_code added {', '.join(needed_disabled)}",
            )
        for key, value in self._tool_config.tools.mypy.boolean_settings.items():
            if u.Infra.unwrap_item(u.Infra.get(mypy, key)) is not value:
                mypy[key] = value
                changes.append(f"tool.mypy.{key} set to {value}")
        return changes
