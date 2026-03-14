"""Phase: Ensure standard pytest configuration without removing project-specific entries."""

from __future__ import annotations

import tomlkit
from tomlkit.container import Container
from tomlkit.items import Item, Table

from flext_infra import c, u
from flext_infra.deps._models import FlextInfraDepsModels


class EnsurePytestConfigPhase:
    """Ensure standard pytest configuration without removing project-specific entries."""

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
        pytest_tbl = u.Infra.ensure_table(tool, c.Infra.Toml.PYTEST)
        ini = u.Infra.ensure_table(pytest_tbl, c.Infra.Toml.INI_OPTIONS)
        if u.Infra.unwrap_item(u.Infra.get(ini, c.Infra.Toml.MINVERSION)) != "8.0":
            ini[c.Infra.Toml.MINVERSION] = "8.0"
            changes.append("tool.pytest.ini_options.minversion set to 8.0")
        current_classes = u.Infra.as_string_list(
            u.Infra.get(ini, c.Infra.Toml.PYTHON_CLASSES)
        )
        if "Test*" not in current_classes:
            ini[c.Infra.Toml.PYTHON_CLASSES] = u.Infra.array(
                sorted({*current_classes, "Test*"}),
            )
            changes.append("tool.pytest.ini_options.python_classes updated")
        standard_files = {"*_test.py", "*_tests.py", "test_*.py"}
        current_files = set(
            u.Infra.as_string_list(u.Infra.get(ini, c.Infra.Toml.PYTHON_FILES))
        )
        if not standard_files.issubset(current_files):
            ini[c.Infra.Toml.PYTHON_FILES] = u.Infra.array(
                sorted(current_files | standard_files),
            )
            changes.append("tool.pytest.ini_options.python_files updated")
        current_addopts = set(
            u.Infra.as_string_list(u.Infra.get(ini, c.Infra.Toml.ADDOPTS))
        )
        needed_addopts = set(self._tool_config.tools.pytest.standard_addopts)
        if not needed_addopts.issubset(current_addopts):
            ini[c.Infra.Toml.ADDOPTS] = u.Infra.array(
                sorted(current_addopts | needed_addopts)
            )
            changes.append("tool.pytest.ini_options.addopts updated")
        current_markers = u.Infra.as_string_list(u.Infra.get(ini, c.Infra.Toml.MARKERS))
        current_names = {m.split(":")[0].strip() for m in current_markers}
        added: list[str] = []
        for marker in self._tool_config.tools.pytest.standard_markers:
            name = marker.split(":")[0].strip()
            if name not in current_names:
                added.append(marker)
        if added:
            ini[c.Infra.Toml.MARKERS] = u.Infra.array(
                sorted([*current_markers, *added])
            )
            names = ", ".join(m.split(":")[0].strip() for m in added)
            changes.append(f"tool.pytest.ini_options.markers: added {names}")
        return changes
