"""Phase: Ensure namespace discovery is reflected across project tooling tables."""

from __future__ import annotations

from pathlib import Path

import tomlkit
from tomlkit.items import Table

from flext_infra import c
from flext_infra._utilities.toml import FlextInfraUtilitiesToml as _Toml
from flext_infra._utilities.toml_parse import FlextInfraUtilitiesTomlParse as _TomlParse

array = _Toml.array
as_string_list = _Toml.as_string_list
ensure_table = _Toml.ensure_table
toml_get = _Toml.get
discover_first_party_namespaces = _TomlParse.discover_first_party_namespaces


class EnsureNamespaceToolingPhase:
    """Ensure namespace discovery is reflected across project tooling tables."""

    def apply(self, doc: tomlkit.TOMLDocument, *, path: Path) -> list[str]:
        changes: list[str] = []
        detected = sorted(discover_first_party_namespaces(path.parent))
        if not detected:
            return changes
        tool: object | None = None
        if c.Infra.Toml.TOOL in doc:
            tool = doc[c.Infra.Toml.TOOL]
        if not isinstance(tool, Table):
            tool = tomlkit.table()
            doc[c.Infra.Toml.TOOL] = tool
        deptry = ensure_table(tool, c.Infra.Toml.DEPTRY)
        current_deptry = sorted(
            as_string_list(toml_get(deptry, c.Infra.Toml.KNOWN_FIRST_PARTY_UNDERSCORE)),
        )
        if current_deptry != detected:
            deptry[c.Infra.Toml.KNOWN_FIRST_PARTY_UNDERSCORE] = array(detected)
            changes.append(f"tool.deptry.known_first_party set to {detected}")
        pyright = ensure_table(tool, c.Infra.Toml.PYRIGHT)
        extra_paths = as_string_list(toml_get(pyright, "extraPaths"))
        if c.Infra.Paths.DEFAULT_SRC_DIR not in extra_paths:
            pyright["extraPaths"] = array(
                sorted({*extra_paths, c.Infra.Paths.DEFAULT_SRC_DIR}),
            )
            changes.append("tool.pyright.extraPaths includes src")
        return changes
