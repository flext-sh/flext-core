"""Phase: Ensure namespace discovery is reflected across project tooling tables."""

from __future__ import annotations

from pathlib import Path

import tomlkit
from tomlkit.container import Container
from tomlkit.items import Item, Table

from flext_infra import c, u


class EnsureNamespaceToolingPhase:
    """Ensure namespace discovery is reflected across project tooling tables."""

    def apply(self, doc: tomlkit.TOMLDocument, *, path: Path) -> list[str]:
        changes: list[str] = []
        detected = sorted(u.Infra.discover_first_party_namespaces(path.parent))
        if not detected:
            return changes
        tool: Item | Container | None = None
        if c.Infra.Toml.TOOL in doc:
            tool = doc[c.Infra.Toml.TOOL]
        if not isinstance(tool, Table):
            tool = tomlkit.table()
            doc[c.Infra.Toml.TOOL] = tool
        deptry = u.Infra.ensure_table(tool, c.Infra.Toml.DEPTRY)
        current_deptry = sorted(
            u.Infra.as_string_list(
                u.Infra.get(deptry, c.Infra.Toml.KNOWN_FIRST_PARTY_UNDERSCORE)
            ),
        )
        if current_deptry != detected:
            deptry[c.Infra.Toml.KNOWN_FIRST_PARTY_UNDERSCORE] = u.Infra.array(detected)
            changes.append(f"tool.deptry.known_first_party set to {detected}")
        pyright = u.Infra.ensure_table(tool, c.Infra.Toml.PYRIGHT)
        extra_paths = u.Infra.as_string_list(u.Infra.get(pyright, "extraPaths"))
        if c.Infra.Paths.DEFAULT_SRC_DIR not in extra_paths:
            pyright["extraPaths"] = u.Infra.array(
                sorted({*extra_paths, c.Infra.Paths.DEFAULT_SRC_DIR}),
            )
            changes.append("tool.pyright.extraPaths includes src")
        return changes
