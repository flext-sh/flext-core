"""Phase: Ensure standard Pyright configuration for strict type checking."""

from __future__ import annotations

from pathlib import Path

import tomlkit
from tomlkit.items import Item, Table

from flext_infra import c, u
from flext_infra.deps._models import FlextInfraDepsModels


class EnsurePyrightConfigPhase:
    """Ensure standard Pyright configuration for strict type checking."""

    def __init__(self, tool_config: FlextInfraDepsModels.ToolConfigDocument) -> None:
        self._tool_config = tool_config

    def _expected_envs(
        self,
        *,
        is_root: bool,
        workspace_root: Path | None,
    ) -> list[dict[str, str]]:
        default_envs: list[dict[str, str]] = [
            {"root": c.Infra.Paths.DEFAULT_SRC_DIR, "reportPrivateUsage": "error"},
            {"root": c.Infra.Directories.TESTS, "reportPrivateUsage": "none"},
        ]
        if not is_root or workspace_root is None:
            return default_envs

        expected_envs: list[dict[str, str]] = []
        root_src = workspace_root / c.Infra.Paths.DEFAULT_SRC_DIR
        root_tests = workspace_root / c.Infra.Directories.TESTS
        if root_src.exists():
            expected_envs.append({
                "root": c.Infra.Paths.DEFAULT_SRC_DIR,
                "reportPrivateUsage": "error",
            })
        if root_tests.exists():
            expected_envs.append({
                "root": c.Infra.Directories.TESTS,
                "reportPrivateUsage": "none",
            })

        child_projects = sorted(
            child
            for child in workspace_root.iterdir()
            if child.is_dir() and (child / c.Infra.Files.PYPROJECT_FILENAME).exists()
        )
        for child_project in child_projects:
            relative_root = child_project.relative_to(workspace_root)
            child_src = child_project / c.Infra.Paths.DEFAULT_SRC_DIR
            child_tests = child_project / c.Infra.Directories.TESTS
            if child_src.exists():
                expected_envs.append({
                    "root": (relative_root / c.Infra.Paths.DEFAULT_SRC_DIR).as_posix(),
                    "reportPrivateUsage": "error",
                })
            if child_tests.exists():
                expected_envs.append({
                    "root": (relative_root / c.Infra.Directories.TESTS).as_posix(),
                    "reportPrivateUsage": "none",
                })
        return expected_envs or default_envs

    def _override_for_kind(
        self,
        project_kind: str,
    ) -> FlextInfraDepsModels.ProjectTypeOverrideConfig | None:
        """Return the project-type override config for the given kind, if any."""
        overrides = self._tool_config.project_type_overrides
        kind_map: dict[str, FlextInfraDepsModels.ProjectTypeOverrideConfig] = {
            "core": overrides.core,
            "domain": overrides.domain,
            "platform": overrides.platform,
            "integration": overrides.integration,
            "app": overrides.app,
        }
        return kind_map.get(project_kind)

    def apply(
        self,
        doc: tomlkit.TOMLDocument,
        *,
        is_root: bool,
        workspace_root: Path | None = None,
        project_kind: str = "core",
    ) -> list[str]:
        changes: list[str] = []
        tool: Item | None = None
        if c.Infra.Toml.TOOL in doc:
            raw_tool = doc[c.Infra.Toml.TOOL]
            if isinstance(raw_tool, Item):
                tool = raw_tool
        if not isinstance(tool, Table):
            tool = tomlkit.table()
            doc[c.Infra.Toml.TOOL] = tool
        pyright = u.Infra.ensure_table(tool, c.Infra.Toml.PYRIGHT)
        expected_envs = self._expected_envs(
            is_root=is_root,
            workspace_root=workspace_root,
        )
        if is_root:
            if (
                u.Infra.unwrap_item(u.Infra.get(pyright, "typeCheckingMode"))
                != c.Infra.Modes.STRICT
            ):
                pyright["typeCheckingMode"] = c.Infra.Modes.STRICT
                changes.append("tool.pyright.typeCheckingMode set to strict")
            u.Infra.ensure_pyright_execution_envs(pyright, expected_envs, changes)
            return changes
        for key, value in self._tool_config.tools.pyright.strict_settings.items():
            if u.Infra.unwrap_item(u.Infra.get(pyright, key)) != value:
                pyright[key] = value
                changes.append(f"tool.pyright.{key} set to {value}")
        # Merge extended_settings with project_kind overrides BEFORE applying
        # to avoid double-change noise (set to X then immediately override to Y)
        merged_settings: dict[str, str] = {
            **self._tool_config.tools.pyright.extended_settings,
        }
        override = self._override_for_kind(project_kind)
        if override is not None:
            merged_settings.update(override.pyright)
        for key, value in merged_settings.items():
            if u.Infra.unwrap_item(u.Infra.get(pyright, key)) != value:
                pyright[key] = value
                changes.append(f"tool.pyright.{key} set to {value}")
        u.Infra.ensure_pyright_execution_envs(pyright, expected_envs, changes)
        return changes
