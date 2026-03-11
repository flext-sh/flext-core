"""Phase: Ensure standard Pyright configuration for strict type checking."""

from __future__ import annotations

from pathlib import Path

import tomlkit
from tomlkit.items import Table

from flext_infra import c
from flext_infra._utilities.toml import FlextInfraUtilitiesToml as _Toml
from flext_infra._utilities.toml_parse import FlextInfraUtilitiesTomlParse as _TomlParse
from flext_infra.deps.tool_config import FlextInfraToolConfigDocument

ensure_table = _Toml.ensure_table
toml_get = _Toml.get
unwrap_item = _Toml.unwrap_item
ensure_pyright_execution_envs = _TomlParse.ensure_pyright_execution_envs


class EnsurePyrightConfigPhase:
    """Ensure standard Pyright configuration for strict type checking."""

    def __init__(self, tool_config: FlextInfraToolConfigDocument) -> None:
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

    def apply(
        self,
        doc: tomlkit.TOMLDocument,
        *,
        is_root: bool,
        workspace_root: Path | None = None,
    ) -> list[str]:
        changes: list[str] = []
        tool: object | None = None
        if c.Infra.Toml.TOOL in doc:
            tool = doc[c.Infra.Toml.TOOL]
        if not isinstance(tool, Table):
            tool = tomlkit.table()
            doc[c.Infra.Toml.TOOL] = tool
        pyright = ensure_table(tool, c.Infra.Toml.PYRIGHT)
        expected_envs = self._expected_envs(
            is_root=is_root,
            workspace_root=workspace_root,
        )
        if is_root:
            if (
                unwrap_item(toml_get(pyright, "typeCheckingMode"))
                != c.Infra.Modes.STRICT
            ):
                pyright["typeCheckingMode"] = c.Infra.Modes.STRICT
                changes.append("tool.pyright.typeCheckingMode set to strict")
            ensure_pyright_execution_envs(pyright, expected_envs, changes)
            return changes
        for key, value in self._tool_config.tools.pyright.strict_settings.items():
            if unwrap_item(toml_get(pyright, key)) != value:
                pyright[key] = value
                changes.append(f"tool.pyright.{key} set to {value}")
        ensure_pyright_execution_envs(pyright, expected_envs, changes)
        return changes
