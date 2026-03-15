"""Pyright phase tests for deps modernizer."""

from __future__ import annotations

from collections.abc import MutableMapping
from pathlib import Path

import tomlkit

from flext_infra import m, u
from flext_infra.deps._phases import EnsurePyrightConfigPhase
from flext_infra.deps.tool_config import FlextInfraDependencyToolConfig
from flext_tests import tm


def _test_tool_config() -> m.Infra.Deps.ToolConfigDocument:
    result = FlextInfraDependencyToolConfig.load_tool_config()
    tm.that(result.is_failure, eq=False)
    if result.is_failure:
        msg = "failed to load tool config"
        raise ValueError(msg)
    return result.value


class TestEnsurePyrightConfigPhase:
    """Tests pyright config phase behavior."""

    def test_apply_root_sets_execution_environments(self, tmp_path: Path) -> None:
        flext_core = tmp_path / "flext-core"
        flext_api = tmp_path / "flext-api"
        (flext_core / "pyproject.toml").parent.mkdir(parents=True, exist_ok=True)
        (flext_api / "pyproject.toml").parent.mkdir(parents=True, exist_ok=True)
        _ = (flext_core / "pyproject.toml").write_text("", encoding="utf-8")
        _ = (flext_api / "pyproject.toml").write_text("", encoding="utf-8")
        (flext_core / "src").mkdir(parents=True, exist_ok=True)
        (flext_core / "tests").mkdir(parents=True, exist_ok=True)
        (flext_api / "src").mkdir(parents=True, exist_ok=True)
        doc = tomlkit.document()
        changes = EnsurePyrightConfigPhase(_test_tool_config()).apply(
            doc,
            is_root=True,
            workspace_root=tmp_path,
        )
        tool = u.Infra.unwrap_item(doc["tool"])
        tm.that(isinstance(tool, MutableMapping), eq=True)
        if not isinstance(tool, MutableMapping):
            return
        pyright = u.Infra.unwrap_item(tool["pyright"])
        tm.that(isinstance(pyright, MutableMapping), eq=True)
        if not isinstance(pyright, MutableMapping):
            return
        envs = u.Infra.unwrap_item(pyright["executionEnvironments"])
        tm.that(isinstance(envs, list), eq=True)
        tm.that(
            envs,
            eq=[
                {"root": "flext-api/src", "reportPrivateUsage": "error"},
                {"root": "flext-core/src", "reportPrivateUsage": "error"},
                {"root": "flext-core/tests", "reportPrivateUsage": "none"},
            ],
        )
        tm.that(
            "tool.pyright.executionEnvironments set with tests reportPrivateUsage=none"
            in changes,
            eq=True,
        )

    def test_apply_subproject_sets_execution_environments(self) -> None:
        doc = tomlkit.document()
        changes = EnsurePyrightConfigPhase(_test_tool_config()).apply(
            doc, is_root=False
        )
        tool = u.Infra.unwrap_item(doc["tool"])
        tm.that(isinstance(tool, MutableMapping), eq=True)
        if not isinstance(tool, MutableMapping):
            return
        pyright = u.Infra.unwrap_item(tool["pyright"])
        tm.that(isinstance(pyright, MutableMapping), eq=True)
        if not isinstance(pyright, MutableMapping):
            return
        envs = u.Infra.unwrap_item(pyright["executionEnvironments"])
        tm.that(isinstance(envs, list), eq=True)
        tm.that(
            envs,
            eq=[
                {"root": "src", "reportPrivateUsage": "error"},
                {"root": "tests", "reportPrivateUsage": "none"},
            ],
        )
        tm.that(
            "tool.pyright.executionEnvironments set with tests reportPrivateUsage=none"
            in changes,
            eq=True,
        )
