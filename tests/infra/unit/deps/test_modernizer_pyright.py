"""Pyright phase tests for deps modernizer."""

from __future__ import annotations

from collections.abc import MutableMapping

import tomlkit

from flext_infra.deps.modernizer import EnsurePyrightConfigPhase, unwrap_item
from flext_infra.deps.tool_config import FlextInfraToolConfigDocument, load_tool_config
from flext_tests import tm


def _test_tool_config() -> FlextInfraToolConfigDocument:
    result = load_tool_config()
    tm.that(result.is_failure, eq=False)
    if result.is_failure:
        msg = "failed to load tool config"
        raise ValueError(msg)
    return result.value


class TestEnsurePyrightConfigPhase:
    """Tests pyright config phase behavior."""

    def test_apply_root_sets_execution_environments(self) -> None:
        doc = tomlkit.document()
        changes = EnsurePyrightConfigPhase(_test_tool_config()).apply(doc, is_root=True)
        tool = unwrap_item(doc["tool"])
        tm.that(isinstance(tool, MutableMapping), eq=True)
        if not isinstance(tool, MutableMapping):
            return
        pyright = unwrap_item(tool["pyright"])
        tm.that(isinstance(pyright, MutableMapping), eq=True)
        if not isinstance(pyright, MutableMapping):
            return
        envs = unwrap_item(pyright["executionEnvironments"])
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

    def test_apply_subproject_sets_execution_environments(self) -> None:
        doc = tomlkit.document()
        changes = EnsurePyrightConfigPhase(_test_tool_config()).apply(
            doc, is_root=False
        )
        tool = unwrap_item(doc["tool"])
        tm.that(isinstance(tool, MutableMapping), eq=True)
        if not isinstance(tool, MutableMapping):
            return
        pyright = unwrap_item(tool["pyright"])
        tm.that(isinstance(pyright, MutableMapping), eq=True)
        if not isinstance(pyright, MutableMapping):
            return
        envs = unwrap_item(pyright["executionEnvironments"])
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
