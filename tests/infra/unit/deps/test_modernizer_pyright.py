from __future__ import annotations

from collections.abc import MutableMapping

import tomlkit

from flext_infra.deps.modernizer import EnsurePyrightConfigPhase, _unwrap_item
from flext_infra.deps.tool_config import FlextInfraToolConfigDocument, load_tool_config
from flext_tests import tm
from tests.infra import h


def _test_tool_config() -> FlextInfraToolConfigDocument:
    result = load_tool_config()
    return tm.ok(result)


class TestEnsurePyrightConfigPhase:
    def test_apply_root_sets_execution_environments(self) -> None:
        doc = tomlkit.document()
        changes = EnsurePyrightConfigPhase(_test_tool_config()).apply(doc, is_root=True)
        tool = _unwrap_item(doc["tool"])
        tm.that(isinstance(tool, MutableMapping), eq=True)
        if not isinstance(tool, MutableMapping):
            return
        pyright = _unwrap_item(tool["pyright"])
        tm.that(isinstance(pyright, MutableMapping), eq=True)
        if not isinstance(pyright, MutableMapping):
            return
        envs = _unwrap_item(pyright["executionEnvironments"])
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
        tool = _unwrap_item(doc["tool"])
        tm.that(isinstance(tool, MutableMapping), eq=True)
        if not isinstance(tool, MutableMapping):
            return
        pyright = _unwrap_item(tool["pyright"])
        tm.that(isinstance(pyright, MutableMapping), eq=True)
        if not isinstance(pyright, MutableMapping):
            return
        envs = _unwrap_item(pyright["executionEnvironments"])
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
        tm.that(hasattr(h, "assert_ok"), eq=True)
