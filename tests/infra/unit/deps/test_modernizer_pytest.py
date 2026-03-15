"""Pytest phase tests for deps modernizer."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import cast

import tomlkit

from flext_infra import m
from flext_infra.deps._phases import EnsurePytestConfigPhase
from flext_infra.deps.tool_config import FlextInfraDependencyToolConfig
from flext_tests import t, tm


def _test_tool_config() -> m.Infra.Deps.ToolConfigDocument:
    result = FlextInfraDependencyToolConfig.load_tool_config()
    tm.that(result.is_failure, eq=False)
    if result.is_failure:
        msg = "failed to load tool config"
        raise ValueError(msg)
    return result.value


class TestEnsurePytestConfigPhase:
    """Tests pytest config phase behavior."""

    def test_ensure_pytest_config_sets_fields(self) -> None:
        doc = tomlkit.document()
        doc["tool"] = tomlkit.table()
        changes = EnsurePytestConfigPhase(_test_tool_config()).apply(doc)
        tm.that(any("minversion" in c for c in changes), eq=True)
        tm.that(any("python_classes" in c for c in changes), eq=True)
        tm.that(any("python_files" in c for c in changes), eq=True)
        tm.that(any("addopts" in c for c in changes), eq=True)
        tm.that(any("markers" in c for c in changes), eq=True)

    def test_ensure_pytest_config_preserves_existing(self) -> None:
        doc = tomlkit.document()
        doc["tool"] = {
            "pytest": {
                "ini_options": {"minversion": "8.0", "python_classes": ["Test*"]}
            },
        }
        _ = EnsurePytestConfigPhase(_test_tool_config()).apply(doc)
        tool = doc["tool"]
        tm.that(isinstance(tool, MutableMapping), eq=True)
        if not isinstance(tool, MutableMapping):
            return
        pytest_section = tool["pytest"]
        tm.that(isinstance(pytest_section, MutableMapping), eq=True)
        if not isinstance(pytest_section, MutableMapping):
            return
        ini_options = pytest_section["ini_options"]
        tm.that(isinstance(ini_options, MutableMapping), eq=True)
        if isinstance(ini_options, MutableMapping):
            tm.that(
                cast("t.Tests.Matcher.MatcherKwargValue", ini_options["minversion"]),
                eq="8.0",
            )


def test_ensure_pytest_config_phase_apply_minversion() -> None:
    doc = tomlkit.document()
    doc["tool"] = tomlkit.table()
    tool = doc["tool"]
    tm.that(isinstance(tool, MutableMapping), eq=True)
    if not isinstance(tool, MutableMapping):
        return
    tool["pytest"] = tomlkit.table()
    pytest_section = tool["pytest"]
    tm.that(isinstance(pytest_section, MutableMapping), eq=True)
    if not isinstance(pytest_section, MutableMapping):
        return
    pytest_section["ini_options"] = tomlkit.table()
    changes = EnsurePytestConfigPhase(_test_tool_config()).apply(doc)
    tm.that(any("minversion set to 8.0" in c for c in changes), eq=True)
    ini_options = pytest_section["ini_options"]
    tm.that(isinstance(ini_options, MutableMapping), eq=True)
    if isinstance(ini_options, MutableMapping):
        tm.that(
            cast("t.Tests.Matcher.MatcherKwargValue", ini_options["minversion"]),
            eq="8.0",
        )


def test_ensure_pytest_config_phase_apply_python_classes() -> None:
    doc = tomlkit.document()
    doc["tool"] = tomlkit.table()
    tool = doc["tool"]
    tm.that(isinstance(tool, MutableMapping), eq=True)
    if not isinstance(tool, MutableMapping):
        return
    tool["pytest"] = tomlkit.table()
    pytest_section = tool["pytest"]
    tm.that(isinstance(pytest_section, MutableMapping), eq=True)
    if not isinstance(pytest_section, MutableMapping):
        return
    pytest_section["ini_options"] = tomlkit.table()
    changes = EnsurePytestConfigPhase(_test_tool_config()).apply(doc)
    tm.that(any("python_classes updated" in c for c in changes), eq=True)


def test_ensure_pytest_config_phase_apply_markers() -> None:
    doc = tomlkit.document()
    doc["tool"] = tomlkit.table()
    tool = doc["tool"]
    tm.that(isinstance(tool, MutableMapping), eq=True)
    if not isinstance(tool, MutableMapping):
        return
    tool["pytest"] = tomlkit.table()
    pytest_section = tool["pytest"]
    tm.that(isinstance(pytest_section, MutableMapping), eq=True)
    if not isinstance(pytest_section, MutableMapping):
        return
    pytest_section["ini_options"] = tomlkit.table()
    changes = EnsurePytestConfigPhase(_test_tool_config()).apply(doc)
    tm.that(any("markers" in c for c in changes), eq=True)
