"""Pyrefly phase tests for deps modernizer."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import cast

import tomlkit

from flext_infra import m
from flext_infra.deps._phases import EnsurePyreflyConfigPhase
from flext_infra.deps.tool_config import FlextInfraDependencyToolConfig
from flext_tests import t, tm


def _test_tool_config() -> m.Infra.Deps.ToolConfigDocument:
    result = FlextInfraDependencyToolConfig.load_tool_config()
    tm.that(result.is_failure, eq=False)
    if result.is_failure:
        msg = "failed to load tool config"
        raise ValueError(msg)
    return result.value


class TestEnsurePyreflyConfigPhase:
    """Tests pyrefly config phase behavior."""

    def test_ensure_pyrefly_config_sets_fields_root(self) -> None:
        doc = tomlkit.document()
        doc["tool"] = tomlkit.table()
        phase = EnsurePyreflyConfigPhase(_test_tool_config())
        changes = phase.apply(doc, is_root=True)
        tm.that(any("python-version" in c for c in changes), eq=True)
        tm.that(any("ignore-errors-in-generated-code" in c for c in changes), eq=True)
        tm.that(any("search-path" in c for c in changes), eq=True)
        tm.that(any("errors" in c for c in changes), eq=True)
        tm.that(any("project-excludes" in c for c in changes), eq=True)

    def test_ensure_pyrefly_config_non_root(self) -> None:
        doc = tomlkit.document()
        doc["tool"] = tomlkit.table()
        changes = EnsurePyreflyConfigPhase(_test_tool_config()).apply(
            doc, is_root=False
        )
        tm.that(len(changes) > 0, eq=True)


def test_ensure_pyrefly_config_phase_apply_python_version() -> None:
    doc = tomlkit.document()
    doc["tool"] = tomlkit.table()
    tool = doc["tool"]
    tm.that(isinstance(tool, MutableMapping), eq=True)
    if not isinstance(tool, MutableMapping):
        return
    tool["pyrefly"] = tomlkit.table()
    changes = EnsurePyreflyConfigPhase(_test_tool_config()).apply(doc, is_root=True)
    tm.that(any("python-version set to 3.13" in c for c in changes), eq=True)
    pyrefly = tool["pyrefly"]
    tm.that(isinstance(pyrefly, MutableMapping), eq=True)
    if isinstance(pyrefly, MutableMapping):
        tm.that(
            cast("t.Tests.Matcher.MatcherKwargValue", pyrefly["python-version"]),
            eq="3.13",
        )


def test_ensure_pyrefly_config_phase_apply_ignore_errors() -> None:
    doc = tomlkit.document()
    doc["tool"] = tomlkit.table()
    tool = doc["tool"]
    tm.that(isinstance(tool, MutableMapping), eq=True)
    if not isinstance(tool, MutableMapping):
        return
    tool["pyrefly"] = tomlkit.table()
    changes = EnsurePyreflyConfigPhase(_test_tool_config()).apply(doc, is_root=True)
    tm.that(
        any("ignore-errors-in-generated-code enabled" in c for c in changes), eq=True
    )
    pyrefly = tool["pyrefly"]
    tm.that(isinstance(pyrefly, MutableMapping), eq=True)
    if isinstance(pyrefly, MutableMapping):
        tm.that(
            cast(
                "t.Tests.Matcher.MatcherKwargValue",
                pyrefly["ignore-errors-in-generated-code"],
            ),
            eq=True,
        )


def test_ensure_pyrefly_config_phase_apply_search_path() -> None:
    doc = tomlkit.document()
    doc["tool"] = tomlkit.table()
    tool = doc["tool"]
    tm.that(isinstance(tool, MutableMapping), eq=True)
    if not isinstance(tool, MutableMapping):
        return
    tool["pyrefly"] = tomlkit.table()
    changes = EnsurePyreflyConfigPhase(_test_tool_config()).apply(doc, is_root=True)
    tm.that("search-path set to" in " ".join(changes), eq=True)


def test_ensure_pyrefly_config_phase_apply_errors() -> None:
    doc = tomlkit.document()
    doc["tool"] = tomlkit.table()
    tool = doc["tool"]
    tm.that(isinstance(tool, MutableMapping), eq=True)
    if not isinstance(tool, MutableMapping):
        return
    tool["pyrefly"] = tomlkit.table()
    changes = EnsurePyreflyConfigPhase(_test_tool_config()).apply(doc, is_root=True)
    tm.that(any("errors" in c for c in changes), eq=True)
