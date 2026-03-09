from __future__ import annotations

from collections.abc import MutableMapping

import tomlkit

from flext_infra.deps.modernizer import EnsurePyreflyConfigPhase
from flext_infra.deps.tool_config import FlextInfraToolConfigDocument, load_tool_config
from flext_tests import tm
from tests.infra import h


def _test_tool_config() -> FlextInfraToolConfigDocument:
    return tm.ok(load_tool_config())


class TestEnsurePyreflyConfigPhase:
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
    pyrefly = doc["tool"]["pyrefly"]
    tm.that(isinstance(pyrefly, MutableMapping), eq=True)
    if isinstance(pyrefly, MutableMapping):
        tm.that(pyrefly["python-version"], eq="3.13")


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
    pyrefly = doc["tool"]["pyrefly"]
    tm.that(isinstance(pyrefly, MutableMapping), eq=True)
    if isinstance(pyrefly, MutableMapping):
        tm.that(pyrefly["ignore-errors-in-generated-code"], eq=True)


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
    tm.that(hasattr(h, "assert_ok"), eq=True)
