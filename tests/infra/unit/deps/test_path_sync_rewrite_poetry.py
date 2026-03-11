from __future__ import annotations

from collections.abc import MutableMapping

import tomlkit
from tomlkit.toml_document import TOMLDocument

from flext_infra.deps.path_sync import FlextInfraDependencyPathSync
from flext_tests import tm

_PATH_SYNC = FlextInfraDependencyPathSync()
_rewrite_poetry = _PATH_SYNC._rewrite_poetry


class TestRewritePoetry:
    def test_rewrite_poetry_no_tool(self) -> None:
        tm.that(_rewrite_poetry(TOMLDocument(), is_root=True, mode="workspace"), eq=[])

    def test_rewrite_poetry_no_poetry(self) -> None:
        doc = TOMLDocument()
        doc["tool"] = tomlkit.table()
        tm.that(_rewrite_poetry(doc, is_root=True, mode="workspace"), eq=[])

    def test_rewrite_poetry_no_dependencies(self) -> None:
        doc = TOMLDocument()
        doc["tool"] = {"poetry": tomlkit.table()}
        tm.that(_rewrite_poetry(doc, is_root=True, mode="workspace"), eq=[])

    def test_rewrite_poetry_non_dict_dependencies(self) -> None:
        doc = TOMLDocument()
        doc["tool"] = {"poetry": {"dependencies": "not-a-dict"}}
        tm.that(_rewrite_poetry(doc, is_root=True, mode="workspace"), eq=[])

    def test_rewrite_poetry_rewrite_path_dep(self) -> None:
        doc = TOMLDocument()
        doc["tool"] = {
            "poetry": {
                "dependencies": {"flext-core": {"path": ".flext-deps/flext-core"}}
            },
        }
        changes = _rewrite_poetry(doc, is_root=True, mode="workspace")
        tm.that(len(changes) > 0, eq=True)
        tool = doc["tool"]
        tm.that(isinstance(tool, MutableMapping), eq=True)
        if not isinstance(tool, MutableMapping):
            return
        poetry = tool["poetry"]
        tm.that(isinstance(poetry, MutableMapping), eq=True)
        if not isinstance(poetry, MutableMapping):
            return
        deps = poetry["dependencies"]
        tm.that(isinstance(deps, MutableMapping), eq=True)
        if not isinstance(deps, MutableMapping):
            return
        core = deps["flext-core"]
        tm.that(isinstance(core, MutableMapping), eq=True)
        if not isinstance(core, MutableMapping):
            return
        tm.that(core["path"], eq="flext-core")

    def test_rewrite_poetry_skip_non_path_dep(self) -> None:
        doc = TOMLDocument()
        doc["tool"] = {"poetry": {"dependencies": {"requests": {"version": "^2.0.0"}}}}
        tm.that(_rewrite_poetry(doc, is_root=True, mode="workspace"), eq=[])

    def test_rewrite_poetry_non_dict_value(self) -> None:
        doc = TOMLDocument()
        doc["tool"] = {"poetry": {"dependencies": {"requests": "^2.0.0"}}}
        tm.that(_rewrite_poetry(doc, is_root=True, mode="workspace"), eq=[])

    def test_rewrite_poetry_empty_path(self) -> None:
        doc = TOMLDocument()
        doc["tool"] = {"poetry": {"dependencies": {"flext-core": {"path": ""}}}}
        tm.that(_rewrite_poetry(doc, is_root=True, mode="workspace"), eq=[])

    def test_rewrite_poetry_non_string_path(self) -> None:
        doc = TOMLDocument()
        doc["tool"] = {"poetry": {"dependencies": {"flext-core": {"path": 123}}}}
        tm.that(_rewrite_poetry(doc, is_root=True, mode="workspace"), eq=[])

    def test_rewrite_poetry_subproject_mode(self) -> None:
        doc = TOMLDocument()
        doc["tool"] = {
            "poetry": {
                "dependencies": {"flext-core": {"path": ".flext-deps/flext-core"}}
            },
        }
        changes = _rewrite_poetry(doc, is_root=False, mode="workspace")
        tm.that(len(changes) > 0, eq=True)
        tool = doc["tool"]
        tm.that(isinstance(tool, MutableMapping), eq=True)
        if not isinstance(tool, MutableMapping):
            return
        poetry = tool["poetry"]
        tm.that(isinstance(poetry, MutableMapping), eq=True)
        if not isinstance(poetry, MutableMapping):
            return
        deps = poetry["dependencies"]
        tm.that(isinstance(deps, MutableMapping), eq=True)
        if not isinstance(deps, MutableMapping):
            return
        core = deps["flext-core"]
        tm.that(isinstance(core, MutableMapping), eq=True)
        if not isinstance(core, MutableMapping):
            return
        tm.that(core["path"], eq="../flext-core")


def test_rewrite_poetry_with_non_dict_value() -> None:
    doc = tomlkit.document()
    doc["tool"] = tomlkit.table()
    tool = doc["tool"]
    tm.that(isinstance(tool, MutableMapping), eq=True)
    if not isinstance(tool, MutableMapping):
        return
    poetry = tomlkit.table()
    tool["poetry"] = poetry
    deps = tomlkit.table()
    deps["flext-core"] = "string-value"
    poetry["dependencies"] = deps
    tm.that(len(_rewrite_poetry(doc, is_root=False, mode="workspace")), eq=0)


def test_rewrite_poetry_no_tool_table() -> None:
    tm.that(
        len(_rewrite_poetry(tomlkit.document(), is_root=False, mode="workspace")), eq=0
    )


def test_rewrite_poetry_no_poetry_table() -> None:
    doc = tomlkit.document()
    doc["tool"] = tomlkit.table()
    tm.that(len(_rewrite_poetry(doc, is_root=False, mode="workspace")), eq=0)
