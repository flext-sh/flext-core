from __future__ import annotations

import tomlkit
from tomlkit.toml_document import TOMLDocument

from flext_infra.deps.path_sync import FlextInfraDependencyPathSync
from flext_tests import tm


class TestRewritePoetry:
    def test_rewrite_poetry_no_tool(self) -> None:
        tm.that(
            FlextInfraDependencyPathSync._rewrite_poetry(
                TOMLDocument(), is_root=True, mode="workspace"
            ),
            eq=[],
        )

    def test_rewrite_poetry_no_poetry(self) -> None:
        doc = TOMLDocument()
        doc["tool"] = tomlkit.table()
        tm.that(
            FlextInfraDependencyPathSync._rewrite_poetry(
                doc, is_root=True, mode="workspace"
            ),
            eq=[],
        )

    def test_rewrite_poetry_no_dependencies(self) -> None:
        doc = TOMLDocument()
        doc["tool"] = {"poetry": tomlkit.table()}
        tm.that(
            FlextInfraDependencyPathSync._rewrite_poetry(
                doc, is_root=True, mode="workspace"
            ),
            eq=[],
        )

    def test_rewrite_poetry_non_dict_dependencies(self) -> None:
        doc = TOMLDocument()
        doc["tool"] = {"poetry": {"dependencies": "not-a-dict"}}
        tm.that(
            FlextInfraDependencyPathSync._rewrite_poetry(
                doc, is_root=True, mode="workspace"
            ),
            eq=[],
        )

    def test_rewrite_poetry_rewrite_path_dep(self) -> None:
        doc = TOMLDocument()
        doc["tool"] = {
            "poetry": {
                "dependencies": {"flext-core": {"path": ".flext-deps/flext-core"}}
            },
        }
        changes = FlextInfraDependencyPathSync._rewrite_poetry(
            doc, is_root=True, mode="workspace"
        )
        tm.that(len(changes) > 0, eq=True)
        tm.that(doc.as_string(), contains='path = "flext-core"')

    def test_rewrite_poetry_skip_non_path_dep(self) -> None:
        doc = TOMLDocument()
        doc["tool"] = {"poetry": {"dependencies": {"requests": {"version": "^2.0.0"}}}}
        tm.that(
            FlextInfraDependencyPathSync._rewrite_poetry(
                doc, is_root=True, mode="workspace"
            ),
            eq=[],
        )

    def test_rewrite_poetry_non_dict_value(self) -> None:
        doc = TOMLDocument()
        doc["tool"] = {"poetry": {"dependencies": {"requests": "^2.0.0"}}}
        tm.that(
            FlextInfraDependencyPathSync._rewrite_poetry(
                doc, is_root=True, mode="workspace"
            ),
            eq=[],
        )

    def test_rewrite_poetry_empty_path(self) -> None:
        doc = TOMLDocument()
        doc["tool"] = {"poetry": {"dependencies": {"flext-core": {"path": ""}}}}
        tm.that(
            FlextInfraDependencyPathSync._rewrite_poetry(
                doc, is_root=True, mode="workspace"
            ),
            eq=[],
        )

    def test_rewrite_poetry_non_string_path(self) -> None:
        doc = TOMLDocument()
        doc["tool"] = {"poetry": {"dependencies": {"flext-core": {"path": 123}}}}
        tm.that(
            FlextInfraDependencyPathSync._rewrite_poetry(
                doc, is_root=True, mode="workspace"
            ),
            eq=[],
        )

    def test_rewrite_poetry_subproject_mode(self) -> None:
        doc = TOMLDocument()
        doc["tool"] = {
            "poetry": {
                "dependencies": {"flext-core": {"path": ".flext-deps/flext-core"}}
            },
        }
        changes = FlextInfraDependencyPathSync._rewrite_poetry(
            doc, is_root=False, mode="workspace"
        )
        tm.that(len(changes) > 0, eq=True)
        tm.that(doc.as_string(), contains='path = "../flext-core"')


def test_rewrite_poetry_with_non_dict_value() -> None:
    doc = tomlkit.document()
    poetry = tomlkit.table()
    deps = tomlkit.table()
    deps["flext-core"] = "string-value"
    poetry["dependencies"] = deps
    tool = tomlkit.table()
    tool["poetry"] = poetry
    doc["tool"] = tool
    tm.that(
        len(
            FlextInfraDependencyPathSync._rewrite_poetry(
                doc, is_root=False, mode="workspace"
            )
        ),
        eq=0,
    )


def test_rewrite_poetry_no_tool_table() -> None:
    tm.that(
        len(
            FlextInfraDependencyPathSync._rewrite_poetry(
                tomlkit.document(), is_root=False, mode="workspace"
            )
        ),
        eq=0,
    )


def test_rewrite_poetry_no_poetry_table() -> None:
    doc = tomlkit.document()
    doc["tool"] = tomlkit.table()
    tm.that(
        len(
            FlextInfraDependencyPathSync._rewrite_poetry(
                doc, is_root=False, mode="workspace"
            )
        ),
        eq=0,
    )
