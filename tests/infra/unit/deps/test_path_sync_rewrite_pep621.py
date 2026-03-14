from __future__ import annotations

import tomlkit
from tomlkit.toml_document import TOMLDocument

from flext_infra.deps.path_sync import FlextInfraDependencyPathSync
from flext_tests import tm

_PATH_SYNC = FlextInfraDependencyPathSync()
_rewrite_pep621 = _PATH_SYNC._rewrite_pep621


class TestRewritePep621:
    def test_rewrite_pep621_no_project(self) -> None:
        doc = TOMLDocument()
        tm.that(
            _rewrite_pep621(doc, is_root=True, mode="workspace", internal_names=set()),
            eq=[],
        )

    def test_rewrite_pep621_no_dependencies(self) -> None:
        doc = TOMLDocument()
        doc["project"] = tomlkit.table()
        tm.that(
            _rewrite_pep621(doc, is_root=True, mode="workspace", internal_names=set()),
            eq=[],
        )

    def test_rewrite_pep621_non_list_dependencies(self) -> None:
        doc = TOMLDocument()
        doc["project"] = {"dependencies": "not-a-list"}
        tm.that(
            _rewrite_pep621(doc, is_root=True, mode="workspace", internal_names=set()),
            eq=[],
        )

    def test_rewrite_pep621_rewrite_path_dep(self) -> None:
        doc = TOMLDocument()
        doc["project"] = {
            "dependencies": ["flext-core @ file://.flext-deps/flext-core"]
        }
        changes = _rewrite_pep621(
            doc,
            is_root=True,
            mode="workspace",
            internal_names={"flext-core"},
        )
        tm.that(len(changes) > 0, eq=True)
        unwrapped = doc.unwrap()
        tm.that(
            "flext-core @ file:./flext-core" in unwrapped["project"]["dependencies"][0],
            eq=True,
        )

    def test_rewrite_pep621_skip_external_dep(self) -> None:
        doc = TOMLDocument()
        doc["project"] = {"dependencies": ["requests>=2.0.0"]}
        tm.that(
            _rewrite_pep621(
                doc,
                is_root=True,
                mode="workspace",
                internal_names={"flext-core"},
            ),
            eq=[],
        )

    def test_rewrite_pep621_with_marker(self) -> None:
        doc = TOMLDocument()
        doc["project"] = {
            "dependencies": [
                'flext-core @ file://.flext-deps/flext-core ; python_version >= "3.8"',
            ],
        }
        changes = _rewrite_pep621(
            doc,
            is_root=True,
            mode="workspace",
            internal_names={"flext-core"},
        )
        tm.that(len(changes) > 0, eq=True)
        unwrapped = doc.unwrap()
        tm.that(
            'python_version >= "3.8"' in unwrapped["project"]["dependencies"][0],
            eq=True,
        )

    def test_rewrite_pep621_non_string_item(self) -> None:
        doc = TOMLDocument()
        doc["project"] = {
            "dependencies": [123, "flext-core @ file://.flext-deps/flext-core"]
        }
        changes = _rewrite_pep621(
            doc,
            is_root=True,
            mode="workspace",
            internal_names={"flext-core"},
        )
        tm.that(len(changes), eq=1)

    def test_rewrite_pep621_subproject_mode(self) -> None:
        doc = TOMLDocument()
        doc["project"] = {
            "dependencies": ["flext-core @ file://.flext-deps/flext-core"]
        }
        changes = _rewrite_pep621(
            doc,
            is_root=False,
            mode="workspace",
            internal_names={"flext-core"},
        )
        tm.that(len(changes) > 0, eq=True)
        unwrapped = doc.unwrap()
        tm.that("../flext-core" in unwrapped["project"]["dependencies"][0], eq=True)


def test_rewrite_pep621_non_string_item() -> None:
    doc = tomlkit.document()
    project = tomlkit.table()
    project["dependencies"] = [123]
    doc["project"] = project
    changes = _rewrite_pep621(
        doc,
        is_root=False,
        mode="workspace",
        internal_names={"flext-core"},
    )
    tm.that(len(changes), eq=0)


def test_rewrite_pep621_no_project_table() -> None:
    doc = tomlkit.document()
    changes = _rewrite_pep621(
        doc,
        is_root=False,
        mode="workspace",
        internal_names={"flext-core"},
    )
    tm.that(len(changes), eq=0)


def test_rewrite_pep621_invalid_path_dep_regex() -> None:
    doc = tomlkit.document()
    project = tomlkit.table()
    project["dependencies"] = ["  flext-core @ file://.flext-deps/flext-core"]
    doc["project"] = project
    changes = _rewrite_pep621(
        doc,
        is_root=True,
        mode="workspace",
        internal_names={"flext-core"},
    )
    tm.that(len(changes), eq=0)
