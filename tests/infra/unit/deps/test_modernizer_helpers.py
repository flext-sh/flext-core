from __future__ import annotations

import tomlkit
from tomlkit.items import Array, Table

from flext_infra.deps.modernizer import (
    _array,
    _as_string_list,
    _canonical_dev_dependencies,
    _dedupe_specs,
    _dep_name,
    _ensure_table,
    _project_dev_groups,
    _unwrap_item,
)
from flext_tests import tm
from tests.infra import h


class TestDepName:
    def test_variants(self) -> None:
        tm.that(_dep_name("requests"), eq="requests")
        tm.that(_dep_name("requests>=2.0"), eq="requests")
        tm.that(
            _dep_name("requests @ git+https://github.com/psf/requests.git"),
            eq="requests",
        )
        tm.that(_dep_name("my_package"), eq="my-package")
        tm.that(_dep_name("  requests  "), eq="requests")
        tm.that(_dep_name(""), eq="")
        tm.that(_dep_name("Django>=3.0,<4.0"), eq="django")


class TestDedupeSpecs:
    def test_dedupe_specs(self) -> None:
        result = _dedupe_specs(["requests>=2.0", "django>=3.0"])
        tm.that(result, length=2)
        result = _dedupe_specs(["requests>=2.0", "requests>=2.1", "django>=3.0"])
        tm.that(result, length=2)
        tm.that([_dep_name(spec) for spec in result], contains="requests")
        tm.that(_dedupe_specs([]), eq=[])
        sorted_result = _dedupe_specs(["zebra>=1.0", "apple>=1.0"])
        tm.that(_dep_name(sorted_result[0]) < _dep_name(sorted_result[1]), eq=True)
        tm.that(_dedupe_specs(["Requests>=2.0", "requests>=2.1"]), length=1)


class TestUnwrapItem:
    def test_unwrap_item_values(self) -> None:
        tm.that(_unwrap_item("test"), eq="test")
        tm.that(_unwrap_item(None), eq=None)
        doc = tomlkit.document()
        doc["key"] = "value"
        tm.that(_unwrap_item(doc["key"]), eq="value")
        value = {"key": "value"}
        tm.that(_unwrap_item(value), eq=value)


class TestAsStringList:
    def test_as_string_list_values(self) -> None:
        tm.that(_as_string_list(["a", "b", "c"]), eq=["a", "b", "c"])
        tm.that(_as_string_list(None), eq=[])
        tm.that(_as_string_list("test"), eq=[])
        tm.that(_as_string_list({"key": "value"}), eq=[])
        tm.that(_as_string_list(["item1", "item2"]), eq=["item1", "item2"])
        doc = tomlkit.document()
        doc["items"] = ["a", "b"]
        tm.that(_as_string_list(doc["items"]), eq=["a", "b"])
        tm.that(_as_string_list(42), eq=[])
        doc["value"] = 42
        tm.that(_as_string_list(doc["value"]), eq=[])


class TestArray:
    def test_array_construction(self) -> None:
        result = _array(["a", "b", "c"])
        tm.that(isinstance(result, Array), eq=True)
        tm.that(result, length=3)
        empty = _array([])
        tm.that(isinstance(empty, Array), eq=True)
        tm.that(empty, length=0)
        single = _array(["single"])
        tm.that(isinstance(single, Array), eq=True)
        tm.that(single, length=1)


class TestEnsureTable:
    def test_ensure_table_paths(self) -> None:
        parent = tomlkit.table()
        created = _ensure_table(parent, "new_key")
        tm.that(isinstance(created, Table), eq=True)
        tm.that("new_key" in parent, eq=True)
        parent["existing"] = tomlkit.table()
        existing = _ensure_table(parent, "existing")
        tm.that(isinstance(existing, Table), eq=True)
        tm.that(existing is parent["existing"], eq=True)
        parent["key"] = "string_value"
        overwritten = _ensure_table(parent, "key")
        tm.that(isinstance(overwritten, Table), eq=True)


class TestProjectDevGroups:
    def test_project_dev_groups_paths(self) -> None:
        doc = tomlkit.document()
        doc["project"] = {
            "optional-dependencies": {
                "dev": ["pytest"],
                "docs": ["sphinx"],
                "security": ["bandit"],
                "test": ["coverage"],
                "typings": ["mypy"],
            },
        }
        result = _project_dev_groups(doc)
        tm.that(result["dev"], eq=["pytest"])
        tm.that(result["docs"], eq=["sphinx"])
        tm.that(result["security"], eq=["bandit"])
        tm.that(result["test"], eq=["coverage"])
        tm.that(result["typings"], eq=["mypy"])
        tm.that(_project_dev_groups(tomlkit.document()), eq={})
        doc2 = tomlkit.document()
        doc2["project"] = {"name": "test"}
        tm.that(_project_dev_groups(doc2), eq={})

    def test_project_dev_groups_partial(self) -> None:
        doc = tomlkit.document()
        doc["project"] = {"optional-dependencies": {"dev": ["pytest"]}}
        result = _project_dev_groups(doc)
        tm.that(result["dev"], eq=["pytest"])
        tm.that(result["docs"], eq=[])


class TestCanonicalDevDependencies:
    def test_canonical_dev_dependencies(self) -> None:
        doc = tomlkit.document()
        doc["project"] = {
            "optional-dependencies": {
                "dev": ["pytest"],
                "docs": ["sphinx"],
                "security": ["bandit"],
                "test": ["coverage"],
                "typings": ["mypy"],
            },
        }
        result = _canonical_dev_dependencies(doc)
        tm.that(result, length=5)
        tm.that(any("pytest" in item for item in result), eq=True)
        tm.that(_canonical_dev_dependencies(tomlkit.document()), eq=[])
        doc2 = tomlkit.document()
        doc2["project"] = {
            "optional-dependencies": {"dev": ["pytest>=7.0"], "test": ["pytest>=6.0"]},
        }
        tm.that(_canonical_dev_dependencies(doc2), length=1)


def test_unwrap_item_with_item() -> None:
    doc = tomlkit.document()
    doc["value"] = "test_value"
    tm.that(_unwrap_item(doc["value"]), eq="test_value")


def test_unwrap_item_with_none() -> None:
    tm.that(_unwrap_item(None), eq=None)


def test_as_string_list_with_item() -> None:
    doc = tomlkit.document()
    doc["items"] = ["a", "b", "c"]
    tm.that(_as_string_list(doc["items"]), eq=["a", "b", "c"])


def test_as_string_list_with_string() -> None:
    tm.that(_as_string_list("string"), eq=[])


def test_as_string_list_with_mapping() -> None:
    tm.that(_as_string_list({"key": "value"}), eq=[])


def test_as_string_list_with_item_unwrap_returns_none() -> None:
    doc = tomlkit.document()
    doc["items"] = 42
    tm.that(_as_string_list(doc["items"]), eq=[])
    tm.that(hasattr(h, "assert_ok"), eq=True)


def test_ensure_table_with_non_table_value_uncovered() -> None:
    parent = tomlkit.table()
    parent["key"] = "string_value"
    result = _ensure_table(parent, "key")
    tm.that(isinstance(result, Table), eq=True)
    tm.that("key" in parent, eq=True)
