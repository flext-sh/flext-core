from __future__ import annotations

"""Helper coverage tests for deps modernizer."""

import tomlkit

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


class TestDepName:
    """Tests dependency name normalization."""

    def test_dep_name_variants(self) -> None:
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
    """Tests dependency spec deduplication."""

    def test_dedupe_specs_paths(self) -> None:
        tm.that(_dedupe_specs(["requests>=2.0", "django>=3.0"]), length=2)
        deduped = _dedupe_specs(["requests>=2.0", "requests>=2.1", "django>=3.0"])
        tm.that(deduped, length=2)
        tm.that([_dep_name(spec) for spec in deduped], contains="requests")
        tm.that(_dedupe_specs([]), eq=[])
        sorted_specs = _dedupe_specs(["zebra>=1.0", "apple>=1.0"])
        tm.that(_dep_name(sorted_specs[0]) < _dep_name(sorted_specs[1]), eq=True)
        tm.that(_dedupe_specs(["Requests>=2.0", "requests>=2.1"]), length=1)


class TestUnwrapItem:
    """Tests tomlkit item unwrapping."""

    def test_unwrap_item_variants(self) -> None:
        tm.that(_unwrap_item("test"), eq="test")
        tm.that(_unwrap_item(None), eq=None)
        doc = tomlkit.document()
        doc["key"] = "value"
        tm.that(_unwrap_item(doc["key"]), eq="value")
        mapping = {"key": "value"}
        tm.that(_unwrap_item(mapping), eq=mapping)


class TestAsStringList:
    """Tests conversion to string list."""

    def test_as_string_list_variants(self) -> None:
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
    """Tests array creation helper."""

    def test_array_builds_items(self) -> None:
        tm.that(len(_array(["a", "b", "c"])), eq=3)
        tm.that(len(_array([])), eq=0)
        tm.that(len(_array(["single"])), eq=1)


class TestEnsureTable:
    """Tests table creation helper."""

    def test_ensure_table_paths(self) -> None:
        parent = tomlkit.table()
        _ = _ensure_table(parent, "new_key")
        tm.that("new_key" in parent, eq=True)
        existing = tomlkit.table()
        parent["existing"] = existing
        tm.that(_ensure_table(parent, "existing") is existing, eq=True)
        parent["key"] = "string_value"
        tm.that("key" in parent, eq=True)


class TestProjectDevGroups:
    """Tests extraction of project dev groups."""

    def test_project_dev_groups(self) -> None:
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
        doc3 = tomlkit.document()
        doc3["project"] = {"optional-dependencies": {"dev": ["pytest"]}}
        partial = _project_dev_groups(doc3)
        tm.that(partial["dev"], eq=["pytest"])
        tm.that(partial["docs"], eq=[])


class TestCanonicalDevDependencies:
    """Tests canonical dev dependency aggregation."""

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
            "optional-dependencies": {"dev": ["pytest>=7.0"], "test": ["pytest>=6.0"]}
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


def test_ensure_table_with_non_table_value_uncovered() -> None:
    parent = tomlkit.table()
    parent["key"] = "string_value"
    _ = _ensure_table(parent, "key")
    tm.that("key" in parent, eq=True)
