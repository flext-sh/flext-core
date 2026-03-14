from __future__ import annotations

import pytest
import tomlkit
import tomlkit.items
from tomlkit.toml_document import TOMLDocument

from flext_infra import t, u
from flext_tests import tb, tm

array = u.Infra.array
as_string_list = u.Infra.as_string_list
canonical_dev_dependencies = u.Infra.canonical_dev_dependencies
dedupe_specs = u.Infra.dedupe_specs
dep_name = u.Infra.dep_name
ensure_table = u.Infra.ensure_table
project_dev_groups = u.Infra.project_dev_groups
unwrap_item = u.Infra.unwrap_item


@pytest.fixture
def doc() -> TOMLDocument:
    return tomlkit.document()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("requests", "requests"),
        ("requests>=2.0", "requests"),
        ("requests @ git+https://github.com/psf/requests.git", "requests"),
        ("my_package", "my-package"),
        ("  requests  ", "requests"),
        ("", ""),
        ("Django>=3.0,<4.0", "django"),
    ],
)
def test_dep_name(raw: str, expected: str) -> None:
    tm.that(dep_name(raw), eq=expected)


@pytest.mark.parametrize(
    ("specs", "expected_length", "expected_names", "check_sorted"),
    [
        (["requests>=2.0", "django>=3.0"], 2, ["requests", "django"], False),
        (["requests>=2.0", "requests>=2.1", "django>=3.0"], 2, ["requests"], False),
        ([], 0, [], False),
        (["zebra>=1.0", "apple>=1.0"], 2, [], True),
        (["Requests>=2.0", "requests>=2.1"], 1, ["requests"], False),
    ],
)
def test_dedupe_specs(
    specs: list[str],
    expected_length: int,
    expected_names: list[str],
    check_sorted: bool,
) -> None:
    deduped = dedupe_specs(specs)
    tm.that(deduped, length=expected_length)
    names = [dep_name(spec) for spec in deduped]
    for expected_name in expected_names:
        tm.that(names, has=expected_name)
    if check_sorted and len(deduped) > 1:
        assert dep_name(deduped[0]) < dep_name(deduped[1])


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("test", "test"),
        (None, None),
        ({"key": "value"}, {"key": "value"}),
    ],
)
def test_unwrap_item(value: t.Infra.InfraValue, expected: t.Infra.InfraValue) -> None:
    tm.that(unwrap_item(value), eq=expected)


def test_unwrap_item_toml_item(doc: TOMLDocument) -> None:
    doc["key"] = "value"
    tm.that(unwrap_item(doc["key"]), eq="value")


def _toml_item(value: str | int | list[str]) -> tomlkit.items.Item:
    """Create a tomlkit item with known types to avoid Unknown in overloads."""
    if isinstance(value, str):
        return tomlkit.items.String.from_raw(value)
    if isinstance(value, int):
        return tomlkit.items.Integer(
            value, trivia=tomlkit.items.Trivia(), raw=str(value)
        )
    # list[str] -> construct Array from String items
    str_items: list[tomlkit.items.Item] = [
        tomlkit.items.String.from_raw(v) for v in value
    ]
    return tomlkit.items.Array(str_items, trivia=tomlkit.items.Trivia())


def _toml_table_item() -> tomlkit.items.Item:
    """Create a tomlkit table item."""
    tbl = tomlkit.table()
    tbl["key"] = "value"
    return tbl


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (_toml_item(["a", "b", "c"]), ["a", "b", "c"]),
        (None, []),
        (_toml_item("test"), []),
        (_toml_table_item(), []),
        (_toml_item(42), []),
    ],
)
def test_as_string_list(value: tomlkit.items.Item | None, expected: list[str]) -> None:
    tm.that(as_string_list(value), eq=expected)


def test_as_string_list_toml_item(doc: TOMLDocument) -> None:
    doc["items"] = ["a", "b"]
    items_array: tomlkit.items.Item = _toml_item(["a", "b"])
    tm.that(as_string_list(items_array), eq=["a", "b"])
    doc["value"] = 42
    int_val: tomlkit.items.Item = _toml_item(42)
    tm.that(as_string_list(int_val), eq=[])


@pytest.mark.parametrize(
    ("items", "expected"),
    [(["a", "b", "c"], 3), ([], 0), (["single"], 1)],
)
def test_array(items: list[str], expected: int) -> None:
    tm.that(len(array(items)), eq=expected)


@pytest.mark.parametrize(
    "mode",
    ["new", "existing", "replace-non-table"],
)
def test_ensure_table(mode: str) -> None:
    parent = tomlkit.table()
    if mode == "existing":
        existing = tomlkit.table()
        parent["key"] = existing
        ensured = ensure_table(parent, "key")
        assert ensured is existing
        return
    if mode == "replace-non-table":
        parent["key"] = "string_value"
        _ = ensure_table(parent, "key")
        tm.that("key" in parent, eq=True)
        return
    _ = ensure_table(parent, "key")
    tm.that("key" in parent, eq=True)


def _doc_with_optional_deps(optional_deps: dict[str, list[str]]) -> TOMLDocument:
    doc = tomlkit.document()
    doc["project"] = tb().add("optional-dependencies", optional_deps).build()
    return doc


@pytest.mark.parametrize(
    ("optional_deps", "expected_dev", "expected_docs"),
    [
        (
            {
                "dev": ["pytest"],
                "docs": ["sphinx"],
                "security": ["bandit"],
                "test": ["coverage"],
                "typings": ["mypy"],
            },
            ["pytest"],
            ["sphinx"],
        ),
        ({"dev": ["pytest"]}, ["pytest"], []),
        ({}, [], []),
    ],
)
def test_project_dev_groups(
    optional_deps: dict[str, list[str]],
    expected_dev: list[str],
    expected_docs: list[str],
) -> None:
    groups = project_dev_groups(_doc_with_optional_deps(optional_deps))
    tm.that(groups.get("dev", []), eq=expected_dev)
    tm.that(groups.get("docs", []), eq=expected_docs)


def test_project_dev_groups_missing_sections(doc: TOMLDocument) -> None:
    tm.that(project_dev_groups(doc), eq={})
    doc["project"] = {"name": "test"}
    tm.that(project_dev_groups(doc), eq={})


@pytest.mark.parametrize(
    ("optional_deps", "expected_length", "expect_pytest"),
    [
        (
            {
                "dev": ["pytest"],
                "docs": ["sphinx"],
                "security": ["bandit"],
                "test": ["coverage"],
                "typings": ["mypy"],
            },
            5,
            True,
        ),
        ({}, 0, False),
        ({"dev": ["pytest>=7.0"], "test": ["pytest>=6.0"]}, 1, True),
    ],
)
def test_canonical_dev_dependencies(
    optional_deps: dict[str, list[str]], expected_length: int, expect_pytest: bool
) -> None:
    result = canonical_dev_dependencies(_doc_with_optional_deps(optional_deps))
    tm.that(result, length=expected_length)
    if expect_pytest:
        assert any("pytest" in item for item in result)
