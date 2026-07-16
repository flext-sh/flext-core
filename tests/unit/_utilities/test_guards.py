"""Behavioral contract tests for flext-core type guards.

Exercises the public guard surface exposed through the test ``u`` facade
(``FlextUtilitiesGuardsTypeProtocol`` / ``FlextUtilitiesGuardsTypeCore``).
Every assertion targets observable return-value behavior of a public guard,
never an internal helper or private attribute.
"""

from __future__ import annotations

import pytest

from flext_core import m, p, t
from tests.utilities import u


class _SampleModel(m.BaseModel):
    """Minimal real Pydantic model used to probe model-exclusion behavior."""

    value: int = 1


class TestsFlextCoreGuards:
    """Public-contract behavior of the flext-core type guards."""

    # ------------------------------------------------------------------
    # matches_type — string specs
    # ------------------------------------------------------------------
    @pytest.mark.parametrize(
        ("value", "spec", "expected"),
        [
            ("x", "str", True),
            (1, "str", False),
            (1, "int", True),
            (1.5, "float", True),
            (True, "bool", True),
            ({"k": "v"}, "mapping", True),
            ([1, 2], "list", True),
            ((1, 2), "tuple", True),
            (None, "none", True),
            ("x", "none", False),
        ],
    )
    def test_matches_type_string_spec_reflects_runtime_type(
        self,
        value: t.JsonValue | None,
        spec: str,
        expected: bool,
    ) -> None:
        assert u.matches_type(value, spec) is expected

    @pytest.mark.parametrize(
        ("value", "spec", "expected"),
        [
            (" a ", "string_non_empty", True),
            ("", "string_non_empty", False),
            ("   ", "string_non_empty", False),
            ({"k": "v"}, "dict_non_empty", True),
            ({}, "dict_non_empty", False),
            (["v"], "list_non_empty", True),
            ([], "list_non_empty", False),
        ],
    )
    def test_matches_type_non_empty_specs_require_content(
        self,
        value: t.JsonValue,
        spec: str,
        expected: bool,
    ) -> None:
        assert u.matches_type(value, spec) is expected

    def test_matches_type_is_case_insensitive_for_string_specs(self) -> None:
        assert u.matches_type("x", "STR") is True
        assert u.matches_type(1, "Int") is True

    def test_matches_type_unknown_string_spec_returns_false(self) -> None:
        assert u.matches_type("x", "no_such_spec") is False

    @pytest.mark.parametrize(
        "spec", ["string_non_empty", "dict_non_empty", "list_non_empty"]
    )
    def test_matches_type_excludes_pydantic_models_from_non_empty_specs(
        self,
        spec: str,
    ) -> None:
        # A populated model would otherwise satisfy dict-like checks; the guard
        # contract deliberately excludes Pydantic models from these specs.
        assert u.matches_type(_SampleModel(), spec) is False

    # ------------------------------------------------------------------
    # matches_type — type and tuple specs
    # ------------------------------------------------------------------
    @pytest.mark.parametrize(
        ("value", "spec", "expected"),
        [
            (5, int, True),
            ("x", int, False),
            (5, (int, str), True),
            ("x", (int, str), True),
            (5.0, (int, str), False),
        ],
    )
    def test_matches_type_type_and_tuple_specs(
        self,
        value: t.JsonValue,
        spec: type | tuple[type, ...],
        expected: bool,
    ) -> None:
        assert u.matches_type(value, spec) is expected

    def test_matches_type_invalid_scalar_spec_returns_false(self) -> None:
        # An out-of-contract spec (a bare scalar) must not match anything.
        assert u.matches_type("x", 123) is False

    # ------------------------------------------------------------------
    # container — recursive JSON-compatibility contract
    # ------------------------------------------------------------------
    @pytest.mark.parametrize(
        "value",
        [
            "x",
            1,
            1.5,
            True,
            {"k": "v"},
            [1, "x", True],
            {"a": {"b": [1, 2]}},
            [[1], [2, 3]],
            [],
            {},
        ],
    )
    def test_container_accepts_scalars_and_nested_json(
        self,
        value: t.JsonValue,
    ) -> None:
        assert u.container(value) is True

    @pytest.mark.parametrize(
        "value",
        [
            None,
            object(),
            [1, object()],
            {"a": object()},
            {"a": [object()]},
        ],
    )
    def test_container_rejects_none_and_non_json_members(
        self,
        value: t.JsonValue | None,
    ) -> None:
        assert u.container(value) is False

    def test_all_container_mapping_values_accepts_json_values(self) -> None:
        mapping: dict[str, t.JsonValue] = {"a": 1, "b": [2, 3], "c": {"d": "e"}}
        assert u.all_container_mapping_values(mapping) is True

    # ------------------------------------------------------------------
    # scalar / primitive
    # ------------------------------------------------------------------
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("x", True),
            (1, True),
            (1.5, True),
            (True, True),
            ([1], False),
            ({}, False),
            (None, False),
        ],
    )
    def test_scalar_identifies_scalar_values(
        self,
        value: t.JsonValue | None,
        expected: bool,
    ) -> None:
        assert u.scalar(value) is expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("x", True),
            (1, True),
            (1.5, True),
            (True, True),
            ([1], False),
            ({"k": 1}, False),
        ],
    )
    def test_primitive_identifies_primitive_values(
        self,
        value: t.JsonValue,
        expected: bool,
    ) -> None:
        assert u.primitive(value) is expected

    # ------------------------------------------------------------------
    # collection guards
    # ------------------------------------------------------------------
    def test_mapping_and_list_value_discriminate_collections(self) -> None:
        assert u.mapping({"k": "v"}) is True
        assert u.mapping([1, 2]) is False
        assert u.list_value([1, 2]) is True
        assert u.list_value((1, 2)) is False

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ([1], True),
            ((1,), True),
            ("text", False),
            (b"bytes", False),
            ({"k": 1}, False),
        ],
    )
    def test_list_like_excludes_strings_and_bytes(
        self,
        value: t.JsonValue,
        expected: bool,
    ) -> None:
        assert u.list_like(value) is expected

    def test_dict_like_accepts_only_mappings(self) -> None:
        assert u.dict_like({"k": "v"}) is True
        assert u.dict_like([1, 2]) is False
        assert u.dict_like("text") is False

    # ------------------------------------------------------------------
    # emptiness / non-empty guards
    # ------------------------------------------------------------------
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (None, True),
            ("", True),
            ("x", False),
            ([], True),
            ([1], False),
            ({}, True),
            ({"k": 1}, False),
            (0, False),
        ],
    )
    def test_empty_value_reports_absence_or_empty_containers(
        self,
        value: t.JsonValue | None,
        expected: bool,
    ) -> None:
        assert u.empty_value(value) is expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [("x", True), (" a ", True), ("", False), ("   ", False), (1, False)],
    )
    def test_string_non_empty_requires_non_blank_string(
        self,
        value: t.GuardInput,
        expected: bool,
    ) -> None:
        assert u.string_non_empty(value) is expected

    def test_dict_non_empty_requires_populated_mapping(self) -> None:
        assert u.dict_non_empty({"k": 1}) is True
        assert u.dict_non_empty({}) is False
        assert u.dict_non_empty(None) is False

    # ------------------------------------------------------------------
    # instance_of / in_ / type_name
    # ------------------------------------------------------------------
    def test_instance_of_matches_concrete_type(self) -> None:
        assert u.instance_of(5, int) is True
        assert u.instance_of("x", int) is False

    @pytest.mark.parametrize(
        ("value", "container", "expected"),
        [
            (1, [1, 2, 3], True),
            (4, [1, 2, 3], False),
            ("k", {"k": 1}, True),
            (2, (1, 2), True),
            (1, {1, 2}, True),
            (1, "not-a-real-container", False),
        ],
    )
    def test_in_membership_only_for_true_containers(
        self,
        value: t.GuardInput,
        container: t.GuardInput,
        expected: bool,
    ) -> None:
        assert u.in_(value, container) is expected

    def test_in_returns_false_for_unhashable_value(self) -> None:
        # A list is unhashable; membership in a set raises TypeError internally,
        # which the guard must swallow into a plain False.
        assert u.in_([1], {1, 2}) is False

    def test_type_name_returns_runtime_qualname(self) -> None:
        assert u.type_name("x") == "str"
        assert u.type_name(1) == "int"
        assert u.type_name([1]) == "list"
        assert u.type_name(None) == "NoneType"
