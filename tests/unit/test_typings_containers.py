"""Behavioral tests for the runtime typing container models (m.Dict / m.ConfigMap / m.ObjectList) and scalar/tuple type contracts.

These tests assert only OBSERVABLE PUBLIC BEHAVIOR: the mapping / sequence API,
model_dump round-trips, validation error paths, and the runtime scalar/tuple
type contracts. No private attributes, internal collaborators, or MRO shape are
inspected.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

import pytest

from flext_tests import m as ftm, tm
from tests import m, t


class TestsFlextCoreTypingsContainers:
    """Verify public container typing aliases."""

    # ---- m.Dict: mapping read contract -------------------------------------

    def test_dict_empty_has_zero_length(self) -> None:
        """An empty m.Dict reports length 0 and is falsy."""
        d = m.Dict(root={})
        tm.that(len(d), eq=0)
        tm.that(bool(d.root), eq=False)

    def test_dict_getitem_returns_stored_values(self) -> None:
        """m.Dict[key] returns the value stored under that key."""
        d = m.Dict(root={"key": "value", "num": 42})
        tm.that(d["key"], eq="value")
        tm.that(d["num"], eq=42)

    def test_dict_getitem_missing_key_raises_key_error(self) -> None:
        """m.Dict[missing] raises KeyError, matching dict semantics."""
        d = m.Dict(root={"key": "value"})
        with pytest.raises(KeyError):
            _ = d["missing"]

    @pytest.mark.parametrize(
        ("present", "expected"), [("key", True), ("missing", False)]
    )
    def test_dict_contains_reports_membership(
        self, present: str, expected: bool
    ) -> None:
        """The 'in' operator reflects actual key membership."""
        d = m.Dict(root={"key": "value"})
        tm.that(present in d, eq=expected)

    def test_dict_get_returns_value_or_default(self) -> None:
        """get() returns the stored value, or the supplied default when absent."""
        d = m.Dict(root={"key": "value"})
        tm.that(d.get("key"), eq="value")
        tm.that(d.get("missing", "fallback"), eq="fallback")

    def test_dict_get_missing_without_default_is_none(self) -> None:
        """get() on an absent key with no default yields None."""
        d = m.Dict(root={"key": "value"})
        tm.that(d.get("missing"), none=True)

    def test_dict_keys_values_items_reflect_contents(self) -> None:
        """keys/values/items views expose exactly the stored pairs."""
        d = m.Dict(root={"a": 1, "b": 2})
        tm.that(sorted(d.keys()), eq=["a", "b"])
        tm.that(set(d.values()), eq={1, 2})
        tm.that(dict(d.items()), eq={"a": 1, "b": 2})

    # ---- m.Dict: mapping mutation contract ---------------------------------

    def test_dict_setdefault_inserts_only_when_absent(self) -> None:
        """Setdefault inserts+returns for a new key, but preserves an existing one."""
        d = m.Dict(root={"a": 1})
        tm.that(d.setdefault("b", 9), eq=9)
        tm.that(d.setdefault("a", 0), eq=1)
        tm.that(d["b"], eq=9)
        tm.that(d["a"], eq=1)

    def test_dict_update_merges_new_pairs(self) -> None:
        """Update merges the supplied mapping into the container."""
        d = m.Dict(root={"a": 1})
        d.update({"b": 2, "a": 3})
        tm.that(d["a"], eq=3)
        tm.that(d["b"], eq=2)
        tm.that(len(d), eq=2)

    def test_dict_pop_removes_and_returns_value(self) -> None:
        """Pop removes the key and returns its value; default covers absence."""
        d = m.Dict(root={"a": 1, "b": 2})
        tm.that(d.pop("a"), eq=1)
        tm.that("a" in d, eq=False)
        tm.that(d.pop("missing", "fallback"), eq="fallback")

    def test_dict_clear_empties_container(self) -> None:
        """Clear removes every entry."""
        d = m.Dict(root={"a": 1, "b": 2})
        d.clear()
        tm.that(len(d), eq=0)

    # ---- m.Dict: serialization + validation contract -----------------------

    def test_dict_model_dump_returns_underlying_mapping(self) -> None:
        """model_dump round-trips the validated mapping as a plain dict."""
        payload: dict[str, t.JsonPayload] = {"a": 1, "b": "two"}
        d = m.Dict(root=payload)
        tm.that(d.model_dump(), eq=payload)

    def test_dict_rejects_non_mapping_root(self) -> None:
        """Constructing m.Dict from a non-mapping value fails validation."""
        with pytest.raises(ftm.ValidationError):
            m.Dict.model_validate(["not", "a", "mapping"])

    # ---- m.ConfigMap: shares the mapping contract --------------------------

    def test_configmap_getitem_returns_settings(self) -> None:
        """m.ConfigMap exposes configuration values by key."""
        cm = m.ConfigMap(root={"timeout": 30, "debug": False})
        tm.that(cm["timeout"], eq=30)
        tm.that(cm["debug"], eq=False)

    def test_configmap_len_counts_entries(self) -> None:
        """m.ConfigMap reports its number of settings via len()."""
        cm = m.ConfigMap(root={"a": 1, "b": 2})
        tm.that(len(cm), eq=2)

    def test_configmap_model_dump_round_trips(self) -> None:
        """m.ConfigMap serializes back to the original settings mapping."""
        payload: dict[str, t.JsonPayload] = {"timeout": 30, "debug": False}
        tm.that(m.ConfigMap(root=payload).model_dump(), eq=payload)

    # ---- m.ObjectList: sequence contract -----------------------------------

    def test_object_list_preserves_order_and_contents(self) -> None:
        """m.ObjectList exposes its validated items in insertion order."""
        ol = m.ObjectList(root=["item1", 42, True])
        tm.that(len(ol), eq=3)
        tm.that(ol.root, eq=["item1", 42, True])

    def test_object_list_empty_is_falsy(self) -> None:
        """An empty m.ObjectList is falsy and has length 0."""
        ol = m.ObjectList(root=[])
        tm.that(len(ol), eq=0)
        tm.that(bool(ol), eq=False)

    def test_object_list_non_empty_is_truthy(self) -> None:
        """A populated m.ObjectList is truthy."""
        tm.that(bool(m.ObjectList(root=["x"])), eq=True)

    def test_object_list_model_dump_returns_list(self) -> None:
        """model_dump round-trips the validated sequence as a plain list."""
        payload: list[t.JsonPayload] = ["a", 1, True]
        tm.that(m.ObjectList(root=payload).model_dump(), eq=payload)

    def test_object_list_rejects_non_sequence_root(self) -> None:
        """Constructing m.ObjectList from a mapping fails validation."""
        with pytest.raises(ftm.ValidationError):
            m.ObjectList.model_validate({"not": "a list"})

    # ---- t.SCALAR_TYPES: runtime scalar contract ---------------------------

    @pytest.mark.parametrize(
        "scalar", ["text", 42, math.pi, True, datetime(2025, 1, 1, tzinfo=UTC)]
    )
    def test_scalar_types_accepts_every_scalar(
        self, scalar: str | float | bool | datetime
    ) -> None:
        """SCALAR_TYPES is an isinstance-usable tuple covering all scalar kinds."""
        tm.that(isinstance(scalar, t.SCALAR_TYPES), eq=True)

    @pytest.mark.parametrize("nonscalar", [["list"], {"dict": 1}, ("tuple",)])
    def test_scalar_types_rejects_containers(
        self, nonscalar: list[str] | dict[str, int] | tuple[str]
    ) -> None:
        """Container values are not members of the scalar runtime contract."""
        tm.that(isinstance(nonscalar, t.SCALAR_TYPES), eq=False)

    # ---- t tuple aliases: arity contract via validation --------------------

    def test_pair_alias_enforces_two_element_arity(self) -> None:
        """t.Pair validates a 2-tuple and rejects other arities."""
        adapter: ftm.TypeAdapter[t.Pair[int, str]] = ftm.TypeAdapter(t.Pair[int, str])
        tm.that(adapter.validate_python((1, "x")), eq=(1, "x"))
        with pytest.raises(ftm.ValidationError):
            adapter.validate_python((1, "x", "extra"))

    def test_triple_alias_enforces_three_element_arity(self) -> None:
        """t.Triple validates a 3-tuple and rejects shorter tuples."""
        adapter: ftm.TypeAdapter[t.Triple[int, str, bool]] = ftm.TypeAdapter(
            t.Triple[int, str, bool]
        )
        tm.that(adapter.validate_python((1, "x", True)), eq=(1, "x", True))
        with pytest.raises(ftm.ValidationError):
            adapter.validate_python((1, "x"))

    def test_int_pair_alias_validates_two_ints(self) -> None:
        """t.IntPair coerces and validates a pair of ints, rejecting wrong arity."""
        adapter: ftm.TypeAdapter[t.IntPair] = ftm.TypeAdapter(t.IntPair)
        tm.that(adapter.validate_python((1, 2)), eq=(1, 2))
        with pytest.raises(ftm.ValidationError):
            adapter.validate_python((1, 2, 3))

    def test_variadic_tuple_alias_accepts_any_length(self) -> None:
        """t.VariadicTuple validates homogeneous tuples of arbitrary length."""
        adapter: ftm.TypeAdapter[t.VariadicTuple[int]] = ftm.TypeAdapter(
            t.VariadicTuple[int]
        )
        tm.that(adapter.validate_python(()), eq=())
        tm.that(adapter.validate_python((1, 2, 3)), eq=(1, 2, 3))
