"""Behavioral tests for the runtime typing container models.

Covers ``m.Dict``, ``m.ConfigMap``, ``m.ObjectList``, and the scalar/tuple
type contracts.

These tests assert only OBSERVABLE PUBLIC BEHAVIOR: the mapping / sequence API,
model_dump round-trips, validation error paths, and the runtime scalar/tuple
type contracts. No private attributes, internal collaborators, or MRO shape are
inspected.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

import flext_tests
import pytest
from typing_extensions import TypeForm

from tests import c, m, t, tm, u


class TestsFlextCoreJsonPayload:
    """Payloads admit native JSON containers and explicitly declared model leaves."""

    @staticmethod
    def test_direct_models_survive_runtime_payload_validation() -> None:
        """The runtime contract preserves a model leaf without serializing it."""
        model = m.ConfigMap(root={"name": "sample"})
        adapter: flext_tests.m.TypeAdapter[t.JsonPayload] = u.type_adapter(
            TypeForm(t.JsonPayload)
        )
        tm.ok(adapter.validate_python(model) is model)

    @staticmethod
    def test_nested_json_survives_runtime_payload_validation() -> None:
        """Nested collections use the existing canonical JsonValue contract."""
        payload: t.JsonPayload = {"items": [{"name": "sample"}]}
        adapter: flext_tests.m.TypeAdapter[t.JsonPayload] = u.type_adapter(
            TypeForm(t.JsonPayload)
        )
        tm.that(adapter.validate_python(payload), eq=payload)

    @staticmethod
    def test_models_inside_json_collections_require_an_owned_model_contract() -> None:
        """The JSON branch must not silently admit arbitrary model instances."""
        model = m.Metadata(attributes={"name": "sample"})
        adapter: flext_tests.m.TypeAdapter[t.JsonPayload] = u.type_adapter(
            TypeForm(t.JsonPayload)
        )
        with pytest.raises(flext_tests.m.ValidationError):
            adapter.validate_python({"items": [{"model": model}]})

    @staticmethod
    def test_runtime_payload_rejects_unmodeled_values() -> None:
        """An unconstrained value must not enter through an unrestricted branch."""
        adapter: flext_tests.m.TypeAdapter[t.JsonPayload] = u.type_adapter(
            TypeForm(t.JsonPayload)
        )

        with pytest.raises(flext_tests.m.ValidationError):
            adapter.validate_python(object())


class TestsFlextCoreTypingsContainers:
    # ---- m.Dict: mapping read contract -------------------------------------
    """Tests for ``FlextCoreTypingsContainers``."""

    @staticmethod
    def test_dict_empty_has_zero_length() -> None:
        """An empty m.Dict reports length 0 and is falsy."""
        d = m.Dict(root={})
        tm.that(len(d), eq=0)
        tm.fail(bool(d.root))

    @staticmethod
    def test_dict_getitem_returns_stored_values() -> None:
        """m.Dict[key] returns the value stored under that key."""
        d = m.Dict(root={"key": "value", "num": 42})
        tm.that(d["key"], eq="value")
        tm.that(d["num"], eq=42)

    @staticmethod
    def test_dict_getitem_missing_key_raises_key_error() -> None:
        """m.Dict[missing] raises KeyError, matching dict semantics."""
        d = m.Dict(root={"key": "value"})
        with pytest.raises(KeyError):
            _ = d["missing"]

    @pytest.mark.parametrize(
        ("present", "expected"),
        [("key", True), ("missing", False)],
    )
    @staticmethod
    def test_dict_contains_reports_membership(
        present: str,
        *,
        expected: bool,
    ) -> None:
        """The 'in' operator reflects actual key membership."""
        d = m.Dict(root={"key": "value"})
        tm.that(present in d, eq=expected)

    @staticmethod
    def test_dict_get_returns_value_or_default() -> None:
        """get() returns the stored value, or the supplied default when absent."""
        d = m.Dict(root={"key": "value"})
        tm.that(d.get("key"), eq="value")
        tm.that(d.get("missing", "fallback"), eq="fallback")

    @staticmethod
    def test_dict_get_missing_without_default_is_none() -> None:
        """get() on an absent key with no default yields None."""
        d = m.Dict(root={"key": "value"})
        tm.that(d.get("missing"), none=True)

    @staticmethod
    def test_dict_keys_values_items_reflect_contents() -> None:
        """keys/values/items views expose exactly the stored pairs."""
        d = m.Dict(root={"a": 1, "b": 2})
        tm.that(sorted(d.keys()), eq=["a", "b"])
        tm.that(set(d.values()), eq={1, 2})
        tm.that(dict(d.items()), eq={"a": 1, "b": 2})

    # ---- m.Dict: mapping mutation contract ---------------------------------

    @staticmethod
    def test_dict_setdefault_inserts_only_when_absent() -> None:
        """Setdefault inserts+returns for a new key, but preserves an existing one."""
        d = m.Dict(root={"a": 1})
        tm.that(d.setdefault("b", 9), eq=9)
        tm.that(d.setdefault("a", 0), eq=1)
        tm.that(d["b"], eq=9)
        tm.that(d["a"], eq=1)

    @staticmethod
    def test_dict_update_merges_new_pairs() -> None:
        """Update merges the supplied mapping into the container."""
        d = m.Dict(root={"a": 1})
        d.update({"b": 2, "a": 3})
        tm.that(d["a"], eq=3)
        tm.that(d["b"], eq=2)
        tm.that(len(d), eq=2)

    @staticmethod
    def test_dict_pop_removes_and_returns_value() -> None:
        """Pop removes the key and returns its value; default covers absence."""
        d = m.Dict(root={"a": 1, "b": 2})
        tm.that(d.pop("a"), eq=1)
        tm.fail("a" in d)
        tm.that(d.pop("missing", "fallback"), eq="fallback")

    @staticmethod
    def test_dict_clear_empties_container() -> None:
        """Clear removes every entry."""
        d = m.Dict(root={"a": 1, "b": 2})
        d.clear()
        tm.that(len(d), eq=0)

    # ---- m.Dict: serialization + validation contract -----------------------

    @staticmethod
    def test_dict_model_dump_returns_underlying_mapping() -> None:
        """model_dump round-trips the validated mapping as a plain dict."""
        payload: t.MutableMappingKV[str, t.JsonPayload] = {"a": 1, "b": "two"}
        d = m.Dict(root=payload)
        tm.that(d.model_dump(), eq=payload)

    @staticmethod
    def test_dict_rejects_non_mapping_root() -> None:
        """Constructing m.Dict from a non-mapping value fails validation."""
        with pytest.raises(flext_tests.m.ValidationError):
            m.Dict.model_validate(["not", "a", "mapping"])

    # ---- m.ConfigMap: shares the mapping contract --------------------------

    @staticmethod
    def test_configmap_getitem_returns_settings() -> None:
        """m.ConfigMap exposes configuration values by key."""
        cm = m.ConfigMap(root={"timeout": 30, "debug": False})
        tm.that(cm["timeout"], eq=30)
        tm.that(cm["debug"], eq=False)

    @staticmethod
    def test_configmap_len_counts_entries() -> None:
        """m.ConfigMap reports its number of settings via len()."""
        cm = m.ConfigMap(root={"a": 1, "b": 2})
        tm.that(len(cm), eq=2)

    @staticmethod
    def test_configmap_model_dump_round_trips() -> None:
        """m.ConfigMap serializes back to the original settings mapping."""
        payload: t.MutableMappingKV[str, t.JsonPayload] = {
            "timeout": 30,
            "debug": False,
        }
        tm.that(m.ConfigMap(root=payload).model_dump(), eq=payload)

    # ---- m.ObjectList: sequence contract -----------------------------------

    @staticmethod
    def test_object_list_preserves_order_and_contents() -> None:
        """m.ObjectList exposes its validated items in insertion order."""
        ol = m.ObjectList(root=["item1", 42, True])
        tm.that(len(ol), eq=3)
        tm.that(ol.root, eq=["item1", 42, True])

    @staticmethod
    def test_object_list_empty_is_falsy() -> None:
        """An empty m.ObjectList is falsy and has length 0."""
        ol = m.ObjectList(root=[])
        tm.that(len(ol), eq=0)
        tm.fail(bool(ol))

    @staticmethod
    def test_object_list_non_empty_is_truthy() -> None:
        """A populated m.ObjectList is truthy."""
        tm.ok(bool(m.ObjectList(root=["x"])))

    @staticmethod
    def test_object_list_model_dump_returns_list() -> None:
        """model_dump round-trips the validated sequence as a plain list."""
        payload: list[t.JsonPayload] = ["a", 1, True]
        tm.that(m.ObjectList(root=payload).model_dump(), eq=payload)

    @staticmethod
    def test_object_list_rejects_non_sequence_root() -> None:
        """Constructing m.ObjectList from a mapping fails validation."""
        with pytest.raises(flext_tests.m.ValidationError):
            m.ObjectList.model_validate({"not": "a list"})

    # ---- c.SCALAR_TYPES: runtime scalar contract ---------------------------

    @pytest.mark.parametrize(
        "scalar",
        ["text", 42, math.pi, True, datetime(2025, 1, 1, tzinfo=UTC)],
    )
    @staticmethod
    def test_scalar_types_accepts_every_scalar(
        *,
        scalar: str | float | bool | datetime,
    ) -> None:
        """SCALAR_TYPES is an isinstance-usable tuple covering all scalar kinds."""
        tm.ok(isinstance(scalar, c.SCALAR_TYPES))

    @pytest.mark.parametrize("nonscalar", [["list"], {"dict": 1}, ("tuple",)])
    @staticmethod
    def test_scalar_types_rejects_containers(
        nonscalar: list[str] | t.MappingKV[str, int] | tuple[str],
    ) -> None:
        """Container values are not members of the scalar runtime contract."""
        tm.fail(isinstance(nonscalar, c.SCALAR_TYPES))

    # ---- t tuple aliases: arity contract via validation --------------------

    @staticmethod
    def test_pair_alias_enforces_two_element_arity() -> None:
        """t.Pair validates a 2-tuple and rejects other arities."""
        adapter: flext_tests.m.TypeAdapter[t.Pair[int, str]] = u.type_adapter(
            TypeForm(t.Pair[int, str])
        )
        tm.that(adapter.validate_python((1, "x")), eq=(1, "x"))
        with pytest.raises(flext_tests.m.ValidationError):
            adapter.validate_python((1, "x", "extra"))

    @staticmethod
    def test_triple_alias_enforces_three_element_arity() -> None:
        """t.Triple validates a 3-tuple and rejects shorter tuples."""
        adapter: flext_tests.m.TypeAdapter[t.Triple[int, str, bool]] = u.type_adapter(
            TypeForm(t.Triple[int, str, bool]),
        )
        tm.that(adapter.validate_python((1, "x", True)), eq=(1, "x", True))
        with pytest.raises(flext_tests.m.ValidationError):
            adapter.validate_python((1, "x"))

    @staticmethod
    def test_int_pair_alias_validates_two_ints() -> None:
        """t.IntPair coerces and validates a pair of ints, rejecting wrong arity."""
        adapter: flext_tests.m.TypeAdapter[t.IntPair] = u.type_adapter(
            TypeForm(t.IntPair)
        )
        tm.that(adapter.validate_python((1, 2)), eq=(1, 2))
        with pytest.raises(flext_tests.m.ValidationError):
            adapter.validate_python((1, 2, 3))

    @staticmethod
    def test_variadic_tuple_alias_accepts_any_length() -> None:
        """t.VariadicTuple validates homogeneous tuples of arbitrary length."""
        adapter: flext_tests.m.TypeAdapter[t.VariadicTuple[int]] = u.type_adapter(
            TypeForm(t.VariadicTuple[int]),
        )
        tm.that(adapter.validate_python(()), eq=())
        tm.that(adapter.validate_python((1, 2, 3)), eq=(1, 2, 3))
