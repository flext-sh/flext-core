"""Behavioral tests for the FlextTypes cached TypeAdapter factory contract.

Exercises the public ``t.*`` surface only: adapter factories validate/reject
runtime data, memoize per-class, honor strict-vs-lax coercion, and preserve the
documented tiered ``Primitives subset Scalar subset Container`` invariant.
"""

from __future__ import annotations

from enum import StrEnum

import pytest
from flext_tests import tm

from tests import c
from tests import t


class _Color(StrEnum):
    RED = "red"


class TestsFlextCoreTypingsNew:
    """Verify public typing utilities."""

    def test_json_value_adapter_accepts_nested_json(self) -> None:
        """json_value_adapter validates arbitrarily nested JSON payloads."""
        payload: t.JsonValue = {"k": [1, 2, "x", True, None]}
        result = t.json_value_adapter().validate_python(payload)
        tm.that(result, eq=payload)

    def test_adapter_factory_is_memoized(self) -> None:
        """Repeated factory calls return the identical cached adapter."""
        tm.that(t.json_value_adapter() is t.json_value_adapter(), eq=True)

    def test_primitives_adapter_accepts_scalar_value(self) -> None:
        """primitives_adapter accepts a bare primitive value."""
        tm.that(t.primitives_adapter().validate_python(5), eq=5)

    def test_primitives_adapter_rejects_collection(self) -> None:
        """primitives_adapter rejects a non-primitive collection."""
        with pytest.raises(c.ValidationError):
            t.primitives_adapter().validate_python([1])

    @pytest.mark.parametrize(("valid", "invalid"), [(7, "7"), (0, 1.5)])
    def test_int_adapter_is_strict(self, valid: int, invalid: object) -> None:
        """int_adapter accepts real ints and rejects coercible non-ints."""
        tm.that(t.int_adapter().validate_python(valid), eq=valid)
        with pytest.raises(c.ValidationError):
            t.int_adapter().validate_python(invalid)

    @pytest.mark.parametrize("bad", [5, b"bytes"])
    def test_str_adapter_rejects_non_str(self, bad: object) -> None:
        """str_adapter is strict: non-string inputs are rejected."""
        tm.that(t.str_adapter().validate_python("hi"), eq="hi")
        with pytest.raises(c.ValidationError):
            t.str_adapter().validate_python(bad)

    def test_binary_content_adapter_accepts_bytes(self) -> None:
        """binary_content_adapter validates raw bytes payloads."""
        tm.that(t.binary_content_adapter().validate_python(b"abc"), eq=b"abc")

    def test_string_set_adapter_deduplicates(self) -> None:
        """string_set_adapter produces a de-duplicated set of strings."""
        tm.that(t.string_set_adapter().validate_python(["a", "a", "b"]), eq={"a", "b"})

    def test_scalar_mapping_adapter_accepts_mixed_scalars(self) -> None:
        """scalar_mapping_adapter validates str-keyed scalar mappings."""
        data = {"a": 1, "b": "x"}
        tm.that(t.scalar_mapping_adapter().validate_python(data), eq=data)

    def test_json_list_adapter_accepts_heterogeneous_items(self) -> None:
        """json_list_adapter validates lists of heterogeneous JSON values."""
        data: t.JsonList = [1, "a", None]
        tm.that(t.json_list_adapter().validate_python(data), eq=data)

    @pytest.mark.parametrize(("valid", "invalid"), [(1, 0), (8080, 70000)])
    def test_port_number_adapter_enforces_range(self, valid: int, invalid: int) -> None:
        """port_number_adapter accepts in-range ports and rejects out-of-range."""
        tm.that(t.port_number_adapter().validate_python(valid), eq=valid)
        with pytest.raises(c.ValidationError):
            t.port_number_adapter().validate_python(invalid)

    def test_hostname_str_adapter_rejects_empty(self) -> None:
        """hostname_str_adapter accepts a hostname and rejects the empty string."""
        tm.that(t.hostname_str_adapter().validate_python("localhost"), eq="localhost")
        with pytest.raises(c.ValidationError):
            t.hostname_str_adapter().validate_python("")

    def test_enum_type_adapter_accepts_strenum_class(self) -> None:
        """enum_type_adapter validates StrEnum subclasses as type values."""
        tm.that(t.enum_type_adapter().validate_python(_Color) is _Color, eq=True)

    def test_str_sequence_adapter_preserves_order(self) -> None:
        """str_sequence_adapter validates ordered string sequences."""
        tm.that(t.str_sequence_adapter().validate_python(["a", "b"]), eq=["a", "b"])

    def test_bool_adapter_accepts_bool(self) -> None:
        """bool_adapter validates boolean values."""
        tm.that(t.bool_adapter().validate_python(True), eq=True)

    def test_tiered_scalar_hierarchy_is_a_subset_chain(self) -> None:
        """Primitives subset Scalar subset Container runtime-type invariant holds."""
        primitives = set(t.PRIMITIVES_TYPES)
        scalar = set(t.SCALAR_TYPES)
        container = set(t.CONTAINER_TYPES)
        tm.that(primitives <= scalar, eq=True)
        tm.that(scalar <= container, eq=True)
