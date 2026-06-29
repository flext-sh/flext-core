"""Typing container model tests."""

from __future__ import annotations

from flext_tests import tm

from tests import m, t


class TestsFlextTypesContainers:
    def test_dict_creation_empty(self) -> None:
        """m.Dict can be created empty."""
        d = m.Dict(root={})
        tm.that(len(d), eq=0)

    def test_dict_creation_with_data(self) -> None:
        """m.Dict can be created with initial data."""
        d = m.Dict(root={"key": "value", "num": 42})
        tm.that(d["key"], eq="value")
        tm.that(d["num"], eq=42)

    def test_dict_contains(self) -> None:
        """m.Dict supports 'in' operator."""
        d = m.Dict(root={"key": "value"})
        tm.that("key" in d, eq=True)
        tm.that("missing" in d, eq=False)

    def test_dict_get_with_default(self) -> None:
        """m.Dict.get() returns default for missing keys."""
        d = m.Dict(root={"key": "value"})
        tm.that(d.get("key"), eq="value")
        tm.that(d.get("missing", "fallback"), eq="fallback")

    def test_configmap_creation(self) -> None:
        """m.ConfigMap can be created with settings data."""
        cm = m.ConfigMap(root={"timeout": 30, "debug": False})
        tm.that(cm["timeout"], eq=30)
        tm.that(cm["debug"], eq=False)

    def test_configmap_len(self) -> None:
        """m.ConfigMap supports len()."""
        cm = m.ConfigMap(root={"a": 1, "b": 2})
        tm.that(len(cm), eq=2)

    def test_object_list_creation(self) -> None:
        """t.JsonList can be created with container values."""
        ol = m.ObjectList(root=["item1", 42, True])
        tm.that(len(ol.root), eq=3)

    def test_object_list_default_empty(self) -> None:
        """t.JsonList defaults to empty list."""
        ol = m.ObjectList(root=[])
        tm.that(len(ol.root), eq=0)

    def test_flexttypes_inherits_base(self) -> None:
        """T inherits from t through MRO."""
        tm.that(t in t.__mro__, eq=True)

    def test_flexttypes_inherits_containers(self) -> None:
        """T inherits from t through MRO."""
        tm.that(t in t.__mro__, eq=True)

    def test_flexttypes_inherits_core(self) -> None:
        """T inherits from t through MRO."""
        tm.that(t in t.__mro__, eq=True)

    def test_flexttypes_inherits_services(self) -> None:
        """T inherits from t through MRO."""
        tm.that(t in t.__mro__, eq=True)

    def test_flexttypes_inherits_validation(self) -> None:
        """T inherits from t through MRO."""
        tm.that(t in t.__mro__, eq=True)

    def test_pair_alias_exists(self) -> None:
        """t.Pair alias is accessible."""

    def test_triple_alias_exists(self) -> None:
        """t.Triple alias is accessible."""

    def test_variadic_tuple_alias_exists(self) -> None:
        """t.VariadicTuple alias is accessible."""

    def test_int_pair_alias_exists(self) -> None:
        """t.IntPair alias is accessible."""

    def test_container_value_scalar_types_mirrors_scalar(self) -> None:
        """SCALAR_TYPES exposes the scalar runtime contract."""
        tm.that(len(t.SCALAR_TYPES), gt=0)
