"""Tests for FlextTypes - dict mixin operations and validator map.

Module: flext_core.typings
Coverage target: lines 320, 324, 331, 351, 355, 359, 363, 367, 371, 457, 470, 476

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pydantic import BaseModel
from flext_core import t


class TestDictMixinOperations:
    """Tests for _DictMixin operations via t.Dict, t.ConfigMap, etc."""

    def test_setitem(self) -> None:
        """__setitem__ works on Dict (line 320)."""
        d = t.Dict(root={"a": "1"})
        d["b"] = "2"
        assert d["b"] == "2"

    def test_delitem(self) -> None:
        """__delitem__ works on Dict (line 324)."""
        d = t.Dict(root={"a": "1", "b": "2"})
        del d["a"]
        assert "a" not in d

    def test_iter(self) -> None:
        """__iter__ returns iterator over keys (line 331)."""
        d = t.Dict(root={"x": "1", "y": "2"})
        keys = list(d.keys())
        assert "x" in keys
        assert "y" in keys

    def test_values(self) -> None:
        """values() returns values view (line 351)."""
        d = t.Dict(root={"a": "1", "b": "2"})
        vals = list(d.values())
        assert "1" in vals
        assert "2" in vals

    def test_update(self) -> None:
        """update() merges another mapping (line 355)."""
        d = t.Dict(root={"a": "1"})
        d.update({"b": "2"})
        assert d["b"] == "2"

    def test_clear(self) -> None:
        """clear() empties the dict (line 359)."""
        d = t.Dict(root={"a": "1"})
        d.clear()
        assert len(d) == 0

    def test_pop(self) -> None:
        """pop() removes and returns item (line 363)."""
        d = t.Dict(root={"a": "1", "b": "2"})
        result = d.pop("a")
        assert result == "1"
        assert "a" not in d

    def test_pop_with_default(self) -> None:
        """pop() returns default when key missing."""
        d = t.Dict(root={"a": "1"})
        result = d.pop("missing", "default_val")
        assert result == "default_val"

    def test_popitem_and_setdefault(self) -> None:
        d = t.Dict(root={"a": "1"})
        key, value = d.popitem()
        assert key == "a"
        assert value == "1"

        d2 = t.Dict(root={})
        assert d2.setdefault("x", "y") == "y"
        assert d2["x"] == "y"


class TestConfigMapDictOps:
    """Test dict mixin ops on ConfigMap specifically."""

    def test_configmap_setitem(self) -> None:
        """ConfigMap supports __setitem__."""
        cm = t.ConfigMap(root={"key": "val"})
        cm["new"] = "item"
        assert cm["new"] == "item"

    def test_configmap_iter(self) -> None:
        """ConfigMap supports __iter__."""
        cm = t.ConfigMap(root={"a": "1", "b": "2"})
        assert set(cm.keys()) == {"a", "b"}


class TestValidatorCallable:
    """Tests for t.ValidatorCallable."""

    def test_validator_callable_invocation(self) -> None:
        """ValidatorCallable can be called (line 457)."""

        def upper(v: t.ScalarValue | BaseModel) -> t.ScalarValue | BaseModel:
            return str(v).upper() if isinstance(v, str) else v

        vc = t.ValidatorCallable(root=upper)
        assert vc("hello") == "HELLO"


class TestValidatorMapMixin:
    """Tests for _ValidatorMapMixin operations via FieldValidatorMap."""

    def test_items(self) -> None:
        """items() returns validator items (line 470)."""

        def noop(v: t.ScalarValue | BaseModel) -> t.ScalarValue | BaseModel:
            return v

        fvm = t.FieldValidatorMap(root={"field1": noop})
        items = list(fvm.items())
        assert len(items) == 1
        assert items[0][0] == "field1"

    def test_values(self) -> None:
        """values() returns validator values (line 476)."""

        def noop(v: t.ScalarValue | BaseModel) -> t.ScalarValue | BaseModel:
            return v

        fvm = t.FieldValidatorMap(root={"field1": noop})
        vals = list(fvm.values())
        assert len(vals) == 1
        assert vals[0] is noop
