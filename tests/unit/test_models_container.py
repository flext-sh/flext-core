"""Behavioral tests for the RootModel container models.

Exercises the PUBLIC contract of ``m.ConfigMap`` (dict-rooted mapping API)
and ``m.ObjectList`` (list-rooted sequence API): construction/validation,
item access, membership, sizing, mutation, and ``model_dump`` round-trips.
No private attributes, no patched internals — only the observable surface a
caller depends on.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import pytest

from tests import m, t


class TestsFlextCoreModelsContainer:
    """Public-contract behavior of the container RootModels."""

    # ------------------------------------------------------------------ #
    # ConfigMap — construction & validation
    # ------------------------------------------------------------------ #

    @staticmethod
    def test_config_map_validates_from_dict() -> None:
        # Arrange / Act
        """Test config map validates from dict."""
        cfg = m.ConfigMap(root={"a": 1, "b": "two"})
        # Assert
        assert cfg.root == {"a": 1, "b": "two"}

    @staticmethod
    def test_config_map_model_validate_accepts_mapping() -> None:
        """Test config map model validate accepts mapping."""
        cfg = m.ConfigMap.model_validate({"x": True})
        assert cfg["x"] is True

    @staticmethod
    def test_config_map_rejects_non_mapping_root() -> None:
        # Pydantic ValidationError subclasses ValueError.
        """Test config map rejects non mapping root."""
        with pytest.raises(m.ValidationError):
            m.ConfigMap.model_validate(["not", "a", "mapping"])

    # ------------------------------------------------------------------ #
    # ConfigMap — read API
    # ------------------------------------------------------------------ #

    @staticmethod
    def test_config_map_getitem_returns_value() -> None:
        """Test config map getitem returns value."""
        cfg = m.ConfigMap(root={"a": 1})
        assert cfg["a"] == 1

    @staticmethod
    def test_config_map_getitem_missing_key_raises() -> None:
        """Test config map getitem missing key raises."""
        cfg = m.ConfigMap(root={"a": 1})
        with pytest.raises(KeyError):
            _ = cfg["absent"]

    @pytest.mark.parametrize(
        ("key", "default", "expected"),
        [("a", None, 1), ("absent", None, None), ("absent", 99, 99)],
    )
    @staticmethod
    def test_config_map_get_returns_value_or_default(
        key: str,
        default: int | None,
        expected: int | None,
    ) -> None:
        """Test config map get returns value or default."""
        cfg = m.ConfigMap(root={"a": 1})
        assert cfg.get(key, default) == expected

    @pytest.mark.parametrize(
        ("key", "present"),
        [("a", True), ("b", True), ("missing", False)],
    )
    @staticmethod
    def test_config_map_contains_reflects_membership(
        key: str,
        *,
        present: bool,
    ) -> None:
        """Test config map contains reflects membership."""
        cfg = m.ConfigMap(root={"a": 1, "b": 2})
        assert (key in cfg) is present

    @staticmethod
    def test_config_map_len_counts_entries() -> None:
        """Test config map len counts entries."""
        assert len(m.ConfigMap(root={"a": 1, "b": 2, "c": 3})) == 3

    @pytest.mark.parametrize(("root", "truthy"), [({}, False), ({"a": 1}, True)])
    @staticmethod
    def test_config_map_bool_reflects_emptiness(
        root: dict[str, t.JsonPayload],
        *,
        truthy: bool,
    ) -> None:
        """Test config map bool reflects emptiness."""
        assert bool(m.ConfigMap(root=root)) is truthy

    @staticmethod
    def test_config_map_keys_values_items_expose_contents() -> None:
        """Test config map keys values items expose contents."""
        cfg = m.ConfigMap(root={"a": 1, "b": 2})
        assert set(cfg.keys()) == {"a", "b"}
        assert set(cfg.values()) == {1, 2}
        assert dict(cfg.items()) == {"a": 1, "b": 2}

    # ------------------------------------------------------------------ #
    # ConfigMap — mutation API
    # ------------------------------------------------------------------ #

    @staticmethod
    def test_config_map_setitem_adds_entry() -> None:
        """Test config map setitem adds entry."""
        cfg = m.ConfigMap(root={"a": 1})
        cfg["b"] = 2
        assert cfg["b"] == 2
        assert len(cfg) == 2

    @staticmethod
    def test_config_map_delitem_removes_entry() -> None:
        """Test config map delitem removes entry."""
        cfg = m.ConfigMap(root={"a": 1, "b": 2})
        del cfg["a"]
        assert "a" not in cfg
        assert len(cfg) == 1

    @staticmethod
    def test_config_map_pop_returns_and_removes() -> None:
        """Test config map pop returns and removes."""
        cfg = m.ConfigMap(root={"a": 1, "b": 2})
        assert cfg.pop("a") == 1
        assert "a" not in cfg

    @staticmethod
    def test_config_map_pop_missing_returns_default() -> None:
        """Test config map pop missing returns default."""
        cfg = m.ConfigMap(root={"a": 1})
        assert cfg.pop("absent", 7) == 7

    @staticmethod
    def test_config_map_popitem_removes_a_pair() -> None:
        """Test config map popitem removes a pair."""
        cfg = m.ConfigMap(root={"a": 1})
        key, value = cfg.popitem()
        assert (key, value) == ("a", 1)
        assert len(cfg) == 0

    @staticmethod
    def test_config_map_setdefault_inserts_when_absent() -> None:
        """Test config map setdefault inserts when absent."""
        cfg = m.ConfigMap(root={"a": 1})
        assert cfg.setdefault("b", 5) == 5
        assert cfg["b"] == 5

    @staticmethod
    def test_config_map_setdefault_keeps_existing() -> None:
        """Test config map setdefault keeps existing."""
        cfg = m.ConfigMap(root={"a": 1})
        assert cfg.setdefault("a", 99) == 1
        assert cfg["a"] == 1

    @staticmethod
    def test_config_map_update_merges_entries() -> None:
        """Test config map update merges entries."""
        cfg = m.ConfigMap(root={"a": 1})
        cfg.update({"a": 10, "b": 2})
        assert cfg["a"] == 10
        assert cfg["b"] == 2

    @staticmethod
    def test_config_map_clear_empties() -> None:
        """Test config map clear empties."""
        cfg = m.ConfigMap(root={"a": 1, "b": 2})
        cfg.clear()
        assert len(cfg) == 0
        assert not cfg

    # ------------------------------------------------------------------ #
    # ConfigMap — serialization
    # ------------------------------------------------------------------ #

    @staticmethod
    def test_config_map_model_dump_returns_plain_mapping() -> None:
        """Test config map model dump returns plain mapping."""
        payload: dict[str, t.JsonPayload] = {"a": 1, "b": "two"}
        assert m.ConfigMap(root=payload).model_dump() == payload

    @staticmethod
    def test_config_map_round_trips_through_model_dump() -> None:
        """Test config map round trips through model dump."""
        original = m.ConfigMap(root={"a": 1, "nested": {"k": "v"}})
        restored = m.ConfigMap.model_validate(original.model_dump())
        assert restored.model_dump() == original.model_dump()

    # ------------------------------------------------------------------ #
    # ObjectList
    # ------------------------------------------------------------------ #

    @staticmethod
    def test_object_list_preserves_order_and_values() -> None:
        """Test object list preserves order and values."""
        values = m.ObjectList(root=["a", 1, True])
        assert values.root == ["a", 1, True]

    @staticmethod
    def test_object_list_len_counts_elements() -> None:
        """Test object list len counts elements."""
        assert len(m.ObjectList(root=["a", 1])) == 2

    @pytest.mark.parametrize(("root", "truthy"), [([], False), (["a"], True)])
    @staticmethod
    def test_object_list_bool_reflects_emptiness(
        root: list[t.JsonPayload],
        *,
        truthy: bool,
    ) -> None:
        """Test object list bool reflects emptiness."""
        assert bool(m.ObjectList(root=root)) is truthy

    @staticmethod
    def test_object_list_model_dump_returns_plain_list() -> None:
        """Test object list model dump returns plain list."""
        payload: list[t.JsonPayload] = ["a", 1, "b"]
        assert m.ObjectList(root=payload).model_dump() == payload

    @staticmethod
    def test_object_list_rejects_non_sequence_root() -> None:
        """Test object list rejects non sequence root."""
        with pytest.raises(m.ValidationError):
            m.ObjectList.model_validate({"root": 12345})
