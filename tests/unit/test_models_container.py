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

from tests import m, t, tm


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
        tm.that(cfg.root, eq={"a": 1, "b": "two"})

    @staticmethod
    def test_config_map_model_validate_accepts_mapping() -> None:
        """Test config map model validate accepts mapping."""
        cfg = m.ConfigMap.model_validate({"x": True})
        tm.that(cfg["x"] is True, truth=True)

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
        tm.that(cfg["a"], eq=1)

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
        tm.that(cfg.get(key, default), eq=expected)

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
        tm.that((key in cfg) is present, truth=True)

    @staticmethod
    def test_config_map_len_counts_entries() -> None:
        """Test config map len counts entries."""
        tm.that(len(m.ConfigMap(root={"a": 1, "b": 2, "c": 3})), eq=3)

    @pytest.mark.parametrize(("root", "truthy"), [({}, False), ({"a": 1}, True)])
    @staticmethod
    def test_config_map_bool_reflects_emptiness(
        root: t.UserOverridesMapping,
        *,
        truthy: bool,
    ) -> None:
        """Test config map bool reflects emptiness."""
        tm.that(bool(m.ConfigMap(root=dict(root))) is truthy, truth=True)

    @staticmethod
    def test_config_map_keys_values_items_expose_contents() -> None:
        """Test config map keys values items expose contents."""
        cfg = m.ConfigMap(root={"a": 1, "b": 2})
        tm.that(set(cfg.keys()), eq={"a", "b"})
        tm.that(set(cfg.values()), eq={1, 2})
        tm.that(dict(cfg.items()), eq={"a": 1, "b": 2})

    # ------------------------------------------------------------------ #
    # ConfigMap — mutation API
    # ------------------------------------------------------------------ #

    @staticmethod
    def test_config_map_setitem_adds_entry() -> None:
        """Test config map setitem adds entry."""
        cfg = m.ConfigMap(root={"a": 1})
        cfg["b"] = 2
        tm.that(cfg["b"], eq=2)
        tm.that(len(cfg), eq=2)

    @staticmethod
    def test_config_map_delitem_removes_entry() -> None:
        """Test config map delitem removes entry."""
        cfg = m.ConfigMap(root={"a": 1, "b": 2})
        del cfg["a"]
        tm.that("a" not in cfg, truth=True)
        tm.that(len(cfg), eq=1)

    @staticmethod
    def test_config_map_pop_returns_and_removes() -> None:
        """Test config map pop returns and removes."""
        cfg = m.ConfigMap(root={"a": 1, "b": 2})
        tm.that(cfg.pop("a"), eq=1)
        tm.that("a" not in cfg, truth=True)

    @staticmethod
    def test_config_map_pop_missing_returns_default() -> None:
        """Test config map pop missing returns default."""
        cfg = m.ConfigMap(root={"a": 1})
        tm.that(cfg.pop("absent", 7), eq=7)

    @staticmethod
    def test_config_map_popitem_removes_a_pair() -> None:
        """Test config map popitem removes a pair."""
        cfg = m.ConfigMap(root={"a": 1})
        key, value = cfg.popitem()
        tm.that((key, value), eq=("a", 1))
        tm.that(len(cfg), eq=0)

    @staticmethod
    def test_config_map_setdefault_inserts_when_absent() -> None:
        """Test config map setdefault inserts when absent."""
        cfg = m.ConfigMap(root={"a": 1})
        tm.that(cfg.setdefault("b", 5), eq=5)
        tm.that(cfg["b"], eq=5)

    @staticmethod
    def test_config_map_setdefault_keeps_existing() -> None:
        """Test config map setdefault keeps existing."""
        cfg = m.ConfigMap(root={"a": 1})
        tm.that(cfg.setdefault("a", 99), eq=1)
        tm.that(cfg["a"], eq=1)

    @staticmethod
    def test_config_map_update_merges_entries() -> None:
        """Test config map update merges entries."""
        cfg = m.ConfigMap(root={"a": 1})
        cfg.update({"a": 10, "b": 2})
        tm.that(cfg["a"], eq=10)
        tm.that(cfg["b"], eq=2)

    @staticmethod
    def test_config_map_clear_empties() -> None:
        """Test config map clear empties."""
        cfg = m.ConfigMap(root={"a": 1, "b": 2})
        cfg.clear()
        tm.that(len(cfg), eq=0)
        tm.that(cfg, truth=False)

    # ------------------------------------------------------------------ #
    # ConfigMap — serialization
    # ------------------------------------------------------------------ #

    @staticmethod
    def test_config_map_model_dump_returns_plain_mapping() -> None:
        """Test config map model dump returns plain mapping."""
        payload: t.UserOverridesMapping = {"a": 1, "b": "two"}
        tm.that(m.ConfigMap(root=dict(payload)).model_dump(), eq=payload)

    @staticmethod
    def test_config_map_round_trips_through_model_dump() -> None:
        """Test config map round trips through model dump."""
        original = m.ConfigMap(root={"a": 1, "nested": {"k": "v"}})
        restored = m.ConfigMap.model_validate(original.model_dump())
        tm.that(restored.model_dump(), eq=original.model_dump())

    # ------------------------------------------------------------------ #
    # ObjectList
    # ------------------------------------------------------------------ #

    @staticmethod
    def test_object_list_preserves_order_and_values() -> None:
        """Test object list preserves order and values."""
        values = m.ObjectList(root=["a", 1, True])
        tm.that(values.root, eq=["a", 1, True])

    @staticmethod
    def test_object_list_len_counts_elements() -> None:
        """Test object list len counts elements."""
        tm.that(len(m.ObjectList(root=["a", 1])), eq=2)

    @pytest.mark.parametrize(("root", "truthy"), [([], False), (["a"], True)])
    @staticmethod
    def test_object_list_bool_reflects_emptiness(
        root: list[t.JsonPayload],
        *,
        truthy: bool,
    ) -> None:
        """Test object list bool reflects emptiness."""
        tm.that(bool(m.ObjectList(root=root)) is truthy, truth=True)

    @staticmethod
    def test_object_list_model_dump_returns_plain_list() -> None:
        """Test object list model dump returns plain list."""
        payload: list[t.JsonPayload] = ["a", 1, "b"]
        tm.that(m.ObjectList(root=payload).model_dump(), eq=payload)

    @staticmethod
    def test_object_list_rejects_non_sequence_root() -> None:
        """Test object list rejects non sequence root."""
        with pytest.raises(m.ValidationError):
            m.ObjectList.model_validate({"root": 12345})
