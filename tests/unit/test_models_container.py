"""Behavioral tests for the RootModel container models.

Exercises the PUBLIC contract of ``m.ConfigMap`` (dict-rooted mapping API)
and ``m.ObjectList`` (list-rooted sequence API): construction/validation,
item access, membership, sizing, mutation, and ``model_dump`` round-trips.
No private attributes, no patched internals — only the observable surface a
caller depends on.
"""

from __future__ import annotations

import pytest

from tests.models import m


class TestsFlextCoreModelsContainer:
    """Public-contract behavior of the container RootModels."""

    # ------------------------------------------------------------------ #
    # ConfigMap — construction & validation
    # ------------------------------------------------------------------ #

    def test_config_map_validates_from_dict(self) -> None:
        # Arrange / Act
        cfg = m.ConfigMap(root={"a": 1, "b": "two"})
        # Assert
        assert cfg.root == {"a": 1, "b": "two"}

    def test_config_map_model_validate_accepts_mapping(self) -> None:
        cfg = m.ConfigMap.model_validate({"x": True})
        assert cfg["x"] is True

    def test_config_map_rejects_non_mapping_root(self) -> None:
        # Pydantic ValidationError subclasses ValueError.
        with pytest.raises(ValueError):
            m.ConfigMap.model_validate(["not", "a", "mapping"])

    # ------------------------------------------------------------------ #
    # ConfigMap — read API
    # ------------------------------------------------------------------ #

    def test_config_map_getitem_returns_value(self) -> None:
        cfg = m.ConfigMap(root={"a": 1})
        assert cfg["a"] == 1

    def test_config_map_getitem_missing_key_raises(self) -> None:
        cfg = m.ConfigMap(root={"a": 1})
        with pytest.raises(KeyError):
            _ = cfg["absent"]

    @pytest.mark.parametrize(
        ("key", "default", "expected"),
        [
            ("a", None, 1),
            ("absent", None, None),
            ("absent", 99, 99),
        ],
    )
    def test_config_map_get_returns_value_or_default(
        self,
        key: str,
        default: int | None,
        expected: int | None,
    ) -> None:
        cfg = m.ConfigMap(root={"a": 1})
        assert cfg.get(key, default) == expected

    @pytest.mark.parametrize(
        ("key", "present"),
        [("a", True), ("b", True), ("missing", False)],
    )
    def test_config_map_contains_reflects_membership(
        self,
        key: str,
        present: bool,
    ) -> None:
        cfg = m.ConfigMap(root={"a": 1, "b": 2})
        assert (key in cfg) is present

    def test_config_map_len_counts_entries(self) -> None:
        assert len(m.ConfigMap(root={"a": 1, "b": 2, "c": 3})) == 3

    @pytest.mark.parametrize(
        ("root", "truthy"),
        [({}, False), ({"a": 1}, True)],
    )
    def test_config_map_bool_reflects_emptiness(
        self,
        root: dict[str, int],
        truthy: bool,
    ) -> None:
        assert bool(m.ConfigMap(root=root)) is truthy

    def test_config_map_keys_values_items_expose_contents(self) -> None:
        cfg = m.ConfigMap(root={"a": 1, "b": 2})
        assert set(cfg.keys()) == {"a", "b"}
        assert set(cfg.values()) == {1, 2}
        assert dict(cfg.items()) == {"a": 1, "b": 2}

    # ------------------------------------------------------------------ #
    # ConfigMap — mutation API
    # ------------------------------------------------------------------ #

    def test_config_map_setitem_adds_entry(self) -> None:
        cfg = m.ConfigMap(root={"a": 1})
        cfg["b"] = 2
        assert cfg["b"] == 2
        assert len(cfg) == 2

    def test_config_map_delitem_removes_entry(self) -> None:
        cfg = m.ConfigMap(root={"a": 1, "b": 2})
        del cfg["a"]
        assert "a" not in cfg
        assert len(cfg) == 1

    def test_config_map_pop_returns_and_removes(self) -> None:
        cfg = m.ConfigMap(root={"a": 1, "b": 2})
        assert cfg.pop("a") == 1
        assert "a" not in cfg

    def test_config_map_pop_missing_returns_default(self) -> None:
        cfg = m.ConfigMap(root={"a": 1})
        assert cfg.pop("absent", 7) == 7

    def test_config_map_popitem_removes_a_pair(self) -> None:
        cfg = m.ConfigMap(root={"a": 1})
        key, value = cfg.popitem()
        assert (key, value) == ("a", 1)
        assert len(cfg) == 0

    def test_config_map_setdefault_inserts_when_absent(self) -> None:
        cfg = m.ConfigMap(root={"a": 1})
        assert cfg.setdefault("b", 5) == 5
        assert cfg["b"] == 5

    def test_config_map_setdefault_keeps_existing(self) -> None:
        cfg = m.ConfigMap(root={"a": 1})
        assert cfg.setdefault("a", 99) == 1
        assert cfg["a"] == 1

    def test_config_map_update_merges_entries(self) -> None:
        cfg = m.ConfigMap(root={"a": 1})
        cfg.update({"a": 10, "b": 2})
        assert cfg["a"] == 10
        assert cfg["b"] == 2

    def test_config_map_clear_empties(self) -> None:
        cfg = m.ConfigMap(root={"a": 1, "b": 2})
        cfg.clear()
        assert len(cfg) == 0
        assert not cfg

    # ------------------------------------------------------------------ #
    # ConfigMap — serialization
    # ------------------------------------------------------------------ #

    def test_config_map_model_dump_returns_plain_mapping(self) -> None:
        payload = {"a": 1, "b": "two"}
        assert m.ConfigMap(root=payload).model_dump() == payload

    def test_config_map_round_trips_through_model_dump(self) -> None:
        original = m.ConfigMap(root={"a": 1, "nested": {"k": "v"}})
        restored = m.ConfigMap.model_validate(original.model_dump())
        assert restored.model_dump() == original.model_dump()

    # ------------------------------------------------------------------ #
    # ObjectList
    # ------------------------------------------------------------------ #

    def test_object_list_preserves_order_and_values(self) -> None:
        values = m.ObjectList(root=["a", 1, True])
        assert values.root == ["a", 1, True]

    def test_object_list_len_counts_elements(self) -> None:
        assert len(m.ObjectList(root=["a", 1])) == 2

    @pytest.mark.parametrize(
        ("root", "truthy"),
        [([], False), (["a"], True)],
    )
    def test_object_list_bool_reflects_emptiness(
        self,
        root: list[str],
        truthy: bool,
    ) -> None:
        assert bool(m.ObjectList(root=root)) is truthy

    def test_object_list_model_dump_returns_plain_list(self) -> None:
        payload = ["a", 1, "b"]
        assert m.ObjectList(root=payload).model_dump() == payload

    def test_object_list_rejects_non_sequence_root(self) -> None:
        with pytest.raises(ValueError):
            m.ObjectList.model_validate({"root": 12345})
