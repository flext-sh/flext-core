"""Container model smoke tests."""

from __future__ import annotations

from tests import m


class TestModelsContainer:
    def test_config_map_root_mapping(self) -> None:
        cfg = m.ConfigMap(root={"a": 1})
        assert cfg["a"] == 1

    def test_object_list_root_sequence(self) -> None:
        values = m.ObjectList(root=["a", 1])
        assert len(values.root) == 2
