"""Generic model smoke tests for current model facade."""

from __future__ import annotations

from tests import m


class TestModelsGenericFullCoverage:
    def test_config_map_generic_usage(self) -> None:
        cfg = m.ConfigMap(root={"x": 1})
        assert cfg["x"] == 1

    def test_object_list_generic_usage(self) -> None:
        values = m.ObjectList(root=["a", 1, True])
        assert len(values.root) == 3
