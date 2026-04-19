"""Focused model tests aligned to current public model facade."""

from __future__ import annotations

from tests import m


class TestsFlextCoreModelsUnit:
    """Smoke tests for stable model contracts."""

    def test_config_map_supports_mapping_access(self) -> None:
        data = m.ConfigMap(root={"a": 1, "b": "x"})
        assert data["a"] == 1
        assert len(data) == 2

    def test_dict_supports_mutation(self) -> None:
        payload = m.Dict(root={"name": "flext"})
        payload["version"] = "0.12.0"
        assert payload["version"] == "0.12.0"

    def test_value_model_validation_and_dump(self) -> None:
        class SampleValue(m.Value):
            name: str
            count: int

        value = SampleValue(name="ok", count=2)
        dumped = value.model_dump(mode="python")
        assert dumped["name"] == "ok"
        assert dumped["count"] == 2
