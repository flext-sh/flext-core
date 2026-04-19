"""Base model smoke tests for internal model layer."""

from __future__ import annotations

from tests import m


class TestModelsBase:
    def test_base_model_validate_and_dump(self) -> None:
        class Sample(m.BaseModel):
            name: str

        model = Sample(name="base")
        dumped = model.model_dump(mode="python")
        assert dumped["name"] == "base"

    def test_value_model_validate(self) -> None:
        class SampleValue(m.Value):
            count: int

        value = SampleValue(count=1)
        assert value.count == 1
