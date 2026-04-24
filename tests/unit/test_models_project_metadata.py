"""Project metadata model smoke tests."""

from __future__ import annotations

from tests import m


class TestsFlextCoreModelsProjectMetadata:
    def test_metadata_instantiation(self) -> None:
        metadata = m.Metadata(attributes={"service": "flext"})
        dumped = metadata.model_dump(mode="python")
        assert "attributes" in dumped

    def test_metadata_attributes_content(self) -> None:
        metadata = m.Metadata(attributes={"k": "v"})
        assert metadata.attributes.get("k") == "v"
