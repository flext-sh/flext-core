"""Context-related model smoke tests for current model facade."""

from __future__ import annotations

from tests import m


class TestModelsContextFullCoverage:
    """Minimal behavioral tests for context model helpers."""

    def test_normalize_to_mapping_accepts_plain_mapping(self) -> None:
        source = {"a": 1, "b": "x"}
        assert m.normalize_to_mapping(source) == source

    def test_context_export_model_dump_contains_expected_fields(self) -> None:
        export = m.ContextExport(data={"k": "v"}, statistics={"sets": 1})
        dumped = export.model_dump(mode="python")
        assert "data" in dumped
        assert "statistics" in dumped
