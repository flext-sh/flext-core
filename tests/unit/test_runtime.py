"""Runtime facade tests for current contracts."""

from __future__ import annotations

from tests import m, t, u


class TestsFlextRuntime:
    """Public runtime utility checks."""

    class ModelDumpCarrier:
        """Simple HasModelDump carrier for protocol-bound runtime tests."""

        def __init__(self, payload: t.ScalarMapping) -> None:
            self._payload = payload

        def model_dump(self) -> t.ScalarMapping:
            return self._payload

    def test_dict_like_detects_mapping_inputs(self) -> None:
        payload: t.JsonPayload = {"a": 1}
        assert u.dict_like(payload)

    def test_dict_like_rejects_scalar_inputs(self) -> None:
        payload: t.JsonPayload = "value"
        assert not u.dict_like(payload)

    def test_normalize_to_container_returns_flat_scalar(self) -> None:
        payload: t.JsonPayload = 42
        normalized = u.normalize_to_container(payload)
        assert isinstance(normalized, int)
        assert normalized == 42

    def test_normalize_model_input_mapping_accepts_model_dump_carrier(self) -> None:
        carrier = self.ModelDumpCarrier({"alpha": 1, "beta": "two"})

        normalized = u.normalize_model_input_mapping(carrier)

        assert normalized == {"alpha": 1, "beta": "two"}

    def test_normalize_model_input_mapping_accepts_nested_payload_mapping(self) -> None:
        normalized = u.normalize_model_input_mapping({"alpha": {"beta": 1}})

        assert normalized == {"alpha": {"beta": 1}}

    def test_normalize_metadata_input_mapping_preserves_explicit_none(self) -> None:
        normalized = u.normalize_metadata_input_mapping({"alpha": None, "beta": 2})

        assert normalized == {"alpha": None, "beta": 2}

    def test_validate_metadata_attributes_accepts_model_dump_carrier(self) -> None:
        carrier = self.ModelDumpCarrier({"alpha": 1})

        normalized = u.validate_metadata_attributes(carrier)

        assert normalized == {"alpha": 1}

    def test_validate_metadata_model_input_accepts_model_dump_carrier(self) -> None:
        carrier = self.ModelDumpCarrier({"alpha": 1})

        metadata = u.validate_metadata_model_input(carrier, m.Metadata)

        assert metadata.attributes["alpha"] == 1
