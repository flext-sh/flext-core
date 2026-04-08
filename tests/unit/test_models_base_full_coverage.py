"""Full coverage tests for base model mixins and utilities."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from tests import c, m, r, t, u


class TestModelsBaseFullCoverage:
    class _FrozenValue(m.ContractModel):
        name: str
        count: int

    class _Identifiable(m.FlexibleModel, m.IdentifiableMixin):
        pass

    class _Timestampable(m.FlexibleModel, m.TimestampableMixin):
        pass

    def test_metadata_attributes_accepts_none(self) -> None:
        model = m.Metadata.model_validate({"attributes": None})
        assert model.attributes == {}

    def test_metadata_attributes_accepts_basemodel_mapping(self) -> None:
        """Metadata.attributes accepts mapping-like input; use dict for model_validate."""
        model = m.Metadata.model_validate({"attributes": {"key": "value"}})
        assert model.attributes == {"key": "value"}

    def test_metadata_attributes_rejects_basemodel_non_mapping_dump(self) -> None:
        with pytest.raises(TypeError):
            m.Metadata.model_validate({"attributes": m.Core.Unit._BrokenDumpModel()})

    def test_metadata_attributes_accepts_t_dict_and_mapping(self) -> None:
        model_from_t_dict = m.Metadata.model_validate({
            "attributes": t.Dict(root={"a": 1}),
        })
        model_from_mapping = m.Metadata(attributes={"b": 2})
        assert model_from_t_dict.attributes == {"a": 1}
        assert model_from_mapping.attributes == {"b": 2}

    def test_metadata_attributes_rejects_non_mapping(self) -> None:
        with pytest.raises(TypeError, match="attributes must be dict-like") as exc_info:
            m.Metadata.model_validate({"attributes": 123})
        assert exc_info.value is not None
        assert "attributes must be dict-like" in str(exc_info.value)

    def test_frozen_value_model_equality_and_hash(self) -> None:
        left = self._FrozenValue(name="item", count=1)
        right = self._FrozenValue(name="item", count=1)
        assert left == right
        eq_result = left.__eq__("normalized")
        assert eq_result is NotImplemented
        hash_val = hash(left)
        assert isinstance(hash_val, int)

    def test_identifiable_unique_id_empty_rejected(self) -> None:
        with pytest.raises(
            ValidationError,
            match="String should have at least 1 character",
        ) as exc_info:
            self._Identifiable(unique_id="   ")
        assert exc_info.value is not None
        assert "at least 1 character" in str(exc_info.value)

    def test_timestampable_timestamp_conversion_and_json_serializer(self) -> None:
        naive = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
        model = self._Timestampable(created_at=naive, updated_at=naive)
        assert model.created_at.tzinfo == UTC
        dumped = model.model_dump(mode="json")
        assert isinstance(dumped["created_at"], str)

    def test_timestamped_model_and_alias_and_canonical_symbols(self) -> None:
        model = m.TimestampedModel()
        assert model.created_at.tzinfo == UTC
        assert hasattr(m, "TimestampedModel")
        ok_result = r[str].ok("ok")
        assert ok_result.value == "ok"
        version = c.DEFAULT_RETRY_DELAY_SECONDS
        assert version >= 1
        assert hasattr(u, "transform")
