"""Behavioral tests for base model mixins and value models.

Asserts the observable public contract of the FlextModelsBase family
(Metadata, ContractModel, FrozenValueModel, IdentifiableMixin,
TimestampableMixin, TimestampedModel) through the public API only:
constructed model state, model_dump output, and validation error paths.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Annotated

import pytest

from tests.constants import c
from tests.models import m


class TestsFlextCoreModelsBaseFullCoverage:
    class _FrozenValue(m.FrozenValueModel):
        name: Annotated[str, m.Field(description="Frozen value name")]
        count: Annotated[int, m.Field(description="Frozen value count")]

        def __hash__(self) -> int:
            return m.FrozenValueModel.__hash__(self)

    class _Identifiable(m.FlexibleModel, m.IdentifiableMixin):
        pass

    class _Timestampable(m.FlexibleModel, m.TimestampableMixin):
        pass

    # --- Metadata.attributes coercion contract ---------------------------

    def test_metadata_attributes_none_coerces_to_empty_mapping(self) -> None:
        model = m.Metadata.model_validate({"attributes": None})
        assert model.attributes == {}

    def test_metadata_attributes_accepts_plain_mapping(self) -> None:
        model = m.Metadata.model_validate({"attributes": {"key": "value"}})
        assert model.attributes == {"key": "value"}

    def test_metadata_attributes_accepts_t_dict_and_kwargs_mapping(self) -> None:
        from_root = m.Metadata.model_validate({"attributes": m.Dict(root={"a": 1})})
        from_kwargs = m.Metadata(attributes={"b": 2})
        assert from_root.attributes == {"a": 1}
        assert from_kwargs.attributes == {"b": 2}

    @pytest.mark.parametrize("bad_value", [123, "text", 4.5, ["a", "b"]])
    def test_metadata_attributes_non_mapping_rejected(self, bad_value: object) -> None:
        with pytest.raises(TypeError, match="attributes must be dict-like"):
            m.Metadata.model_validate({"attributes": bad_value})

    def test_metadata_attributes_broken_dump_object_rejected(self) -> None:
        with pytest.raises(TypeError):
            m.Metadata.model_validate({"attributes": m.Tests._BrokenDumpModel()})

    # --- Metadata defaults and immutability ------------------------------

    def test_metadata_defaults_are_populated(self) -> None:
        model = m.Metadata()
        assert model.created_at.tzinfo == UTC
        assert model.version == "1.0.0"
        assert model.tags == ()
        assert model.created_by is None
        assert model.attributes == {}

    def test_metadata_is_frozen(self) -> None:
        model = m.Metadata()
        with pytest.raises(c.ValidationError):
            setattr(model, "version", "2.0.0")

    def test_metadata_rejects_extra_fields(self) -> None:
        with pytest.raises(c.ValidationError):
            m.Metadata.model_validate({"unknown_field": 1})

    # --- FrozenValueModel value semantics --------------------------------

    def test_frozen_value_equal_by_value(self) -> None:
        left = self._FrozenValue(name="item", count=1)
        right = self._FrozenValue(name="item", count=1)
        assert left == right
        assert hash(left) == hash(right)

    def test_frozen_value_differs_when_fields_differ(self) -> None:
        left = self._FrozenValue(name="item", count=1)
        other = self._FrozenValue(name="item", count=2)
        assert left != other

    def test_frozen_value_not_equal_to_foreign_type(self) -> None:
        assert self._FrozenValue(name="item", count=1) != "item"

    def test_frozen_value_is_hashable_in_set(self) -> None:
        a = self._FrozenValue(name="x", count=1)
        b = self._FrozenValue(name="x", count=1)
        d = self._FrozenValue(name="y", count=2)
        assert len({a, b, d}) == 2

    # --- IdentifiableMixin -----------------------------------------------

    def test_identifiable_autogenerates_unique_id(self) -> None:
        first = self._Identifiable()
        second = self._Identifiable()
        assert first.unique_id != second.unique_id
        assert len(first.unique_id) == 36

    def test_identifiable_blank_unique_id_rejected(self) -> None:
        with pytest.raises(
            c.ValidationError,
            match="String should have at least 1 character",
        ):
            self._Identifiable(unique_id="   ")

    # --- TimestampableMixin ----------------------------------------------

    def test_timestampable_normalizes_to_utc(self) -> None:
        moment = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
        model = self._Timestampable(created_at=moment, updated_at=moment)
        assert model.created_at.tzinfo == UTC

    def test_timestampable_serializes_timestamps_as_iso_strings(self) -> None:
        moment = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
        model = self._Timestampable(created_at=moment)
        dumped = model.model_dump(mode="json")
        assert dumped["created_at"] == moment.isoformat()
        assert dumped["updated_at"] is None

    def test_timestampable_rejects_updated_before_created(self) -> None:
        created = datetime(2026, 1, 2, 12, 0, 0, tzinfo=UTC)
        earlier = created - timedelta(days=1)
        with pytest.raises(c.ValidationError):
            self._Timestampable(created_at=created, updated_at=earlier)

    # --- TimestampedModel -------------------------------------------------

    def test_timestamped_model_defaults_to_utc_now(self) -> None:
        model = m.TimestampedModel()
        assert model.created_at.tzinfo == UTC
        assert model.updated_at is None
