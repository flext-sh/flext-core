"""Full coverage tests for base model mixins and utilities."""

from __future__ import annotations

# mypy: follow_imports=skip, disable-error-code=valid-type
# pyright: basic, reportMissingImports=false, reportImplicitOverride=false, reportUnknownVariableType=false, reportUnknownLambdaType=false, reportUnusedCallResult=false, reportPrivateUsage=false
from datetime import UTC, datetime

import pytest
from flext_core import c, m, r, t, u
from pydantic import BaseModel, ValidationError


class _FrozenValue(m.FrozenStrictModel):
    name: str
    count: int


class _Identifiable(m.IdentifiableMixin):
    pass


class _Timestampable(m.TimestampableMixin):
    pass


class _BrokenDumpModel(BaseModel):
    value: int = 1

    def __getattribute__(self, name: str) -> object:
        if name == "model_dump":
            return lambda *args, **kwargs: [1]
        return super().__getattribute__(name)


def test_metadata_attributes_accepts_none() -> None:
    model = m.Metadata.model_validate({"attributes": None})
    assert model.attributes == {}


def test_metadata_attributes_accepts_basemodel_mapping() -> None:
    class _Attrs(BaseModel):
        key: str

    model = m.Metadata.model_validate({"attributes": _Attrs(key="value")})
    assert model.attributes == {"key": "value"}


def test_metadata_attributes_rejects_basemodel_non_mapping_dump() -> None:
    with pytest.raises(TypeError, match="must dump to mapping"):
        m.Metadata.model_validate({"attributes": _BrokenDumpModel()})


def test_metadata_attributes_accepts_t_dict_and_mapping() -> None:
    model_from_t_dict = m.Metadata.model_validate({"attributes": t.Dict(root={"a": 1})})
    model_from_mapping = m.Metadata(attributes={"b": 2})
    assert model_from_t_dict.attributes == {"a": 1}
    assert model_from_mapping.attributes == {"b": 2}


def test_metadata_attributes_rejects_non_mapping() -> None:
    with pytest.raises(TypeError, match="attributes must be dict-like"):
        m.Metadata.model_validate({"attributes": 123})


def test_frozen_value_model_equality_and_hash() -> None:
    left = _FrozenValue(name="item", count=1)
    right = _FrozenValue(name="item", count=1)
    assert left == right
    assert left.__eq__(object()) is NotImplemented
    assert isinstance(hash(left), int)


def test_identifiable_unique_id_empty_rejected() -> None:
    with pytest.raises(
        ValidationError, match="String should have at least 1 character"
    ):
        _Identifiable(unique_id="   ")


def test_timestampable_timestamp_conversion_and_json_serializer() -> None:
    naive = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    model = _Timestampable(created_at=naive, updated_at=naive)
    assert model.created_at.tzinfo == UTC
    dumped = model.model_dump(mode="json")
    assert isinstance(dumped["created_at"], str)


def test_timestamped_model_and_alias_and_canonical_symbols() -> None:
    model = m.TimestampedModel()
    assert model.created_at.tzinfo == UTC
    assert hasattr(m, "TimestampedModel")
    assert r[str].ok("ok").value == "ok"
    assert c.Performance.DEFAULT_VERSION >= 1
    assert hasattr(u, "mapper")
