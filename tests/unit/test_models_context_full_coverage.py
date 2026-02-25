from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import pytest
import structlog.contextvars
from pydantic import BaseModel

from flext_core import m, t
from flext_core._models.base import FlextModelFoundation
from flext_core._models.context import (
    FlextModelsContext,
    _normalize_statistics_before,
    _normalize_to_mapping,
)


class _ModelDumpNotDict(BaseModel):
    def model_dump(
        self, *args: object, **kwargs: object
    ) -> dict[str, t.ConfigMapValue]:
        _ = (args, kwargs)
        return cast("dict[str, t.ConfigMapValue]", "not-dict")


class _ModelWithNoCallableDump:
    model_dump = "bad"


class _MappingLike(Mapping[str, t.ConfigMapValue]):
    def __init__(self, data: dict[str, t.ConfigMapValue]) -> None:
        self._data = data

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, key: str) -> t.ConfigMapValue:
        return self._data[key]


def test_to_general_value_dict_with_config_map() -> None:
    value = m.ConfigMap(root={"k": "v"})
    assert FlextModelsContext.to_general_value_dict(value) == {"k": "v"}


def test_structlog_proxy_context_var_get_set_reset_paths() -> None:
    structlog.contextvars.clear_contextvars()
    proxy = FlextModelsContext.StructlogProxyContextVar[str]("proxy_key", default="def")
    assert proxy.get() == "def"

    token = proxy.set("abc")
    assert proxy.get() == "abc"
    assert token.previous_value == "def"

    token_none = proxy.set(None)
    assert token_none.key == "proxy_key"
    assert proxy.get() == "def"

    FlextModelsContext.StructlogProxyContextVar.reset(
        FlextModelsContext.StructlogProxyToken(key="proxy_key", previous_value=None)
    )
    FlextModelsContext.StructlogProxyContextVar.reset(
        FlextModelsContext.StructlogProxyToken(
            key="proxy_key", previous_value="restored"
        )
    )
    assert proxy.get() == "restored"


def test_structlog_proxy_context_var_default_when_key_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proxy = FlextModelsContext.StructlogProxyContextVar[str]("missing_key", default="d")
    monkeypatch.setattr(
        "flext_core._models.context.structlog.contextvars.get_contextvars",
        lambda: {"other": "x"},
    )
    assert proxy.get() == "d"


def test_context_data_normalize_and_json_checks() -> None:
    nested = {"a": [{"b": 1}]}
    normalized = FlextModelsContext.ContextData.normalize_to_general_value(nested)
    assert isinstance(normalized, dict)

    FlextModelsContext.ContextData.check_json_serializable(
        cast(t.GeneralValueType, {"k": [1, "x"]})
    )

    with pytest.raises(TypeError):
        FlextModelsContext.ContextData.check_json_serializable(
            cast(t.GeneralValueType, {"bad": object()})
        )

    obj = object()
    assert (
        FlextModelsContext.ContextData.normalize_to_general_value(
            cast(t.GeneralValueType, obj)
        )
        is obj
    )


def test_context_data_validate_dict_serializable_error_paths() -> None:
    with pytest.raises(TypeError, match="Value must be a dictionary or Metadata"):
        FlextModelsContext.ContextData.validate_dict_serializable(
            cast("t.Dict | Mapping[str, t.ConfigMapValue] | BaseModel | None", 123)
        )

    with pytest.raises(TypeError, match="Value must be a dictionary or Metadata"):
        FlextModelsContext.ContextData.validate_dict_serializable(_ModelDumpNotDict())

    with pytest.raises(TypeError, match="Value must be a dictionary or Metadata"):
        FlextModelsContext.ContextData.validate_dict_serializable(
            cast(
                "t.Dict | Mapping[str, t.ConfigMapValue] | BaseModel | None",
                _ModelWithNoCallableDump(),
            )
        )

    metadata_input = FlextModelFoundation.Metadata(attributes={"a": 1})
    assert FlextModelsContext.ContextData.validate_dict_serializable(
        metadata_input
    ) == {"a": 1}

    class _GoodModel(BaseModel):
        b: int = 2

    assert FlextModelsContext.ContextData.validate_dict_serializable(_GoodModel()) == {
        "b": 2
    }


def test_context_data_validate_dict_serializable_none_and_mapping() -> None:
    assert FlextModelsContext.ContextData.validate_dict_serializable(None) == {}

    as_mapping = _MappingLike({"k": "v"})
    assert FlextModelsContext.ContextData.validate_dict_serializable(as_mapping) == {
        "k": "v"
    }


def test_context_data_validator_forces_non_dict_normalized_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "flext_core._models.context.FlextModelsContext.ContextData.normalize_to_general_value",
        lambda _v: "not-dict",
    )
    with pytest.raises(TypeError, match="Normalized value must be dict"):
        FlextModelsContext.ContextData.validate_dict_serializable({"a": 1})

    monkeypatch.setattr(
        "flext_core._models.context.FlextRuntime.is_dict_like",
        lambda _v: False,
    )
    with pytest.raises(TypeError, match="Value must be a dictionary or Metadata"):
        FlextModelsContext.ContextData.validate_dict_serializable({"a": 1})


def test_context_export_serializable_and_validators() -> None:
    FlextModelsContext.ContextExport.check_json_serializable(
        cast(t.GeneralValueType, {"k": [1, True]})
    )
    with pytest.raises(TypeError):
        FlextModelsContext.ContextExport.check_json_serializable(
            cast(t.GeneralValueType, {"x": object()})
        )

    with pytest.raises(TypeError, match="Value must be a dict or Pydantic model"):
        FlextModelsContext.ContextExport.validate_dict_serializable(_ModelDumpNotDict())

    with pytest.raises(TypeError, match="Value must be a dict or Pydantic model"):
        FlextModelsContext.ContextExport.validate_dict_serializable(
            cast(
                "t.Dict | Mapping[str, t.ConfigMapValue] | BaseModel | None",
                _ModelWithNoCallableDump(),
            )
        )

    assert FlextModelsContext.ContextExport.validate_dict_serializable(None) == {}


def test_context_export_validate_dict_serializable_mapping_and_errors() -> None:
    assert FlextModelsContext.ContextExport.validate_dict_serializable({"a": 1}) == {
        "a": 1
    }

    as_mapping = _MappingLike({"k": "v"})
    assert FlextModelsContext.ContextExport.validate_dict_serializable(as_mapping) == {
        "k": "v"
    }

    with pytest.raises(TypeError, match="Value must be a dict or Pydantic model"):
        FlextModelsContext.ContextExport.validate_dict_serializable(
            cast("t.Dict | Mapping[str, t.ConfigMapValue] | BaseModel | None", 123)
        )

    metadata_input = FlextModelFoundation.Metadata(attributes={"m": 3})
    assert FlextModelsContext.ContextExport.validate_dict_serializable(
        metadata_input
    ) == {"m": 3}

    class _GoodExportModel(BaseModel):
        c: int = 4

    assert FlextModelsContext.ContextExport.validate_dict_serializable(
        _GoodExportModel()
    ) == {"c": 4}

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        "flext_core._models.context.FlextRuntime.is_dict_like",
        lambda _v: False,
    )
    with pytest.raises(TypeError, match="Value must be a dict or Pydantic model"):
        FlextModelsContext.ContextExport.validate_dict_serializable({"a": 1})
    monkeypatch.undo()


def test_context_export_statistics_validator_and_computed_fields() -> None:
    assert _normalize_statistics_before(None) == {}
    assert _normalize_statistics_before({"a": 1}) == {"a": 1}

    class StatsModel(BaseModel):
        a: int = 1

    assert _normalize_statistics_before(StatsModel()) == {"a": 1}

    with pytest.raises(TypeError, match="statistics must be dict or BaseModel"):
        _normalize_statistics_before(cast(t.GuardInputValue, "x"))

    exported = m.ContextExport(data={"k": "v"}, statistics={"sets": 1})
    assert exported.total_data_items == 1
    assert exported.has_statistics is True


def test_scope_data_validators_and_errors() -> None:
    class ScopeModel(BaseModel):
        a: int = 1

    assert _normalize_to_mapping(None) == {}
    assert _normalize_to_mapping({"a": 1}) == {"a": 1}
    assert _normalize_to_mapping(ScopeModel()) == {"a": 1}

    with pytest.raises(TypeError, match="must be dict or BaseModel"):
        _normalize_to_mapping(cast(t.GuardInputValue, 123))

    assert _normalize_to_mapping(None) == {}
    assert _normalize_to_mapping({"a": 1}) == {"a": 1}
    assert _normalize_to_mapping(ScopeModel()) == {"a": 1}

    with pytest.raises(TypeError, match="must be dict or BaseModel"):
        _normalize_to_mapping(cast(t.GuardInputValue, 123))


def test_statistics_and_custom_fields_validators() -> None:
    class Payload(BaseModel):
        p: int = 2

    assert _normalize_to_mapping({"x": 1}) == {"x": 1}
    assert _normalize_to_mapping(Payload()) == {"p": 2}
    assert _normalize_to_mapping(None) == {}
    with pytest.raises(TypeError, match="must be dict or BaseModel"):
        _normalize_to_mapping(cast(t.GuardInputValue, "bad"))

    assert _normalize_to_mapping({"x": 1}) == {"x": 1}
    assert _normalize_to_mapping(Payload()) == {"p": 2}
    assert _normalize_to_mapping(None) == {}
    with pytest.raises(TypeError, match="must be dict or BaseModel"):
        _normalize_to_mapping(cast(t.GuardInputValue, "bad"))


def test_context_data_metadata_normalizer_paths() -> None:
    md = FlextModelFoundation.Metadata(attributes={"k": "v"})
    assert FlextModelsContext.normalize_metadata(md).attributes["k"] == "v"
    assert FlextModelsContext.normalize_metadata({"x": 1}).attributes["x"] == 1

    with pytest.raises(
        TypeError, match="metadata must be None, dict, or FlextModelsBase.Metadata"
    ):
        _ = FlextModelsContext.normalize_metadata(cast("t.ConfigMap", 1.23))
