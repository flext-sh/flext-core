"""Tests for Context models full coverage."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import cast, override

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


class _ModelWithNoCallableDump:
    model_dump = "bad"


class _MappingLike(Mapping[str, t.ContainerValue]):
    def __init__(self, data: dict[str, t.ContainerValue]) -> None:
        super().__init__()
        self._data = data

    @override
    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    @override
    def __len__(self) -> int:
        return len(self._data)

    @override
    def __getitem__(self, key: str) -> t.ContainerValue:
        return self._data[key]


def test_to_general_value_dict_removed() -> None:
    """to_general_value_dict was removed during infra migration."""
    assert not hasattr(FlextModelsContext, "to_general_value_dict")


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
    check_result = FlextModelsContext.ContextData.check_json_serializable(
        cast("t.ContainerValue", {"k": [1, "x"]})
    )
    assert check_result is None
    with pytest.raises(TypeError):
        FlextModelsContext.ContextData.check_json_serializable(
            cast("t.ContainerValue", {"bad": object()})
        )
    obj = object()
    normalized_obj = FlextModelsContext.ContextData.normalize_to_general_value(
        cast("t.ContainerValue", obj)
    )
    assert normalized_obj is obj


def test_context_data_validate_dict_serializable_error_paths() -> None:
    with pytest.raises(
        TypeError, match="Value must be a dictionary or Metadata"
    ) as exc_info:
        _ = FlextModelsContext.ContextData.validate_dict_serializable(
            cast(
                "t.Dict | Mapping[str, t.ContainerValue] | BaseModel | None",
                cast("object", 123),
            )
        )
    assert exc_info.value is not None
    with pytest.raises(
        TypeError, match="Value must be a dictionary or Metadata"
    ) as exc_info2:
        _ = FlextModelsContext.ContextData.validate_dict_serializable(
            cast(
                "t.Dict | Mapping[str, t.ContainerValue] | BaseModel | None",
                cast("object", _ModelWithNoCallableDump()),
            )
        )
    assert exc_info2.value is not None
    metadata_input = FlextModelFoundation.Metadata(attributes={"a": 1})
    result = FlextModelsContext.ContextData.validate_dict_serializable(metadata_input)
    assert result == {"a": 1}

    class _GoodModel(BaseModel):
        b: int = 2

    result_b = FlextModelsContext.ContextData.validate_dict_serializable(_GoodModel())
    assert result_b == {"b": 2}


def test_context_data_validate_dict_serializable_none_and_mapping() -> None:
    result_none = FlextModelsContext.ContextData.validate_dict_serializable(None)
    assert result_none == {}
    as_mapping = _MappingLike({"k": "v"})
    result_mapping = FlextModelsContext.ContextData.validate_dict_serializable(
        as_mapping
    )
    assert result_mapping == {"k": "v"}


@pytest.mark.parametrize(
    ("input_value", "expected_result"),
    [
        ({"a": 1}, {"a": 1}),
        ({"nested": {"b": 2}}, {"nested": {"b": 2}}),
        ({}, {}),
    ],
    ids=["simple-dict", "nested-dict", "empty-dict"],
)
def test_context_data_validate_dict_serializable_real_dicts(
    input_value: dict[str, t.ContainerValue],
    expected_result: dict[str, t.ContainerValue],
) -> None:
    """Test validate_dict_serializable with real dict inputs."""
    result = FlextModelsContext.ContextData.validate_dict_serializable(input_value)
    assert result == expected_result


def test_context_export_serializable_and_validators() -> None:
    check_result = FlextModelsContext.ContextData.check_json_serializable(
        cast("t.ContainerValue", {"k": [1, True]})
    )
    assert check_result is None
    with pytest.raises(TypeError):
        FlextModelsContext.ContextData.check_json_serializable(
            cast("t.ContainerValue", {"x": object()})
        )
    with pytest.raises(TypeError, match="Value must be a dict or Pydantic model"):
        _ = FlextModelsContext.ContextExport.validate_dict_serializable(
            cast(
                "t.Dict | Mapping[str, t.ContainerValue] | BaseModel | None",
                cast("object", _ModelWithNoCallableDump()),
            )
        )
    result = FlextModelsContext.ContextExport.validate_dict_serializable(None)
    assert result == {}


@pytest.mark.parametrize(
    ("input_value", "expected_result"),
    [
        ({"a": 1}, {"a": 1}),
        (None, {}),
    ],
    ids=["dict-input", "none-input"],
)
def test_context_export_validate_dict_serializable_valid(
    input_value: dict[str, t.ContainerValue] | None,
    expected_result: dict[str, t.ContainerValue],
) -> None:
    """Test ContextExport.validate_dict_serializable with valid inputs."""
    result = FlextModelsContext.ContextExport.validate_dict_serializable(input_value)
    assert result == expected_result


def test_context_export_validate_dict_serializable_mapping_and_models() -> None:
    """Test ContextExport.validate_dict_serializable with Mapping and model inputs."""
    as_mapping = _MappingLike({"k": "v"})
    result_mapping = FlextModelsContext.ContextExport.validate_dict_serializable(
        as_mapping
    )
    assert result_mapping == {"k": "v"}
    with pytest.raises(TypeError, match="Value must be a dict or Pydantic model"):
        _ = FlextModelsContext.ContextExport.validate_dict_serializable(
            cast(
                "t.Dict | Mapping[str, t.ContainerValue] | BaseModel | None",
                cast("object", 123),
            )
        )
    metadata_input = FlextModelFoundation.Metadata(attributes={"m": 3})
    result_meta = FlextModelsContext.ContextExport.validate_dict_serializable(
        metadata_input
    )
    assert result_meta == {"m": 3}

    class _GoodExportModel(BaseModel):
        c: int = 4

    result_export = FlextModelsContext.ContextExport.validate_dict_serializable(
        _GoodExportModel()
    )
    assert result_export == {"c": 4}


def test_context_export_statistics_validator_and_computed_fields() -> None:
    stats_none = _normalize_statistics_before(None)
    assert stats_none == {}
    stats_dict = _normalize_statistics_before({"a": 1})
    assert stats_dict == {"a": 1}

    class StatsModel(BaseModel):
        a: int = 1

    stats_model = _normalize_statistics_before(StatsModel())
    assert stats_model == {"a": 1}
    with pytest.raises(ValueError, match="Cannot normalize"):
        _normalize_statistics_before(cast("t.ContainerValue", "x"))
    exported = m.ContextExport(data={"k": "v"}, statistics={"sets": 1})
    assert exported.total_data_items == 1
    assert exported.has_statistics is True


def test_scope_data_validators_and_errors() -> None:

    class ScopeModel(BaseModel):
        a: int = 1

    result_none = _normalize_to_mapping(None)
    assert result_none == {}
    result_dict = _normalize_to_mapping({"a": 1})
    assert result_dict == {"a": 1}
    result_scope = _normalize_to_mapping(ScopeModel())
    assert result_scope == {"a": 1}
    with pytest.raises(ValueError, match="Cannot normalize"):
        _normalize_to_mapping(cast("t.ContainerValue", 123))
    result_none2 = _normalize_to_mapping(None)
    assert result_none2 == {}
    result_dict2 = _normalize_to_mapping({"a": 1})
    assert result_dict2 == {"a": 1}
    result_scope2 = _normalize_to_mapping(ScopeModel())
    assert result_scope2 == {"a": 1}
    with pytest.raises(ValueError, match="Cannot normalize"):
        _normalize_to_mapping(cast("t.ContainerValue", 123))


def test_statistics_and_custom_fields_validators() -> None:

    class Payload(BaseModel):
        p: int = 2

    result_x1 = _normalize_to_mapping({"x": 1})
    assert result_x1 == {"x": 1}
    result_payload1 = _normalize_to_mapping(Payload())
    assert result_payload1 == {"p": 2}
    result_none1 = _normalize_to_mapping(None)
    assert result_none1 == {}
    with pytest.raises(ValueError, match="Cannot normalize"):
        _normalize_to_mapping(cast("t.ContainerValue", "bad"))
    result_x2 = _normalize_to_mapping({"x": 1})
    assert result_x2 == {"x": 1}
    result_payload2 = _normalize_to_mapping(Payload())
    assert result_payload2 == {"p": 2}
    result_none2 = _normalize_to_mapping(None)
    assert result_none2 == {}
    with pytest.raises(ValueError, match="Cannot normalize"):
        _normalize_to_mapping(cast("t.ContainerValue", "bad"))


def test_context_data_metadata_normalizer_removed() -> None:
    """normalize_metadata was removed during infra migration."""
    assert not hasattr(FlextModelsContext, "normalize_metadata")
