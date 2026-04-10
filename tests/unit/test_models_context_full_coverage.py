"""Tests for Context models full coverage."""

from __future__ import annotations

from typing import cast

import pytest
import structlog.contextvars
from pydantic import BaseModel

import flext_core._models._context._proxy_var as _proxy_var_mod
from flext_core import FlextModelsContextProxyVar
from flext_tests import tm
from tests import m, t


class _ModelWithNoCallableDump:
    model_dump = "bad"


def test_to_general_value_dict_removed() -> None:
    """to_general_value_dict was removed during infra migration."""
    tm.that(not hasattr(m, "to_general_value_dict"), eq=True)


def test_structlog_proxy_context_var_get_set_reset_paths() -> None:
    structlog.contextvars.clear_contextvars()
    proxy = FlextModelsContextProxyVar.StructlogProxyContextVar(
        "proxy_key",
        default="def",
    )
    tm.that(proxy.get(), eq="def")
    token = proxy.set("abc")
    tm.that(proxy.get(), eq="abc")
    tm.that(token.previous_value, eq="def")
    token_none = proxy.set(None)
    tm.that(token_none.key, eq="proxy_key")
    tm.that(proxy.get(), eq="def")
    m.StructlogProxyContextVar.reset(
        m.StructlogProxyToken(key="proxy_key", previous_value=None),
    )
    m.StructlogProxyContextVar.reset(
        m.StructlogProxyToken(
            key="proxy_key",
            previous_value="restored",
        ),
    )
    tm.that(proxy.get(), eq="restored")


def test_structlog_proxy_context_var_default_when_key_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proxy = FlextModelsContextProxyVar.StructlogProxyContextVar(
        "missing_key",
        default="d",
    )

    monkeypatch.setattr(
        _proxy_var_mod.structlog.contextvars,
        "get_contextvars",
        lambda: {"other": "x"},
    )
    tm.that(proxy.get(), eq="d")


def test_context_data_normalize_and_json_checks() -> None:
    nested: t.NormalizedValue = cast("t.NormalizedValue", {"a": [{"b": 1}]})
    m.ContextData.normalize_to_container(nested)
    check_result = m.ContextData.check_json_serializable(
        cast("t.ValueOrModel", {"k": [1, "x"]}),
    )
    tm.that(check_result, none=True)
    with pytest.raises(TypeError):
        m.ContextData.check_json_serializable(
            cast("t.ValueOrModel", {"normalized"}),
        )
    obj = cast("t.ValueOrModel", _ModelWithNoCallableDump())
    with pytest.raises(TypeError):
        m.ContextData.normalize_to_container(obj)


def test_context_data_validate_dict_serializable_error_paths() -> None:
    with pytest.raises(ValueError) as exc_info:
        _ = m.ContextData.validate_dict_serializable(
            cast("t.Dict | t.ConfigurationMapping | BaseModel | None", "123"),
        )
    tm.that(exc_info.value, none=False)
    with pytest.raises(TypeError) as exc_info2:
        _ = m.ContextData.validate_dict_serializable(
            cast(
                "t.Dict | t.ConfigurationMapping | BaseModel | None",
                cast("t.NormalizedValue", _ModelWithNoCallableDump()),
            ),
        )
    tm.that(exc_info2.value, none=False)
    metadata_input = m.Metadata(attributes={"a": 1})
    result = m.ContextData.validate_dict_serializable(metadata_input)
    tm.that(result, eq={"a": 1})

    class _GoodModel(BaseModel):
        b: int = 2

    result_b = m.ContextData.validate_dict_serializable(_GoodModel())
    tm.that(result_b, eq={"b": 2})


def test_context_data_validate_dict_serializable_none_and_mapping() -> None:
    result_none = m.ContextData.validate_dict_serializable(None)
    tm.that(result_none, eq={})
    as_mapping: t.ConfigurationMapping = {"k": "v"}
    result_mapping = m.ContextData.validate_dict_serializable(
        as_mapping,
    )
    tm.that(result_mapping, eq={"k": "v"})


@pytest.mark.parametrize(
    ("input_value", "expected_result"),
    [
        (t.Dict(root={"a": 1}), {"a": 1}),
        (t.Dict(root={"nested": {"b": 2}}), {"nested": {"b": 2}}),
        (t.Dict(root={}), {}),
    ],
    ids=["simple-dict", "nested-dict", "empty-dict"],
)
def test_context_data_validate_dict_serializable_real_dicts(
    input_value: t.Dict,
    expected_result: t.ContainerMapping,
) -> None:
    """Test validate_dict_serializable with real dict inputs."""
    result = m.ContextData.validate_dict_serializable(input_value)
    tm.that(result, eq=expected_result)


def test_context_export_serializable_and_validators() -> None:
    check_result = m.ContextData.check_json_serializable(
        cast("t.ValueOrModel", {"k": [1, True]}),
    )
    tm.that(check_result, none=True)
    with pytest.raises(TypeError):
        _ = m.ContextData.check_json_serializable(
            cast("t.ValueOrModel", {"normalized"}),
        )
    with pytest.raises(TypeError):
        _ = m.ContextExport.validate_dict_serializable(
            cast(
                "t.Dict | t.ConfigurationMapping | BaseModel | None",
                cast("t.NormalizedValue", _ModelWithNoCallableDump()),
            ),
        )
    result = m.ContextExport.validate_dict_serializable(None)
    tm.that(result, eq={})


@pytest.mark.parametrize(
    ("input_value", "expected_result"),
    [
        ({"a": 1}, {"a": 1}),
        (None, {}),
    ],
    ids=["dict-input", "none-input"],
)
def test_context_export_validate_dict_serializable_valid(
    input_value: t.ConfigurationMapping | None,
    expected_result: t.ConfigurationMapping,
) -> None:
    """Test ContextExport.validate_dict_serializable with valid inputs."""
    result = m.ContextExport.validate_dict_serializable(input_value)
    tm.that(result, eq=expected_result)


def test_context_export_validate_dict_serializable_mapping_and_models() -> None:
    """Test ContextExport.validate_dict_serializable with Mapping and model inputs."""
    as_mapping: t.ConfigurationMapping = {"k": "v"}
    result_mapping = m.ContextExport.validate_dict_serializable(
        as_mapping,
    )
    tm.that(result_mapping, eq={"k": "v"})
    with pytest.raises(ValueError):
        _ = m.ContextExport.validate_dict_serializable(
            cast(
                "t.Dict | t.ConfigurationMapping | BaseModel | None",
                "123",
            ),
        )
    metadata_input = m.Metadata(attributes={"m": 3})
    result_meta = m.ContextExport.validate_dict_serializable(
        metadata_input,
    )
    tm.that(result_meta, eq={"m": 3})

    class _GoodExportModel(BaseModel):
        c: int = 4

    result_export = m.ContextExport.validate_dict_serializable(
        _GoodExportModel(),
    )
    tm.that(result_export, eq={"c": 4})


def test_context_export_statistics_validator_and_computed_fields() -> None:
    stats_none = m.normalize_to_mapping(None)
    tm.that(stats_none, eq={})
    stats_dict = m.normalize_to_mapping({"a": 1})
    tm.that(stats_dict, eq={"a": 1})

    class StatsModel(BaseModel):
        a: int = 1

    stats_model = m.normalize_to_mapping(StatsModel())
    tm.that(stats_model, eq={"a": 1})
    with pytest.raises(ValueError, match="Cannot normalize"):
        _ = m.normalize_to_mapping(cast("t.ValueOrModel", "x"))
    exported = m.ContextExport(data={"k": "v"}, statistics={"sets": 1})
    total_items = cast("int", exported.total_data_items)
    tm.that(total_items, eq=1)
    has_stats = cast("bool", exported.has_statistics)
    tm.that(has_stats is True, eq=True)


def test_scope_data_validators_and_errors() -> None:

    class ScopeModel(BaseModel):
        a: int = 1

    result_none = m.normalize_to_mapping(None)
    tm.that(result_none, eq={})
    result_dict = m.normalize_to_mapping({"a": 1})
    tm.that(result_dict, eq={"a": 1})
    result_scope = m.normalize_to_mapping(ScopeModel())
    tm.that(result_scope, eq={"a": 1})
    with pytest.raises(ValueError, match="Cannot normalize"):
        _ = m.normalize_to_mapping(cast("t.ValueOrModel", 123))
    result_none2 = m.normalize_to_mapping(None)
    tm.that(result_none2, eq={})
    result_dict2 = m.normalize_to_mapping({"a": 1})
    tm.that(result_dict2, eq={"a": 1})
    result_scope2 = m.normalize_to_mapping(ScopeModel())
    tm.that(result_scope2, eq={"a": 1})
    with pytest.raises(ValueError, match="Cannot normalize"):
        _ = m.normalize_to_mapping(cast("t.ValueOrModel", 123))


def test_statistics_and_custom_fields_validators() -> None:

    class Payload(BaseModel):
        p: int = 2

    result_x1 = m.normalize_to_mapping({"x": 1})
    tm.that(result_x1, eq={"x": 1})
    result_payload1 = m.normalize_to_mapping(Payload())
    tm.that(result_payload1, eq={"p": 2})
    result_none1 = m.normalize_to_mapping(None)
    tm.that(result_none1, eq={})
    with pytest.raises(ValueError, match="Cannot normalize"):
        _ = m.normalize_to_mapping(cast("t.ValueOrModel", "bad"))
    result_x2 = m.normalize_to_mapping({"x": 1})
    tm.that(result_x2, eq={"x": 1})
    result_payload2 = m.normalize_to_mapping(Payload())
    tm.that(result_payload2, eq={"p": 2})
    result_none2 = m.normalize_to_mapping(None)
    tm.that(result_none2, eq={})
    with pytest.raises(ValueError, match="Cannot normalize"):
        _ = m.normalize_to_mapping(cast("t.ValueOrModel", "bad"))


def test_context_data_metadata_normalizer_removed() -> None:
    """normalize_metadata was removed during infra migration."""
    tm.that(not hasattr(m, "normalize_metadata"), eq=True)
