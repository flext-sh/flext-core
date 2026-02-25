from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
import builtins
import importlib

import pytest
from pydantic import TypeAdapter as PydanticTypeAdapter
from pydantic import ValidationError as PydanticValidationError

_core = importlib.import_module("flext_core")
c = _core.c
m = _core.m
r = _core.r
t = _core.t
u = _core.u


class _Validator:
    def __init__(self, ok: bool, description: str = "validator") -> None:
        self.ok = ok
        self.description = description
        self.predicate = type("Predicate", (), {"description": description})()

    def __call__(self, value: object) -> bool:
        return self.ok

    def __and__(self, other: object) -> _Validator:
        return self

    def __or__(self, other: object) -> _Validator:
        return self

    def __invert__(self) -> _Validator:
        return self


class _ItemsFailing:
    def items(self) -> object:
        return 123


class _ModelDumpDict:
    def model_dump(self) -> dict[str, t.GeneralValueType]:
        return {"x": 1}


class _ModelDumpTypeError:
    def model_dump(self) -> dict[str, t.GeneralValueType]:
        msg = "boom"
        raise TypeError(msg)


class _BadDictKey:
    def __hash__(self) -> int:
        return 1

    def __str__(self) -> str:
        msg = "bad key"
        raise TypeError(msg)


class _SimpleModel(m.ValueObject):
    value: int


@dataclass
class _SimpleDataclass:
    name: str
    value: int


class _BadVars:
    __slots__ = ()


class _BrokenDictLike:
    def items(self) -> object:
        return 123


class _BrokenPydantic:
    def model_dump(self) -> dict[str, t.GeneralValueType]:
        msg = "broken dump"
        raise ValueError(msg)


class _Event:
    def __init__(self) -> None:
        self.event_type = "evt"
        self.aggregate_id = "agg"
        self.unique_id = "uid"
        self.created_at = datetime.now()


def test_network_nested_failures() -> None:
    uri_adapter = PydanticTypeAdapter(t.Validation.UriString)
    with pytest.raises(PydanticValidationError):
        uri_adapter.validate_python(None)
    with pytest.raises(PydanticValidationError):
        uri_adapter.validate_python("")
    assert uri_adapter.validate_python("example.com") == "example.com"
    assert uri_adapter.validate_python("ftp://example.com") == "ftp://example.com"

    port_adapter = PydanticTypeAdapter(t.Validation.PortNumber)
    with pytest.raises(PydanticValidationError):
        port_adapter.validate_python(None)
    with pytest.raises(PydanticValidationError):
        port_adapter.validate_python(c.Network.MAX_PORT + 1)


def test_normalization_private_branch_coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    assert u.Validation._handle_pydantic_model(_ModelDumpDict()) == {"x": 1}
    assert u.Validation._ensure_general_value_type("x") == "x"
    assert "class" in u.Validation._ensure_general_value_type(str)

    cfg = t.ConfigMap(root={"k": 1})
    assert u.Validation._normalize_by_type(cfg) == {"k": 1}

    normalized_set = u.Validation._normalize_by_type({"a", "b"})
    assert isinstance(normalized_set, dict)
    assert normalized_set["type"] == "set"

    class _Opaque:
        def __str__(self) -> str:
            return "opaque"

    assert u.Validation._normalize_by_type(_Opaque()) == "opaque"

    assert u.Validation._convert_items_to_dict(
        iter([("a", 1), "skip", ("x", "y", "z")])
    ) == {"a": 1}

    with pytest.raises(
        TypeError,
        match=r"Cannot convert _ItemsFailing\.items\(\) to dict",
    ):
        u.Validation._extract_dict_from_component(_ItemsFailing())

    assert u.Validation._convert_items_result_to_dict(
        iter([("a", 1), "skip", ("x", "y", "z")])
    ) == {"a": 1}

    with pytest.raises(
        TypeError,
        match=r"Cannot convert _BrokenDictLike\.items\(\) to dict",
    ):
        u.Validation._convert_to_mapping(_BrokenDictLike())

    loop: dict[str, t.GeneralValueType] = {}
    loop["self"] = loop
    monkeypatch.setattr(
        "flext_core._utilities.validation.FlextUtilitiesValidation._convert_to_mapping",
        staticmethod(lambda component: loop),
    )
    normalized = u.Validation._normalize_dict_like({"ignored": 1})
    assert isinstance(normalized["self"], dict)
    assert normalized["self"]["type"] == "circular_reference"

    class _MappingOnly(Mapping[str, t.GeneralValueType]):
        def __getitem__(self, key: str) -> t.GeneralValueType:
            if key == "a":
                return 1
            msg = "missing"
            raise KeyError(msg)

        def __iter__(self):
            yield "a"

        def __len__(self) -> int:
            return 1

    assert u.Validation._normalize_value(_MappingOnly()) == {"a": 1}

    original_is_type = (
        u.Validation.Guards.is_type if hasattr(u.Validation, "Guards") else None
    )
    monkeypatch.setattr(
        "flext_core._utilities.validation.FlextUtilitiesGuards.is_type",
        lambda value, kind: kind == "mapping",
    )
    fallback = u.Validation._normalize_value(object())
    assert isinstance(fallback, dict)
    assert len(fallback) == 1
    if original_is_type is not None:
        _ = original_is_type


def test_set_helper_pipeline_sort_and_primitive_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_normalize = u.Validation._normalize_component
    monkeypatch.setattr(
        u.Validation,
        "_normalize_component",
        staticmethod(lambda item, visited=None: (item, [1])),
    )
    result_set = u.Validation._normalize_set_helper({"a"})
    assert any(isinstance(item, str) for item in result_set)
    monkeypatch.setattr(
        u.Validation,
        "_normalize_component",
        staticmethod(original_normalize),
    )

    failure = u.Validation.validate_pipeline("x", [lambda _: r[bool].fail("bad")])
    assert failure.is_failure
    assert "bad" in str(failure.error)

    class _JsonBoom:
        def __str__(self) -> str:
            return "boom"

    sort_tuple = u.Validation.sort_key(_JsonBoom())
    assert sort_tuple[0] == "other"
    assert sort_tuple[1] == '"boom"'

    @dataclass
    class _DC:
        x: int

    assert u.Validation._is_dataclass_instance(_DC(1)) is True

    assert u.Validation._normalize_primitive_or_bytes(1) == (True, 1)
    primitive_bytes = u.Validation._normalize_primitive_or_bytes(b"ab")
    assert primitive_bytes[0] is True
    assert isinstance(primitive_bytes[1], dict)
    assert u.Validation._normalize_primitive_or_bytes(object()) == (False, None)


def test_pydantic_and_collection_normalizers_and_key_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(TypeError, match="Failed to dump Pydantic value"):
        u.Validation._normalize_pydantic_value(_ModelDumpTypeError())

    assert isinstance(u.Validation._normalize_mapping({"b": 2, "a": 1}), dict)
    assert isinstance(u.Validation._normalize_sequence([1, 2]), dict)
    assert isinstance(u.Validation._normalize_set({1, 2}), dict)
    assert u.Validation._normalize_vars(_BadVars())["type"] == "repr"

    assert u.Validation._generate_key_pydantic(_SimpleModel(value=1), _SimpleModel)
    assert u.Validation._generate_key_pydantic(_BrokenPydantic(), _SimpleModel) is None

    dc = _SimpleDataclass(name="a", value=1)
    assert u.Validation._generate_key_dataclass(dc, _SimpleDataclass)

    class _BadFieldDataclass:
        __dataclass_fields__ = None

    assert (
        u.Validation._generate_key_dataclass(_BadFieldDataclass(), _SimpleDataclass)
        is None
    )

    bad_key = _BadDictKey()
    data: dict[object, t.GeneralValueType] = {bad_key: 1}
    assert u.Validation._generate_key_dict(data, dict) is None

    original_hash = builtins.hash

    def fake_hash(value: object) -> int:
        if isinstance(value, str):
            msg = "str hash fail"
            raise TypeError(msg)
        return original_hash(value)

    monkeypatch.setattr(builtins, "hash", fake_hash)
    key = u.Validation.generate_cache_key("abc", str)
    assert key.startswith("str_")


def test_sort_dict_keys_tuple_and_list_paths_and_initialize() -> None:
    tuple_result = u.Validation.sort_dict_keys(("z", {"b": 2, "a": 1}, object(), None))
    assert isinstance(tuple_result, tuple)
    assert tuple_result[0] == "z"

    list_result = u.Validation.sort_dict_keys([{"b": 2, "a": 1}, object()])
    assert isinstance(list_result, list)
    assert isinstance(list_result[0], dict)
    assert isinstance(list_result[1], str)

    class _Obj:
        initialized: bool = False

    o = _Obj()
    u.Validation.initialize(o, "initialized")
    assert o.initialized is True


def test_validation_methods_more_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    assert u.Validation.validate_uri("http://x", allowed_schemes=["https"]).is_failure
    assert u.Validation.validate_port_number(None).is_failure
    assert u.Validation._validate_numeric_constraint(
        None, lambda x: x > 0, "must pass"
    ).is_failure

    assert u.Validation.validate_http_status_codes([object()]).is_failure
    assert u.Validation.validate_http_status_codes(["not-int"]).is_failure

    timeout_error = u.Validation.format_error_message(
        TimeoutError(), timeout_seconds=2.5
    )
    assert "timed out" in timeout_error

    assert u.Validation.validate_batch_services(t.ServiceMap(root={})).is_failure
    assert u.Validation.validate_batch_services(
        t.ServiceMap(root={"_x": "svc"})
    ).is_failure
    assert u.Validation.validate_batch_services(
        t.ServiceMap(root={"x": None})
    ).is_failure

    assert u.Validation.validate_dispatch_config(None).is_failure
    assert u.Validation.validate_dispatch_config(object()).is_failure
    assert u.Validation.validate_dispatch_config({"correlation_id": 1}).is_failure
    assert u.Validation.validate_dispatch_config({"timeout": "soon"}).is_failure

    assert u.Validation._validate_event_structure(None).is_failure
    assert u.Validation._validate_event_fields(
        type("E", (), {"event_type": "", "aggregate_id": "a", "data": None})()
    ).is_failure
    assert u.Validation._validate_event_fields(
        type("E", (), {"event_type": "ok", "aggregate_id": "", "data": None})()
    ).is_failure
    assert u.Validation._validate_event_fields(
        type("E", (), {"event_type": "ok", "aggregate_id": "a", "data": object()})()
    ).is_failure

    monkeypatch.setattr(
        "flext_core._utilities.validation.FlextUtilitiesValidation._validate_event_structure",
        staticmethod(lambda event: r[bool].ok(True)),
    )
    assert u.Validation.validate_domain_event(None).is_failure

    monkeypatch.setattr(
        "flext_core._utilities.validation.FlextUtilitiesValidation._validate_event_structure",
        staticmethod(lambda event: r[bool].ok(True)),
    )
    monkeypatch.setattr(
        "flext_core._utilities.validation.FlextUtilitiesValidation._validate_event_fields",
        staticmethod(lambda event: r[bool].fail("bad fields")),
    )
    assert u.Validation.validate_domain_event(_Event()).is_failure


def test_validate_internal_check_helpers_and_entity() -> None:
    validator = _Validator(ok=False, description="pred-desc")
    assert u.Validation._validate_get_desc(validator) == "pred-desc"

    any_fail = u.Validation._validate_check_any("x", (validator,), "field: ")
    assert any_fail.is_failure
    assert "None of the validators passed" in str(any_fail.error)

    all_fail_fast = u.Validation._validate_check_all(
        "x",
        (validator,),
        "field: ",
        fail_fast=True,
        collect_errors=False,
    )
    assert all_fail_fast.is_failure
    assert "Validation failed" in str(all_fail_fast.error)

    assert u.Validation.entity(None).is_failure


def test_validate_all_timestamp_guard_branches_and_shortcuts() -> None:
    assert u.Validation.validate_all(
        [1, -1], lambda v: v > 0, fail_fast=True
    ).is_failure
    assert u.Validation.validate_timestamp_format("") is True

    type_err = u.Validation._guard_check_type("x", int, "Value", None)
    assert type_err is not None and "must be int" in type_err
    assert u.Validation._guard_check_type("x", int, "Value", "custom") == "custom"

    v_fail = _Validator(ok=False, description="desc")
    validator_err = u.Validation._guard_check_validator("x", v_fail, "Field", None)
    assert validator_err is not None and "failed" in validator_err
    assert (
        u.Validation._guard_check_validator("x", v_fail, "Field", "custom") == "custom"
    )

    pred_err = u.Validation._guard_check_predicate("x", lambda v: False, "Field", None)
    assert pred_err is not None and "failed" in pred_err

    def _raises(_: object) -> bool:
        msg = "boom"
        raise RuntimeError(msg)

    raised_err = u.Validation._guard_check_predicate("x", _raises, "Field", None)
    assert raised_err is not None and "raised" in raised_err

    assert u.Validation._guard_check_condition("x", int, "Field", None) is not None
    assert (
        u.Validation._guard_check_condition("x", (int, float), "Field", None)
        is not None
    )
    assert u.Validation._guard_check_condition("x", v_fail, "Field", None) is not None
    assert u.Validation._guard_check_condition("x", 123, "Field", None) is not None

    assert (
        u.Validation._guard_handle_failure("err", return_value=True, default="fallback")
        == "fallback"
    )
    fail_result = u.Validation._guard_handle_failure(
        "err", return_value=False, default=None
    )
    assert isinstance(fail_result, r)
    assert fail_result.is_failure

    assert u.Validation._guard_non_empty({}, "Value must be").is_failure
    assert u.Validation._guard_non_empty([], "Value must be").is_failure
    assert u.Validation._guard_non_empty(1, "Value must be").is_failure

    assert u.Validation._guard_shortcut(1, "int", "Value").is_success
    assert u.Validation._guard_shortcut("x", "int", "Value").is_failure
    assert u.Validation._guard_shortcut("x", "does_not_exist", "Value").is_failure

    assert u.Validation.guard("abc", str, return_value=True) == "abc"


def test_result_helpers_and_type_adapter_and_hostname_paths() -> None:
    assert u.Validation.ResultHelpers.err(r[int].ok(1), default="d") == "d"
    assert u.Validation.ResultHelpers.val(r[int].fail("x"), default=5) == 5
    assert u.Validation.ResultHelpers.vals(r[dict[str, int]].fail("x")) == []
    assert u.Validation.ResultHelpers.vals({}, default=None) == []
    assert u.Validation.ResultHelpers.or_(None, None, default="z") == "z"

    with pytest.raises(TypeError):
        u.Validation.ResultHelpers.try_(
            lambda: (_ for _ in ()).throw(TypeError("x")), catch=ValueError
        )

    then_fail = u.Validation.ResultHelpers.then(
        r[int].fail("bad"), lambda v: r[str].ok(str(v))
    )
    assert then_fail.is_failure

    assert u.Validation.ResultHelpers.empty(r[list[int]].fail("x")) is True
    assert u.Validation.ResultHelpers.empty(None) is True
    assert u.Validation.ResultHelpers.empty("") is True

    cfg = t.ConfigMap(root={"a": 1})
    assert u.Validation.ResultHelpers.in_("a", cfg) is True

    assert u.Validation.ResultHelpers.count([1, 2, 3]) == 3
    assert u.Validation.ResultHelpers.count({"a": 1, "b": 2}, lambda v: v > 1) == 1

    adapter_fail = u.Validation.TypeAdapter.validate({"value": "x"}, _SimpleModel)
    assert adapter_fail.is_failure

    serialize_fail = u.Validation.TypeAdapter.serialize(object(), type[object])
    assert serialize_fail.is_failure

    parse_fail = u.Validation.TypeAdapter.parse_json(None, dict[str, int])
    assert parse_fail.is_failure

    assert u.Validation.validate_hostname_format(None).is_failure
    assert u.Validation.validate_hostname_format("").is_failure


def test_validate_with_validators_failure_and_exception_paths() -> None:
    fail_result = u.Validation.validate_with_validators(
        "x", _Validator(ok=False, description="bad")
    )
    assert fail_result.is_failure
    assert "bad" in str(fail_result.error)

    class _RaisingValidator(_Validator):
        def __call__(self, value: object) -> bool:
            msg = "explode"
            raise RuntimeError(msg)

    error_result = u.Validation.validate_with_validators(
        "x", _RaisingValidator(ok=True)
    )
    assert error_result.is_failure
    assert "Validator error" in str(error_result.error)


def test_remaining_required_line_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(TypeError, match=r"Cannot convert object to dict"):
        u.Validation._extract_dict_from_component(object())

    original_normalize = u.Validation._normalize_component
    original_is_type = u.Guards.is_type
    monkeypatch.setattr(
        "flext_core._utilities.validation.FlextUtilitiesValidation._normalize_component",
        staticmethod(lambda item, visited=None: ("x", [1])),
    )
    monkeypatch.setattr(
        "flext_core._utilities.validation.FlextUtilitiesGuards.is_type",
        lambda value, condition: (
            isinstance(value, tuple)
            if condition is tuple
            else original_is_type(value, condition)
        ),
    )
    set_result = u.Validation._normalize_set_helper({"seed"})
    assert any(isinstance(item, str) for item in set_result)
    monkeypatch.setattr(
        "flext_core._utilities.validation.FlextUtilitiesValidation._normalize_component",
        staticmethod(original_normalize),
    )

    class _PipelineFailure:
        is_success = False
        is_failure = True
        value = False
        error = "pipeline"

    pipe = u.Validation.validate_pipeline("x", [lambda _: _PipelineFailure()])
    assert pipe.is_failure

    v_ok = _Validator(ok=True, description="ok")
    assert u.Validation._guard_check_validator("x", v_ok, "Field", None) is None

    assert (
        u.Validation._guard_check_predicate("x", lambda _: False, "Field", "e") == "e"
    )

    def _raise(_: object) -> bool:
        msg = "bad"
        raise RuntimeError(msg)

    assert u.Validation._guard_check_predicate("x", _raise, "Field", "e2") == "e2"

    class _AlwaysValidatorMeta(type):
        def __instancecheck__(self, instance: object) -> bool:
            return isinstance(instance, _Validator)

    class _AlwaysValidator(metaclass=_AlwaysValidatorMeta):
        pass

    monkeypatch.setattr(
        "flext_core._utilities.validation.p.ValidatorSpec",
        _AlwaysValidator,
    )
    assert u.Validation._guard_check_condition("x", _Validator(ok=False), "Field", None)

    assert u.Validation._guard_non_empty({"k": 1}, "Value must be").is_success
    assert u.Validation._guard_non_empty([1], "Value must be").is_success

    assert u.Validation.ResultHelpers.empty(r[list[int]].ok([])) is True
    assert u.Validation.ResultHelpers.ends("abc", "") is True

    assert u.Validation._ensure_to_list(None, None) == []
    assert u.Validation._ensure_to_list("x", None) == ["x"]
    assert u.Validation._ensure_to_dict(None, None) == {}
    assert u.Validation._ensure_to_dict("x", None) == {"value": "x"}

    assert u.Validation.ensure([1, "a"], target_type="str_list") == ["1", "a"]
    assert u.Validation.ensure("x", target_type="str_list", default=[1, 2]) == ["x"]
    assert u.Validation.ensure({"k": 1}, target_type="auto") == {"k": 1}

    class _AdapterBoom:
        def __init__(self, type_: object) -> None:
            self.type_ = type_

        def validate_json(self, json_str: str) -> object:
            msg = "json boom"
            raise RuntimeError(msg)

    monkeypatch.setattr(
        "flext_core._utilities.validation.PydanticTypeAdapter",
        _AdapterBoom,
    )
    parse_fail = u.Validation.TypeAdapter.parse_json('{"k": 1}', dict[str, int])
    assert parse_fail.is_failure
