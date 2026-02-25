from __future__ import annotations

import importlib
from typing import cast

from pydantic import BaseModel
from returns.io import IOSuccess

from flext_core import FlextRuntime, r
from flext_core.typings import JsonValue

result_module = importlib.import_module("flext_core.result")


class _ValidationLikeError(Exception):
    def errors(self) -> list[dict[str, JsonValue]]:
        return [{"loc": ["value"], "msg": "bad value"}]


class _ErrorsModel(BaseModel):
    value: int

    @classmethod
    def model_validate(
        cls,
        obj: object,
        *,
        strict: bool | None = None,
        extra: str | None = None,
        from_attributes: bool | None = None,
        context: dict[str, object] | None = None,
        by_alias: bool | None = None,
        by_name: bool | None = None,
    ):
        _ = strict, extra, from_attributes, context, by_alias, by_name
        _ = obj
        raise _ValidationLikeError()


class _PlainErrorModel(BaseModel):
    value: int

    @classmethod
    def model_validate(
        cls,
        obj: object,
        *,
        strict: bool | None = None,
        extra: str | None = None,
        from_attributes: bool | None = None,
        context: dict[str, object] | None = None,
        by_alias: bool | None = None,
        by_name: bool | None = None,
    ):
        _ = strict, extra, from_attributes, context, by_alias, by_name
        _ = obj
        raise RuntimeError("plain boom")


class _TargetModel(BaseModel):
    value: int


def test_type_guards_and_protocol_name() -> None:
    ok_res = r[int].ok(1)
    fail_res: r[int] = cast("r[int]", r.fail("x"))
    assert result_module.is_success_result(ok_res)
    assert result_module.is_failure_result(fail_res)
    assert ok_res._protocol_name() == "FlextResult"


def test_init_fallback_and_lazy_result_property() -> None:
    fallback = r[int](value=9, is_success=True)
    assert fallback.is_success
    assert fallback.value == 9

    lazy_ok = r[int](value=5, is_success=True)
    assert lazy_ok._result is None
    _ = lazy_ok.result

    lazy_fail = r[int](error="nope", is_success=False)
    assert lazy_fail._result is None
    _ = lazy_fail.result


def test_map_flat_map_and_then_paths() -> None:
    mapped_fail = r[int].ok(2).map(lambda _: (_ for _ in ()).throw(ValueError("m")))
    assert mapped_fail.is_failure
    assert mapped_fail.error == "m"

    runtime_ok = FlextRuntime.RuntimeResult(value=20, is_success=True)
    flat_ok = r[int].ok(1).flat_map(lambda _: runtime_ok)
    assert flat_ok.is_success
    assert flat_ok.value == 20

    runtime_fail: FlextRuntime.RuntimeResult[int] = FlextRuntime.RuntimeResult(
        error="inner", is_success=False
    )
    flat_fail = r[int].ok(1).flat_map(lambda _: runtime_fail)
    assert flat_fail.is_failure
    assert flat_fail.error == "inner"

    and_then_ok = r[int].ok(3).and_then(lambda v: r[str].ok(str(v)))
    assert and_then_ok.is_success
    assert and_then_ok.value == "3"


def test_recover_tap_and_tap_error_paths() -> None:
    assert r[int].ok(1).recover(lambda _e: 99).value == 1
    failed_for_recover: r[int] = cast("r[int]", r.fail("bad"))
    recovered: r[int] = failed_for_recover.recover(lambda _e: 42)
    assert recovered.is_success
    assert recovered.value == 42

    seen: list[int] = []
    _ = r[int].ok(7).tap(lambda v: seen.append(v))
    assert seen == [7]

    err_seen: list[str] = []
    _ = r[int].fail("boom").tap_error(lambda e: err_seen.append(e))
    assert err_seen == ["boom"]


def test_from_validation_and_to_model_paths() -> None:
    success_result = r[_TargetModel].from_validation({"value": 10}, _TargetModel)
    assert success_result.is_success
    assert success_result.value.value == 10

    err_result = r[_ErrorsModel].from_validation({"value": "x"}, _ErrorsModel)
    assert err_result.is_failure
    assert "Validation failed" in (err_result.error or "")
    assert "bad value" in (err_result.error or "")

    plain_result = r[_PlainErrorModel].from_validation({"value": "x"}, _PlainErrorModel)
    assert plain_result.is_failure
    assert "plain boom" in (plain_result.error or "")

    failure_to_model = r[dict[str, int]].fail("already failed").to_model(_TargetModel)
    assert failure_to_model.is_failure
    assert failure_to_model.error == "already failed"

    success_to_model = r[dict[str, int]].ok({"value": 9}).to_model(_TargetModel)
    assert success_to_model.is_success
    assert success_to_model.value.value == 9

    invalid_to_model = r[dict[str, str]].ok({"value": "bad"}).to_model(_TargetModel)
    assert invalid_to_model.is_failure
    assert "Model conversion failed" in (invalid_to_model.error or "")


def test_lash_runtime_result_and_from_io_result_fallback() -> None:
    runtime_ok = FlextRuntime.RuntimeResult(value=99, is_success=True)
    failed_for_lash: r[int] = cast("r[int]", r.fail("x"))
    lash_ok: r[int] = failed_for_lash.lash(
        lambda _e: cast("FlextRuntime.RuntimeResult[int]", runtime_ok)
    )
    assert lash_ok.is_success
    assert lash_ok.value == 99

    runtime_fail: FlextRuntime.RuntimeResult[int] = FlextRuntime.RuntimeResult(
        error="recovery failed", is_success=False
    )
    failed_for_lash_2: r[int] = cast("r[int]", r.fail("x"))
    lash_fail: r[int] = failed_for_lash_2.lash(lambda _e: runtime_fail)
    assert lash_fail.is_failure
    assert lash_fail.error == "recovery failed"

    good = r[int].from_io_result(IOSuccess(1))
    assert good.is_success

    invalid: r[int] = r[int].from_io_result(cast("object", object()))
    assert invalid.is_failure
    assert invalid.error == "Invalid IOResult structure"
