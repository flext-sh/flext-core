"""Tests for r to achieve full coverage.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import MutableSequence, Sequence
from typing import cast

from flext_core import r
from flext_tests import tm
from tests import t

from ._models_impl import _ErrorsModel, _PlainErrorModel, _TargetModel


class _ValidationLikeError(ValueError):
    def errors(self) -> Sequence[t.ContainerMapping]:
        return [{"loc": ["value"], "msg": "bad value"}]


def test_validation_like_error_structure() -> None:
    err = _ValidationLikeError("validation")
    details = err.errors()
    tm.that(details[0]["msg"], eq="bad value")


def test_type_guards_result() -> None:
    ok_res = r[t.Container].ok("ok")
    fail_res = r[t.Container].fail("x")
    tm.that(r.is_success_result(ok_res), eq=True)
    tm.that(r.is_failure_result(fail_res), eq=True)


def test_init_fallback_and_lazy_returns_result_property() -> None:
    fallback = r[int](value=9, is_success=True)
    tm.ok(fallback)
    tm.that(fallback.value, eq=9)
    lazy_ok = r[int](value=5, is_success=True)
    assert lazy_ok._result is None
    _ = lazy_ok._returns_result
    lazy_fail = r[int](error="nope", is_success=False)
    assert lazy_fail._result is None
    _ = lazy_fail._returns_result


def test_map_flat_map_and_then_paths() -> None:
    mapped_fail = r[int].ok(2).map(lambda _: (_ for _ in ()).throw(ValueError("m")))
    tm.fail(mapped_fail)
    tm.that(mapped_fail.error, eq="m")
    runtime_ok: r[int] = cast("r[int]", r[int].ok(20))
    flat_ok: r[int] = r[int].ok(1).flat_map(lambda _: runtime_ok)
    tm.ok(flat_ok)
    tm.that(flat_ok.value, eq=20)
    runtime_fail: r[int] = cast(
        "r[int]",
        r(
            error="inner",
            is_success=False,
            error_code=None,
            error_data=None,
        ),
    )
    flat_fail: r[int] = r[int].ok(1).flat_map(lambda _: runtime_fail)
    tm.fail(flat_fail)
    tm.that(flat_fail.error, eq="inner")
    and_then_ok: r[str] = r[int].ok(3).flat_map(lambda v: r[str].ok(str(v)))
    tm.ok(and_then_ok)
    tm.that(and_then_ok.value, eq="3")


def test_recover_tap_and_tap_error_paths() -> None:
    tm.that(r[int].ok(1).recover(lambda _e: 99).value, eq=1)
    failed_for_recover: r[int] = cast("r[int]", r.fail("bad"))
    recovered: r[int] = failed_for_recover.recover(lambda _e: 42)
    tm.ok(recovered)
    tm.that(recovered.value, eq=42)
    seen: MutableSequence[int] = []
    _ = r[int].ok(7).tap(lambda v: seen.append(v))
    tm.that(seen, eq=[7])
    err_seen: MutableSequence[str] = []
    _ = r[int].fail("boom").tap_error(lambda e: err_seen.append(e))
    tm.that(err_seen, eq=["boom"])


def test_from_validation_and_to_model_paths() -> None:
    success_result = r[_TargetModel].from_validation(
        {"value": 10},
        _TargetModel,
    )
    tm.ok(success_result)
    tm.that(success_result.value.value, eq=10)
    err_result = r[_ErrorsModel].from_validation(
        {"value": "x"},
        _ErrorsModel,
    )
    tm.fail(err_result)
    tm.that((err_result.error or ""), has="Validation failed")
    tm.that((err_result.error or ""), has="bad value")
    plain_result = r[_PlainErrorModel].from_validation(
        {"value": "x"},
        _PlainErrorModel,
    )
    tm.fail(plain_result)
    tm.that((plain_result.error or ""), has="plain boom")
    failure_to_model = (
        r[t.IntMapping]
        .fail("already failed")
        .to_model(
            _TargetModel,
        )
    )
    tm.fail(failure_to_model)
    tm.that(failure_to_model.error, eq="already failed")
    success_to_model = (
        r[t.IntMapping]
        .ok({"value": 9})
        .to_model(
            _TargetModel,
        )
    )
    tm.ok(success_to_model)
    tm.that(success_to_model.value.value, eq=9)
    invalid_to_model = (
        r[t.StrMapping]
        .ok({"value": "bad"})
        .to_model(
            _TargetModel,
        )
    )
    tm.fail(invalid_to_model)
    tm.that((invalid_to_model.error or ""), has="Model conversion failed")


def test_lash_runtime_result_paths() -> None:
    runtime_ok2: r[int] = r[int].ok(99)
    failed_for_lash: r[int] = cast("r[int]", r.fail("x"))
    lash_ok: r[int] = failed_for_lash.lash(lambda _e: runtime_ok2)
    tm.ok(lash_ok)
    tm.that(lash_ok.value, eq=99)
    runtime_fail2: r[int] = r(
        error="recovery failed",
        is_success=False,
        error_code=None,
        error_data=None,
    )
    failed_for_lash_2: r[int] = cast("r[int]", r.fail("x"))
    lash_fail: r[int] = failed_for_lash_2.lash(lambda _e: runtime_fail2)
    tm.fail(lash_fail)
    tm.that(lash_fail.error, eq="recovery failed")
