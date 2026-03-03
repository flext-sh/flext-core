"""FlextResult (r) — exercises ALL public API methods with golden file validation."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel
from returns.io import IOFailure, IOSuccess
from returns.maybe import Nothing, Some

from flext_core import FlextExceptions, FlextResult, r, u

_RESULTS: list[str] = []


def _check(label: str, value: object) -> None:
    _RESULTS.append(f"{label}: {_ser(value)}")


def _section(name: str) -> None:
    if _RESULTS:
        _RESULTS.append("")
    _RESULTS.append(f"[{name}]")


def _ser(v: object) -> str:
    if v is None:
        return "None"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        return repr(v)
    if u.is_list(v):
        return "[" + ", ".join(_ser(x) for x in v) + "]"
    if u.is_dict_like(v):
        pairs = ", ".join(
            f"{_ser(k)}: {_ser(val)}"
            for k, val in sorted(v.items(), key=lambda kv: str(kv[0]))
        )
        return "{" + pairs + "}"
    if isinstance(v, type):
        return v.__name__
    return type(v).__name__


def _verify() -> None:
    actual = "\n".join(_RESULTS).strip() + "\n"
    me = Path(__file__)
    expected_path = me.with_suffix(".expected")
    n = sum(1 for line in _RESULTS if ": " in line and not line.startswith("["))
    if expected_path.exists():
        expected = expected_path.read_text(encoding="utf-8")
        if actual == expected:
            sys.stdout.write(f"PASS: {me.stem} ({n} checks)\n")
        else:
            actual_path = me.with_suffix(".actual")
            actual_path.write_text(actual, encoding="utf-8")
            sys.stdout.write(
                f"FAIL: {me.stem} — diff {expected_path.name} {actual_path.name}\n"
            )
            sys.exit(1)
    else:
        expected_path.write_text(actual, encoding="utf-8")
        sys.stdout.write(f"GENERATED: {expected_path.name} ({n} checks)\n")


class _Person(BaseModel):
    name: str
    age: int


@dataclass
class _Handle:
    value: int
    cleaned: bool = False


def _bind_probe(result_obj: object, delta: int) -> object:
    try:
        method = getattr(result_obj, "bind")
    except AttributeError as exc:
        return f"AttributeError:{exc}"
    if not callable(method):
        return "bind-not-callable"
    try:
        return method(lambda n: r[int].ok(n + delta))
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        return f"{type(exc).__name__}:{exc}"


def _bind_status(value: object) -> object:
    if isinstance(value, FlextResult):
        return {
            "is_success": value.is_success,
            "error": value.error,
            "unwrap_or": value.unwrap_or(-1),
        }
    return value


def demo_factories_and_guards() -> None:
    """Exercise factory constructors, decorator wrapping, and type guards."""
    _section("factories_and_guards")

    ok_result = r[int].ok(10)
    _check("ok.value", ok_result.value)

    failed = r[int].fail(
        "boom",
        error_code="E_DEMO",
        error_data={"stage": "factory"},
        exception=ValueError("bad input"),
    )
    _check("fail.error", failed.error)
    _check("fail.error_code", failed.error_code)
    _check("fail.error_data", failed.error_data)

    @r.safe
    def parse_int(value: str) -> int:
        return int(value)

    safe_ok = parse_int("42")
    safe_fail = parse_int("x")
    _check("safe.success.unwrap_or", safe_ok.unwrap_or(0))
    _check("safe.failure.error", safe_fail.error)

    callable_ok = r[str].create_from_callable(lambda: "created")
    callable_fail = r[str].create_from_callable(
        lambda: (_ for _ in ()).throw(RuntimeError("callable failed")),
        error_code="E_CALL",
    )
    callable_none = r[str].create_from_callable(lambda: None, error_code="E_NONE")
    _check("create_from_callable.success", callable_ok.unwrap_or("fallback"))
    _check("create_from_callable.failure.code", callable_fail.error_code)
    _check("create_from_callable.none.error", callable_none.error)

    _check("is_success_result.ok", r.is_success_result(ok_result))
    _check("is_success_result.fail", r.is_success_result(failed))
    _check("is_failure_result.fail", r.is_failure_result(failed))
    _check("is_failure_result.string", r.is_failure_result("plain"))


def demo_properties_and_unwrap() -> None:
    """Exercise result properties and unwrap behavior for both states."""
    _section("properties_and_unwrap")

    success = r[str].ok("value")
    failure = r[str].fail("missing", error_code="E_PROP", error_data={"x": 1})

    _check("prop.success.is_success", success.is_success)
    _check("prop.success.is_failure", success.is_failure)
    _check("prop.failure.is_success", failure.is_success)
    _check("prop.failure.is_failure", failure.is_failure)
    _check("prop.success.value", success.value)
    _check("prop.success.data", success.data)
    _check("prop.success.result_self", success.result is success)
    _check("prop.failure.error", failure.error)
    _check("prop.failure.error_code", failure.error_code)
    _check("prop.failure.error_data", failure.error_data)

    _check("unwrap.success", success.unwrap())
    try:
        failure.unwrap()
        _check("unwrap.failure.raises", False)
    except RuntimeError as exc:
        _check("unwrap.failure.raises", True)
        _check("unwrap.failure.type", type(exc).__name__)

    _check("unwrap_or.success", success.unwrap_or("default"))
    _check("unwrap_or.failure", failure.unwrap_or("default"))


def demo_transform_chain_and_recover() -> None:
    """Exercise transformation and chaining APIs for success/failure paths."""
    _section("transform_chain_and_recover")

    base_ok = r[int].ok(5)
    base_fail = r[int].fail("bad-number")

    _check("map.success", base_ok.map(lambda n: n + 1).unwrap_or(-1))
    _check("map.failure", base_fail.map(lambda n: n + 1).is_failure)
    _check(
        "map.exception_to_failure",
        base_ok.map(lambda _: (_ for _ in ()).throw(ValueError("map exploded"))).error,
    )

    _check(
        "flat_map.success",
        base_ok.flat_map(lambda n: r[int].ok(n * 2)).unwrap_or(-1),
    )
    _check(
        "flat_map.failure",
        base_ok.flat_map(lambda _: r[int].fail("flat failed")).error,
    )
    _check(
        "and_then.success", base_ok.and_then(lambda n: r[int].ok(n - 2)).unwrap_or(-1)
    )
    _check("and_then.failure", base_fail.and_then(lambda n: r[int].ok(n)).is_failure)

    bind_ok = _bind_probe(base_ok, 3)
    bind_fail = _bind_probe(base_fail, 3)
    _check("bind.success", _bind_status(bind_ok))
    _check("bind.failure", _bind_status(bind_fail))

    _check("alt.success_unchanged", base_ok.alt(lambda e: f"alt:{e}").unwrap_or(-1))
    _check("alt.failure_changed", base_fail.alt(lambda e: f"alt:{e}").error)
    _check(
        "map_error.failure_changed",
        base_fail.map_error(lambda e: f"mapped:{e}").error,
    )
    _check(
        "map_error.success_unchanged",
        base_ok.map_error(lambda e: f"mapped:{e}").unwrap_or(-1),
    )

    _check(
        "lash.failure_recovered",
        base_fail.lash(lambda e: r[int].ok(len(e))).unwrap_or(-1),
    )
    _check(
        "lash.success_unchanged",
        base_ok.lash(lambda _: r[int].ok(99)).unwrap_or(-1),
    )
    _check("recover.failure", base_fail.recover(lambda e: len(e)).unwrap_or(-1))
    _check("recover.success", base_ok.recover(lambda _: 0).unwrap_or(-1))


def demo_side_effects_and_folds() -> None:
    """Exercise side-effect helpers, map_or, fold, and filter."""
    _section("side_effects_and_folds")

    side_effects: list[int] = []
    error_effects: list[str] = []

    ok_value = r[int].ok(7)
    fail_value = r[int].fail("oops")

    _check("tap.success", ok_value.tap(lambda n: side_effects.append(n)).is_success)
    _check("tap.failure", fail_value.tap(lambda n: side_effects.append(n)).is_failure)
    _check("tap.log", side_effects)

    _check(
        "tap_error.failure",
        fail_value.tap_error(lambda e: error_effects.append(e)).is_failure,
    )
    _check(
        "tap_error.success",
        ok_value.tap_error(lambda e: error_effects.append(e)).is_success,
    )
    _check("tap_error.log", error_effects)

    _check("map_or.success_default", ok_value.map_or(0))
    _check("map_or.failure_default", fail_value.map_or(0))
    _check("map_or.success_func", ok_value.map_or("none", lambda n: f"n={n}"))
    _check("map_or.failure_func", fail_value.map_or("none", lambda n: f"n={n}"))

    _check(
        "fold.success",
        ok_value.fold(on_failure=lambda e: f"fail:{e}", on_success=lambda n: f"ok:{n}"),
    )
    _check(
        "fold.failure",
        fail_value.fold(
            on_failure=lambda e: f"fail:{e}",
            on_success=lambda n: f"ok:{n}",
        ),
    )

    _check("filter.success_pass", ok_value.filter(lambda n: n > 0).is_success)
    _check("filter.success_fail", ok_value.filter(lambda n: n < 0).is_failure)
    _check("filter.failure_stays", fail_value.filter(lambda n: n > 0).is_failure)


def demo_conversions_and_models() -> None:
    """Exercise conversion APIs and Pydantic model integration."""
    _section("conversions_and_models")

    ok_value = r[int].ok(8)
    fail_value = r[int].fail("io-err")

    _check("to_maybe.success", ok_value.to_maybe().unwrap())
    _check("to_maybe.failure", fail_value.to_maybe().value_or(123))

    _check("from_maybe.some", r[str].from_maybe(Some("x"), "empty").unwrap_or("none"))
    _check("from_maybe.nothing", r[str].from_maybe(Nothing, "empty").error)

    io_value = ok_value.to_io()
    _check("to_io.success.type", type(io_value).__name__)
    try:
        fail_value.to_io()
        _check("to_io.failure.raises", False)
    except FlextExceptions.ValidationError as exc:
        _check("to_io.failure.raises", True)
        _check("to_io.failure.type", type(exc).__name__)

    _check("to_io_result.success.type", type(ok_value.to_io_result()).__name__)
    _check("to_io_result.failure.type", type(fail_value.to_io_result()).__name__)

    from_io_ok = r[int].from_io_result(IOSuccess(11))
    from_io_fail = r[int].from_io_result(IOFailure("x"))
    from_io_bad = r[int].from_io_result("bad-io-result")
    _check("from_io_result.success", from_io_ok.is_success)
    _check("from_io_result.failure", from_io_fail.error)
    _check("from_io_result.invalid", from_io_bad.error)

    valid_data = {"name": "Ada", "age": 30}
    invalid_data = {"name": "Ada", "age": "bad"}

    from_validation_ok = r[_Person].from_validation(valid_data, _Person)
    from_validation_fail = r[_Person].from_validation(invalid_data, _Person)
    _check("from_validation.success", from_validation_ok.is_success)
    _check("from_validation.failure", from_validation_fail.is_failure)

    _check(
        "to_model.success",
        r[dict[str, object]].ok(valid_data).to_model(_Person).value.age,
    )
    _check(
        "to_model.from_failure",
        r[dict[str, object]].fail("missing").to_model(_Person).error,
    )
    _check(
        "to_model.validation_failure",
        r[dict[str, object]].ok(invalid_data).to_model(_Person).is_failure,
    )


def demo_collections_and_resource() -> None:
    """Exercise collection helpers and resource management wrapper."""
    _section("collections_and_resource")

    def to_even(n: int) -> FlextResult[int]:
        if n % 2 == 0:
            return r[int].ok(n)
        return r[int].fail(f"odd:{n}")

    _check(
        "traverse.success",
        r[list].traverse([2, 4, 6], to_even, fail_fast=True).unwrap_or([]),
    )
    _check(
        "traverse.fail_fast",
        r[list].traverse([2, 3, 4], to_even, fail_fast=True).error,
    )
    _check(
        "traverse.collect",
        r[list].traverse([1, 3, 5], to_even, fail_fast=False).error,
    )

    acc_ok = r.accumulate_errors(r[int].ok(1), r[int].ok(2))
    acc_fail = r.accumulate_errors(r[int].ok(1), r[int].fail("e1"), r[int].fail("e2"))
    _check("accumulate_errors.success", acc_ok.unwrap_or([]))
    _check("accumulate_errors.failure", acc_fail.error)

    cleaned_values: list[int] = []

    def make_handle() -> _Handle:
        return _Handle(value=21)

    def clean_handle(handle: _Handle) -> None:
        handle.cleaned = True
        cleaned_values.append(handle.value)

    success_resource = r[int].with_resource(
        make_handle,
        lambda handle: r[int].ok(handle.value * 2),
        cleanup=clean_handle,
    )
    failure_resource = r[int].with_resource(
        make_handle,
        lambda _: r[int].fail("resource op failed"),
        cleanup=clean_handle,
    )
    no_cleanup_resource = r[int].with_resource(
        make_handle,
        lambda handle: r[int].ok(handle.value + 1),
    )

    _check("with_resource.success", success_resource.unwrap_or(-1))
    _check("with_resource.failure", failure_resource.error)
    _check("with_resource.no_cleanup", no_cleanup_resource.unwrap_or(-1))
    _check("with_resource.cleanup_calls", cleaned_values)


def main() -> None:
    """Run all sections and verify against the golden file."""
    demo_factories_and_guards()
    demo_properties_and_unwrap()
    demo_transform_chain_and_recover()
    demo_side_effects_and_folds()
    demo_conversions_and_models()
    demo_collections_and_resource()
    _verify()


if __name__ == "__main__":
    main()
