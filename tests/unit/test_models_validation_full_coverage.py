from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Never, cast

from pydantic import BaseModel
from pytest import MonkeyPatch

from flext_core import c, m, r, t, u


class _Simple(BaseModel):
    x: int


class _BadInvariant:
    def check_invariants(self) -> None:
        raise ValueError("bad invariant")


class _BrokenDumpModel(BaseModel):
    value: int = 1


def _always_fail_int(_: int) -> r[int]:
    return r[int].fail("x")


def _ok_int(value: int) -> r[int]:
    return r[int].ok(value)


def test_models_validation_branch_paths() -> None:
    assert c.Errors.UNKNOWN_ERROR
    assert isinstance(m.Categories(), m.Categories)
    assert isinstance(_BadInvariant(), _BadInvariant)
    assert r[int].ok(1).is_success
    assert isinstance(t.ConfigMap.model_validate({"k": 1}), t.ConfigMap)
    assert u.Conversion.to_str(1) == "1"

    validation_failure_message = cast(
        Callable[[str | None, str], str],
        getattr(m.Validation, "_validation_failure_message"),
    )
    validate_performance = cast(
        Callable[..., r[bool]],
        getattr(m.Validation, "validate_performance"),
    )
    validate_batch = cast(
        Callable[..., r[bool]],
        getattr(m.Validation, "validate_batch"),
    )
    validate_cqrs_patterns = cast(
        Callable[..., r[bool]],
        getattr(m.Validation, "validate_cqrs_patterns"),
    )
    validate_event_structure = cast(
        Callable[[t.ConfigMapValue], r[bool]],
        getattr(m.Validation, "_validate_event_structure"),
    )
    event_has_attr = cast(
        Callable[[t.ConfigMapValue, str], bool],
        getattr(m.Validation, "_event_has_attr"),
    )
    validate_entity_relationships = cast(
        Callable[[t.GeneralValueType | None], r[bool]],
        getattr(m.Validation, "validate_entity_relationships"),
    )
    validate_uri = cast(
        Callable[[str | None], r[str]], getattr(m.Validation, "validate_uri")
    )
    validate_port_number = cast(
        Callable[[int | None], r[int]],
        getattr(m.Validation, "validate_port_number"),
    )
    validate_required_string = cast(
        Callable[[str | None], r[str]],
        getattr(m.Validation, "validate_required_string"),
    )
    validate_choice = cast(
        Callable[[str, Sequence[str]], r[str]],
        getattr(m.Validation, "validate_choice"),
    )
    validate_length = cast(
        Callable[..., r[str]], getattr(m.Validation, "validate_length")
    )
    validate_pattern = cast(
        Callable[[str, str], r[str]],
        getattr(m.Validation, "validate_pattern"),
    )

    msg: str = validation_failure_message(None, "fallback")
    assert msg.endswith("fallback)")

    slow = validate_performance(_Simple(x=1), max_validation_time_ms=-1)
    assert slow.is_failure

    ff = validate_batch([1], _always_fail_int, fail_fast=True)
    assert ff.is_failure

    cqrs_ok = validate_cqrs_patterns(1, "command", [_ok_int])
    assert cqrs_ok.is_success

    missing = validate_event_structure({"event_type": "e"})
    assert missing.is_failure

    has_attr = event_has_attr({"k": 1}, "k")
    assert has_attr is True

    none_entity = validate_entity_relationships(None)
    assert none_entity.is_failure

    bad_uri = validate_uri(None)
    assert bad_uri.is_failure

    bad_port = validate_port_number(None)
    assert bad_port.is_failure

    req_none = validate_required_string(None)
    assert req_none.is_failure

    bad_choice = validate_choice("x", [])
    assert bad_choice.is_failure

    too_long = validate_length("abc", max_length=1)
    assert too_long.is_failure

    bad_pattern = validate_pattern("a", "[")
    assert bad_pattern.is_failure


def test_models_validation_uncovered_exception_and_event_paths(
    monkeypatch: MonkeyPatch,
) -> None:
    def _raise_model_dump(
        self: _BrokenDumpModel,
        **kwargs: t.GeneralValueType,
    ) -> dict[str, t.GeneralValueType]:
        _ = (self, kwargs)
        raise RuntimeError("dump failed")

    monkeypatch.setattr(_BrokenDumpModel, "model_dump", _raise_model_dump)

    validate_performance = cast(
        Callable[..., r[bool]],
        getattr(m.Validation, "validate_performance"),
    )
    event_has_attr = cast(
        Callable[[t.ConfigMapValue, str], bool],
        getattr(m.Validation, "_event_has_attr"),
    )
    validate_uri = cast(
        Callable[[str | None], r[str]], getattr(m.Validation, "validate_uri")
    )

    failed_perf = validate_performance(_BrokenDumpModel(value=1))
    assert failed_perf.is_failure

    assert event_has_attr(t.ConfigMap(root={}), "missing") is False

    import flext_core._models.base as validation_models

    def _raise_bad_uri(_uri: str) -> Never:
        raise ValueError("bad uri")

    monkeypatch.setattr(validation_models, "urlparse", _raise_bad_uri)
    bad_uri = validate_uri("http://ok")
    assert bad_uri.is_failure
