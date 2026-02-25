from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Never, Protocol, cast

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


class _ValidationLike(Protocol):
    def validate_email(self, v: str) -> str: ...

    def _validation_failure_message(self, error: str | None, fallback: str) -> str: ...

    def validate_performance(
        self,
        model: BaseModel,
        *,
        max_validation_time_ms: int | None = None,
    ) -> r[bool]: ...

    def validate_batch(
        self,
        items: Sequence[int],
        validator: Callable[[int], r[int]],
        *,
        fail_fast: bool = False,
    ) -> r[bool]: ...

    def validate_cqrs_patterns(
        self,
        payload: int,
        pattern_type: str,
        validators: Sequence[Callable[[int], r[int]]],
    ) -> r[bool]: ...

    def _validate_event_structure(self, payload: t.ConfigMapValue) -> r[bool]: ...

    def _event_has_attr(self, payload: t.ConfigMapValue, attr: str) -> bool: ...

    def validate_entity_relationships(
        self, entity: t.GeneralValueType | None
    ) -> r[bool]: ...

    def validate_uri(self, uri: str | None) -> r[str]: ...

    def validate_port_number(self, port: int | None) -> r[int]: ...

    def validate_required_string(self, value: str | None) -> r[str]: ...

    def validate_choice(self, value: str, choices: Sequence[str]) -> r[str]: ...

    def validate_length(
        self,
        value: str,
        *,
        max_length: int | None = None,
    ) -> r[str]: ...

    def validate_pattern(self, value: str, pattern: str) -> r[str]: ...


validation = cast(_ValidationLike, m.Validation)


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

    msg: str = validation._validation_failure_message(None, "fallback")
    assert msg.endswith("fallback)")

    slow = validation.validate_performance(_Simple(x=1), max_validation_time_ms=-1)
    assert slow.is_failure

    ff = validation.validate_batch([1], _always_fail_int, fail_fast=True)
    assert ff.is_failure

    cqrs_ok = validation.validate_cqrs_patterns(1, "command", [_ok_int])
    assert cqrs_ok.is_success

    missing = validation._validate_event_structure({"event_type": "e"})
    assert missing.is_failure

    has_attr = validation._event_has_attr({"k": 1}, "k")
    assert has_attr is True

    none_entity = validation.validate_entity_relationships(None)
    assert none_entity.is_failure

    bad_uri = validation.validate_uri(None)
    assert bad_uri.is_failure

    bad_port = validation.validate_port_number(None)
    assert bad_port.is_failure

    req_none = validation.validate_required_string(None)
    assert req_none.is_failure

    bad_choice = validation.validate_choice("x", [])
    assert bad_choice.is_failure

    too_long = validation.validate_length("abc", max_length=1)
    assert too_long.is_failure

    bad_pattern = validation.validate_pattern("a", "[")
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

    failed_perf = validation.validate_performance(_BrokenDumpModel(value=1))
    assert failed_perf.is_failure

    assert validation._event_has_attr(t.ConfigMap(root={}), "missing") is False

    import flext_core._models.base as validation_models

    def _raise_bad_uri(_uri: str) -> Never:
        raise ValueError("bad uri")

    monkeypatch.setattr(validation_models, "urlparse", _raise_bad_uri)
    bad_uri = validation.validate_uri("http://ok")
    assert bad_uri.is_failure
