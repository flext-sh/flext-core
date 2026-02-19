from __future__ import annotations

from pydantic import BaseModel

from flext_core import c, m, r, t, u


class _Simple(BaseModel):
    x: int


class _BadInvariant:
    def check_invariants(self) -> None:
        raise ValueError("bad invariant")


class _BrokenDumpModel(BaseModel):
    value: int = 1

    def model_dump(self, **kwargs: object) -> dict[str, object]:
        _ = kwargs
        raise RuntimeError("dump failed")


def test_models_validation_branch_paths() -> None:
    assert c.Errors.UNKNOWN_ERROR
    assert isinstance(m.Categories(), m.Categories)
    assert r[int].ok(1).is_success
    assert isinstance(t.ConfigMap.model_validate({"k": 1}), t.ConfigMap)
    assert u.Conversion.to_str(1) == "1"

    msg = m.Validation._validation_failure_message(None, "fallback")
    assert msg.endswith("fallback)")

    slow = m.Validation.validate_performance(_Simple(x=1), max_validation_time_ms=-1)
    assert slow.is_failure

    ff = m.Validation.validate_batch([1], lambda _: r[int].fail("x"), fail_fast=True)
    assert ff.is_failure

    cqrs_ok = m.Validation.validate_cqrs_patterns(
        1, "command", [lambda v: r[int].ok(v)]
    )
    assert cqrs_ok.is_success

    missing = m.Validation._validate_event_structure({"event_type": "e"})
    assert missing.is_failure

    has_attr = m.Validation._event_has_attr({"k": 1}, "k")
    assert has_attr is True

    none_entity = m.Validation.validate_entity_relationships(None)
    assert none_entity.is_failure

    bad_uri = m.Validation.validate_uri(None)
    assert bad_uri.is_failure

    bad_port = m.Validation.validate_port_number(None)
    assert bad_port.is_failure

    req_none = m.Validation.validate_required_string(None)
    assert req_none.is_failure

    bad_choice = m.Validation.validate_choice("x", [])
    assert bad_choice.is_failure

    too_long = m.Validation.validate_length("abc", max_length=1)
    assert too_long.is_failure

    bad_pattern = m.Validation.validate_pattern("a", "[")
    assert bad_pattern.is_failure


def test_models_validation_uncovered_exception_and_event_paths(monkeypatch) -> None:
    failed_perf = m.Validation.validate_performance(_BrokenDumpModel(value=1))
    assert failed_perf.is_failure

    assert m.Validation._event_has_attr(object(), "missing") is False

    import flext_core._models.validation as validation_models

    monkeypatch.setattr(
        validation_models,
        "urlparse",
        lambda _uri: (_ for _ in ()).throw(ValueError("bad uri")),
    )
    bad_uri = m.Validation.validate_uri("http://ok")
    assert bad_uri.is_failure
