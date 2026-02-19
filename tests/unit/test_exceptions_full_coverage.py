from __future__ import annotations

from types import MappingProxyType

from flext_core import c, e, m, r, t, u


def test_base_error_normalize_metadata_merges_existing_metadata_model() -> None:
    err = e.BaseError("x", metadata={"a": 1})
    merged = e.BaseError._normalize_metadata(err.metadata, {"b": 2})
    assert merged.attributes["a"] == 1
    assert merged.attributes["b"] == 2
    assert c.Errors.UNKNOWN_ERROR
    assert isinstance(m.Metadata(attributes={}), m.Metadata)
    assert r[str].ok("ok").is_success
    assert u.Conversion.to_str(1) == "1"


def test_authentication_error_normalizes_extra_kwargs_into_context() -> None:
    err = e.AuthenticationError(
        "auth",
        auth_method="token",
        user_id="u1",
        ip="127.0.0.1",
    )
    assert err.to_dict()["ip"] == "127.0.0.1"


def test_not_found_error_correlation_id_selection_and_extra_kwargs() -> None:
    err_explicit = e.NotFoundError(
        "missing",
        resource_id="x",
        correlation_id="explicit-cid",
        metadata={"m": "n"},
        extra_value=3,
    )
    assert err_explicit.correlation_id == "explicit-cid"
    assert err_explicit.to_dict()["extra_value"] == 3

    err_preserved = e.NotFoundError(
        "missing",
        resource_id="x",
        **{"correlation_id": "preserved-cid"},
    )
    assert err_preserved.correlation_id == "preserved-cid"


def test_get_str_from_kwargs_and_merge_metadata_context_paths() -> None:
    kwargs: dict[str, t.MetadataAttributeValue] = {"value": 123}
    assert e._get_str_from_kwargs(kwargs, "value") == "123"
    assert e._get_str_from_kwargs(kwargs, "missing") is None

    context: dict[str, t.MetadataAttributeValue] = {}

    err = e.BaseError("meta", metadata={"x": 1})
    meta = err.metadata
    config_attrs = t.ConfigMap.model_validate({"k": 1, "z": "q"})
    object.__setattr__(meta, "attributes", config_attrs)
    e._merge_metadata_into_context(context, meta)
    assert context["k"] == 1
    assert context["z"] == "q"

    context2: dict[str, t.MetadataAttributeValue] = {}
    mapping_obj = MappingProxyType({"p": 7})
    e._merge_metadata_into_context(context2, mapping_obj)
    assert context2["p"] == 7


def test_exceptions_uncovered_metadata_paths() -> None:
    metadata = e.BaseError("x", metadata={"a": 1}).metadata
    same = e.BaseError._normalize_metadata(metadata, {})
    assert same is metadata

    import flext_core.exceptions as exceptions_module

    raw = exceptions_module._Metadata(attributes={"x": 1})
    object.__setattr__(raw, "attributes", {"x": 1, "y": "z"})
    merged: dict[str, t.MetadataAttributeValue] = {}
    e._merge_metadata_into_context(merged, raw)
    assert merged["x"] == 1
    assert merged["y"] == "z"
