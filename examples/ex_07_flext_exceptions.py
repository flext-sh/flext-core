"""FlextExceptions (e) golden-file example for public API coverage."""

from __future__ import annotations

import sys
from pathlib import Path

from flext_core import FlextExceptions, c, e, r, t, u

type _SerValue = t.JsonPrimitive | list[_SerValue] | dict[str, _SerValue] | None

_RESULTS: list[str] = []


def _check(label: str, value: _SerValue) -> None:
    _RESULTS.append(f"{label}: {_ser(value)}")


def _section(name: str) -> None:
    if _RESULTS:
        _RESULTS.append("")
    _RESULTS.append(f"[{name}]")


def _ser(v: _SerValue) -> str:
    if v is None:
        return "None"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        return repr(v)
    if u.is_list(v):
        return "[" + ", ".join(_ser(item) for item in v) + "]"
    if u.is_dict_like(v):
        pairs = ", ".join(
            f"{_ser(k)}: {_ser(value)}"
            for k, value in sorted(v.items(), key=lambda kv: str(kv[0]))
        )
        return "{" + pairs + "}"
    return type(v).__name__


def _verify() -> None:
    actual = "\n".join(_RESULTS).strip() + "\n"
    me = Path(__file__)
    expected_path = me.with_suffix(".expected")
    checks = sum(1 for line in _RESULTS if ": " in line and not line.startswith("["))
    if expected_path.exists():
        expected = expected_path.read_text(encoding="utf-8")
        if expected == actual:
            sys.stdout.write(f"PASS: {me.stem} ({checks} checks)\n")
            return
        actual_path = me.with_suffix(".actual")
        actual_path.write_text(actual, encoding="utf-8")
        sys.stdout.write(
            f"FAIL: {me.stem} — diff {expected_path.name} {actual_path.name}\n"
        )
        sys.exit(1)
    expected_path.write_text(actual, encoding="utf-8")
    sys.stdout.write(f"GENERATED: {expected_path.name} ({checks} checks)\n")


def _exercise_base_error() -> None:
    _section("base_error")

    base = e.BaseError(
        "base boom",
        error_code="E_BASE",
        context={"scope": "demo"},
        metadata={"channel": "example"},
        correlation_id="corr-base-1",
        auto_correlation=False,
        auto_log=False,
        operation="create",
    )
    _check("base.class", type(base).__name__)
    _check("base.message", base.message)
    _check("base.error_code", base.error_code)
    _check("base.correlation_id", base.correlation_id)
    _check("base.auto_log", base.auto_log)
    _check("base.meta.scope", base.metadata.attributes.get("scope"))
    _check("base.meta.channel", base.metadata.attributes.get("channel"))
    _check("base.meta.operation", base.metadata.attributes.get("operation"))

    payload = base.to_dict()
    _check("base.to_dict.type", payload.get("error_type"))
    _check("base.to_dict.message", payload.get("message"))
    _check("base.to_dict.error_code", payload.get("error_code"))
    _check("base.to_dict.has_timestamp", "timestamp" in payload)

    auto_corr = e.BaseError(
        "auto corr",
        error_code="E_AUTO",
        auto_correlation=True,
        auto_log=False,
    )
    _check(
        "base.auto_corr.prefix",
        auto_corr.correlation_id is not None
        and auto_corr.correlation_id.startswith("exc_"),
    )


def _exercise_specific_exceptions() -> None:
    _section("subclasses")

    msg = "invalid"
    try:
        raise e.ValidationError(msg, field="email", value="bad")
    except e.ValidationError as exc:
        _check("ValidationError.field", exc.field)
        _check("ValidationError.value", exc.value)

    msg = "bad cfg"
    try:
        raise e.ConfigurationError(
            msg,
            config_key="db.host",
            config_source="env",
        )
    except e.ConfigurationError as exc:
        _check("ConfigurationError.config_key", exc.config_key)
        _check("ConfigurationError.config_source", exc.config_source)

    msg = "down"
    try:
        raise e.ConnectionError(msg, host="127.0.0.1", port=5432, timeout=3.5)
    except e.ConnectionError as exc:
        _check("ConnectionError.host", exc.host)
        _check("ConnectionError.port", exc.port)
        _check("ConnectionError.timeout", exc.timeout)

    msg = "late"
    try:
        raise e.TimeoutError(msg, timeout_seconds=2.0, operation="sync")
    except e.TimeoutError as exc:
        _check("TimeoutError.timeout_seconds", exc.timeout_seconds)
        _check("TimeoutError.operation", exc.operation)

    msg = "auth fail"
    try:
        raise e.AuthenticationError(msg, auth_method="token", user_id="u-1")
    except e.AuthenticationError as exc:
        _check("AuthenticationError.auth_method", exc.auth_method)
        _check("AuthenticationError.user_id", exc.user_id)

    msg = "nope"
    try:
        raise e.AuthorizationError(
            msg,
            user_id="u-2",
            resource="invoice:7",
            permission="read",
        )
    except e.AuthorizationError as exc:
        _check("AuthorizationError.user_id", exc.user_id)
        _check("AuthorizationError.resource", exc.resource)
        _check("AuthorizationError.permission", exc.permission)

    msg = "missing"
    try:
        raise e.NotFoundError(msg, resource_type="User", resource_id="404")
    except e.NotFoundError as exc:
        _check("NotFoundError.resource_type", exc.resource_type)
        _check("NotFoundError.resource_id", exc.resource_id)

    msg = "conflict"
    try:
        raise e.ConflictError(
            msg,
            resource_type="User",
            resource_id="13",
            conflict_reason="duplicate",
        )
    except e.ConflictError as exc:
        _check("ConflictError.resource_type", exc.resource_type)
        _check("ConflictError.resource_id", exc.resource_id)
        _check("ConflictError.conflict_reason", exc.conflict_reason)

    msg = "slow down"
    try:
        raise e.RateLimitError(
            msg,
            limit=100,
            window_seconds=60,
            retry_after=1.5,
        )
    except e.RateLimitError as exc:
        _check("RateLimitError.limit", exc.limit)
        _check("RateLimitError.window_seconds", exc.window_seconds)
        _check("RateLimitError.retry_after", exc.retry_after)

    msg = "open"
    try:
        raise e.CircuitBreakerError(
            msg,
            service_name="billing",
            failure_count=5,
            reset_timeout=30.0,
        )
    except e.CircuitBreakerError as exc:
        _check("CircuitBreakerError.service_name", exc.service_name)
        _check("CircuitBreakerError.failure_count", exc.failure_count)
        _check("CircuitBreakerError.reset_timeout", exc.reset_timeout)

    msg = "wrong type"
    try:
        raise e.TypeError(msg, expected_type=str, actual_type=int)
    except e.TypeError as exc:
        _check(
            "TypeError.expected_type",
            exc.expected_type.__name__ if exc.expected_type is not None else None,
        )
        _check(
            "TypeError.actual_type",
            exc.actual_type.__name__ if exc.actual_type is not None else None,
        )

    msg = "failed op"
    try:
        raise e.OperationError(msg, operation="publish", reason="quota")
    except e.OperationError as exc:
        _check("OperationError.operation", exc.operation)
        _check("OperationError.reason", exc.reason)

    msg = "bad attr"
    try:
        raise e.AttributeAccessError(
            msg,
            attribute_name="secret",
            attribute_context="UserModel",
        )
    except e.AttributeAccessError as exc:
        _check("AttributeAccessError.attribute_name", exc.attribute_name)
        _check("AttributeAccessError.attribute_context", exc.attribute_context)


def _exercise_factories_and_helpers() -> None:
    _section("factories_helpers")

    created_validation = e.create_error("ValidationError", "factory validation")
    _check("create_error.ValidationError", type(created_validation).__name__)

    created_attribute = e.create_error("AttributeError", "factory attribute")
    _check("create_error.AttributeError", type(created_attribute).__name__)

    created_dynamic = e.create(
        "dynamic",
        error_code="E_DYNAMIC",
        field="username",
        value="",
        metadata={"caller": "example"},
        correlation_id="corr-dyn-1",
    )
    _check("create.type", type(created_dynamic).__name__)
    _check("create.error_code", created_dynamic.error_code)
    _check("create.metadata.caller", created_dynamic.metadata.attributes.get("caller"))
    _check("create.correlation_id", created_dynamic.correlation_id)

    prepared = e.prepare_exception_kwargs(
        {
            "correlation_id": "corr-prep",
            "metadata": {"k": "v"},
            "auto_log": True,
            "auto_correlation": True,
            "config": "cfg-a",
            "field": "existing",
            "custom": "x",
        },
        {"field": "forced"},
    )
    prep_corr, prep_metadata, prep_auto_log, prep_auto_corr, prep_config, prep_extra = (
        prepared
    )
    _check("prepare.correlation_id", prep_corr)
    _check("prepare.metadata_type", type(prep_metadata).__name__)
    _check("prepare.auto_log", prep_auto_log)
    _check("prepare.auto_correlation", prep_auto_corr)
    _check("prepare.config", prep_config)
    _check("prepare.extra.field", prep_extra.get("field"))
    _check("prepare.extra.custom", prep_extra.get("custom"))

    extracted_corr, extracted_meta = e.extract_common_kwargs({
        "correlation_id": "corr-ext",
        "metadata": {"x": "1"},
        "field": "f",
    })
    _check("extract.correlation_id", extracted_corr)
    _check("extract.metadata.kind", type(extracted_meta).__name__)

    instance_factory = FlextExceptions()
    instance_created = instance_factory(
        "from __call__",
        error_code="E_CALL",
        field="title",
        value="",
    )
    _check("__call__.type", type(instance_created).__name__)
    _check("__call__.error_code", instance_created.error_code)


def _exercise_metrics() -> None:
    _section("metrics")

    e.clear_metrics()
    e.record_exception(e.ValidationError)
    e.record_exception(e.ValidationError)
    e.record_exception(e.ConfigurationError)

    metrics = e.get_metrics()
    metric_map = metrics.root
    _check("metrics.total_exceptions", metric_map.get("total_exceptions"))
    counts_obj = metric_map.get("exception_counts")
    if u.is_dict_like(counts_obj):
        counts_map = dict(counts_obj)
        _check("metrics.validation_count", counts_map.get("ValidationError"))
        _check("metrics.configuration_count", counts_map.get("ConfigurationError"))
    else:
        _check("metrics.validation_count", "missing")
        _check("metrics.configuration_count", "missing")

    _check("metrics.summary_nonempty", bool(metric_map.get("exception_counts_summary")))
    _check("metrics.unique_types", metric_map.get("unique_exception_types"))

    e.clear_metrics()
    after_clear = e.get_metrics().root
    _check("metrics.cleared_total", after_clear.get("total_exceptions"))


def main() -> None:
    """Run all sections and validate the golden output."""
    _section("imports")
    _check("import.e_is_FlextExceptions", e is FlextExceptions)
    _check("import.r_ok", r[str].ok("ok").is_success)
    _check("import.constant", c.Errors.UNKNOWN_ERROR)

    msg = "boom"
    try:
        raise ValueError(msg)
    except ValueError as exc:
        _check("style.raise_msg", str(exc))

    _exercise_base_error()
    _exercise_specific_exceptions()
    _exercise_factories_and_helpers()
    _exercise_metrics()
    _verify()


if __name__ == "__main__":
    main()
