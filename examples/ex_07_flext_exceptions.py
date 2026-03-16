"""Golden-file example for FlextExceptions public API."""

from __future__ import annotations

from typing import override

from flext_core import FlextExceptions, c, e, m, r

from .shared import Examples


class Ex07FlextExceptions(Examples):
    """Exercise FlextExceptions API with deterministic output checks."""

    @override
    def exercise(self) -> None:
        self.section("imports")
        self.check("import.e_is_FlextExceptions", e is FlextExceptions)
        self.check("import.r_ok", r[str].ok("ok").is_success)
        self.check("import.constant", c.Errors.UNKNOWN_ERROR)
        try:
            msg = "boom"
            raise ValueError(msg)
        except ValueError as exc:
            self.check("style.raise_msg", str(exc))
        self._exercise_base_error()
        self._exercise_specific_exceptions()
        self._exercise_factories_and_helpers()
        self._exercise_metrics()

    def _exercise_base_error(self) -> None:
        self.section("base_error")
        base = e.BaseError(
            "base boom",
            error_code="E_BASE",
            context=t.ConfigMap(root={"scope": "demo"}),
            metadata=m.Metadata(attributes={"channel": "example"}),
            correlation_id="corr-base-1",
            auto_correlation=False,
            auto_log=False,
            operation="create",
        )
        self.check("base.class", type(base).__name__)
        self.check("base.message", base.message)
        self.check("base.error_code", base.error_code)
        self.check("base.correlation_id", base.correlation_id or "")
        self.check("base.auto_log", base.auto_log)
        self.check("base.meta.scope", str(base.metadata.attributes.get("scope") or ""))
        self.check(
            "base.meta.channel", str(base.metadata.attributes.get("channel") or "")
        )
        self.check(
            "base.meta.operation", str(base.metadata.attributes.get("operation") or "")
        )
        payload = base.to_dict()
        self.check("base.to_dict.type", str(payload.get("error_type") or ""))
        self.check("base.to_dict.message", str(payload.get("message") or ""))
        self.check("base.to_dict.error_code", str(payload.get("error_code") or ""))
        self.check("base.to_dict.has_timestamp", "timestamp" in payload)
        auto_corr = e.BaseError(
            "auto corr", error_code="E_AUTO", auto_correlation=True, auto_log=False
        )
        self.check(
            "base.auto_corr.prefix", (auto_corr.correlation_id or "").startswith("exc_")
        )

    def _exercise_specific_exceptions(self) -> None:
        self.section("subclasses")
        try:
            msg = "invalid"
            raise e.ValidationError(msg, field="email", value="bad")
        except e.ValidationError as exc:
            self.check("ValidationError.field", exc.field or "")
            self.check("ValidationError.value", str(exc.value or ""))
        try:
            msg = "bad cfg"
            raise e.ConfigurationError(msg, config_key="db.host", config_source="env")
        except e.ConfigurationError as exc:
            self.check("ConfigurationError.config_key", exc.config_key or "")
            self.check("ConfigurationError.config_source", exc.config_source or "")
        try:
            msg = "down"
            raise e.ConnectionError(msg, host="127.0.0.1", port=5432, timeout=3.5)
        except e.ConnectionError as exc:
            self.check("ConnectionError.host", exc.host or "")
            self.check("ConnectionError.port", exc.port or 0)
            self.check("ConnectionError.timeout", exc.timeout or 0.0)
        try:
            msg = "late"
            raise e.TimeoutError(msg, timeout_seconds=2.0, operation="sync")
        except e.TimeoutError as exc:
            self.check("TimeoutError.timeout_seconds", exc.timeout_seconds or 0.0)
            self.check("TimeoutError.operation", exc.operation or "")
        try:
            msg = "auth fail"
            raise e.AuthenticationError(msg, auth_method="token", user_id="u-1")
        except e.AuthenticationError as exc:
            self.check("AuthenticationError.auth_method", exc.auth_method or "")
            self.check("AuthenticationError.user_id", exc.user_id or "")
        try:
            msg = "nope"
            raise e.AuthorizationError(
                msg, user_id="u-2", resource="invoice:7", permission="read"
            )
        except e.AuthorizationError as exc:
            self.check("AuthorizationError.user_id", exc.user_id or "")
            self.check("AuthorizationError.resource", exc.resource or "")
            self.check("AuthorizationError.permission", exc.permission or "")
        try:
            msg = "missing"
            raise e.NotFoundError(msg, resource_type="User", resource_id="404")
        except e.NotFoundError as exc:
            self.check("NotFoundError.resource_type", exc.resource_type or "")
            self.check("NotFoundError.resource_id", exc.resource_id or "")
        try:
            msg = "conflict"
            raise e.ConflictError(
                msg, resource_type="User", resource_id="13", conflict_reason="duplicate"
            )
        except e.ConflictError as exc:
            self.check("ConflictError.resource_type", exc.resource_type or "")
            self.check("ConflictError.resource_id", exc.resource_id or "")
            self.check("ConflictError.conflict_reason", exc.conflict_reason or "")
        try:
            msg = "slow down"
            raise e.RateLimitError(msg, limit=100, window_seconds=60, retry_after=1.5)
        except e.RateLimitError as exc:
            self.check("RateLimitError.limit", exc.limit or 0)
            self.check("RateLimitError.window_seconds", exc.window_seconds or 0)
            self.check("RateLimitError.retry_after", exc.retry_after or 0.0)
        try:
            msg = "open"
            raise e.CircuitBreakerError(
                msg, service_name="billing", failure_count=5, reset_timeout=30.0
            )
        except e.CircuitBreakerError as exc:
            self.check("CircuitBreakerError.service_name", exc.service_name or "")
            self.check("CircuitBreakerError.failure_count", exc.failure_count or 0)
            self.check("CircuitBreakerError.reset_timeout", exc.reset_timeout or 0.0)
        try:
            msg = "wrong type"
            raise e.TypeError(msg, expected_type=str, actual_type=int)
        except e.TypeError as exc:
            self.check(
                "TypeError.expected_type",
                exc.expected_type.__name__ if exc.expected_type else "",
            )
            self.check(
                "TypeError.actual_type",
                exc.actual_type.__name__ if exc.actual_type else "",
            )
        try:
            msg = "failed op"
            raise e.OperationError(msg, operation="publish", reason="quota")
        except e.OperationError as exc:
            self.check("OperationError.operation", exc.operation or "")
            self.check("OperationError.reason", exc.reason or "")
        try:
            msg = "bad attr"
            raise e.AttributeAccessError(
                msg, attribute_name="secret", attribute_context="UserModel"
            )
        except e.AttributeAccessError as exc:
            self.check("AttributeAccessError.attribute_name", exc.attribute_name or "")
            self.check(
                "AttributeAccessError.attribute_context",
                str(exc.attribute_context or ""),
            )

    def _exercise_factories_and_helpers(self) -> None:
        self.section("factories_helpers")
        self.check(
            "create_error.ValidationError",
            type(e.create("ValidationError", "factory validation")).__name__,
        )
        self.check(
            "create_error.AttributeError",
            type(e.create("AttributeError", "factory attribute")).__name__,
        )
        created_dynamic = e.create(
            "dynamic",
            error_code="E_DYNAMIC",
            field="username",
            value="",
            correlation_id="corr-dyn-1",
        )
        self.check("create.type", type(created_dynamic).__name__)
        self.check("create.error_code", created_dynamic.error_code)
        self.check(
            "create.metadata.caller",
            str(created_dynamic.metadata.attributes.get("caller") or ""),
        )
        self.check("create.correlation_id", created_dynamic.correlation_id or "")
        instance_created = FlextExceptions()(
            "from __call__", error_code="E_CALL", field="title", value=""
        )
        self.check("__call__.type", type(instance_created).__name__)
        self.check("__call__.error_code", instance_created.error_code)

    def _exercise_metrics(self) -> None:
        self.section("metrics")
        e.clear_metrics()
        e.record_exception(e.ValidationError)
        e.record_exception(e.ValidationError)
        e.record_exception(e.ConfigurationError)
        metric_map = e.get_metrics().root
        self.check("metrics.total_exceptions", metric_map.get("total_exceptions", 0))
        self.check("metrics.has_exception_counts", "exception_counts" in metric_map)
        self.check(
            "metrics.summary_nonempty", bool(metric_map.get("exception_counts_summary"))
        )
        self.check("metrics.unique_types", metric_map.get("unique_exception_types", 0))
        e.clear_metrics()
        self.check(
            "metrics.cleared_total", e.get_metrics().root.get("total_exceptions", 0)
        )


if __name__ == "__main__":
    Ex07FlextExceptions(__file__).run()
