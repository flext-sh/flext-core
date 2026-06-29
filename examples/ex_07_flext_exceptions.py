"""Golden-file example for e public API."""

from __future__ import annotations

from pathlib import Path
from typing import override

from flext_core import c, e, r

from .ex_07_flext_exceptions_helpers import Ex07FlextExceptionSubclasses
from .models import m


class Ex07FlextExceptions(Ex07FlextExceptionSubclasses):
    """Exercise e API with deterministic output checks."""

    @override
    def exercise(self) -> None:
        self.section("imports")
        self.audit_check("import.e_is_FlextExceptions", e.__name__ == "e")
        self.audit_check("import.r_ok", r[str].ok("ok").success)
        self.audit_check("import.constant", c.ErrorCode.UNKNOWN_ERROR)
        try:
            raise ValueError(m.Examples.ErrorMessages.BOOM)
        except ValueError as exc:
            self.audit_check("style.raise_msg", str(exc))
        self._exercise_base_error()
        self._exercise_specific_exceptions()
        self._exercise_factories_and_helpers()
        self._exercise_metrics()

    def _exercise_base_error(self) -> None:
        self.section("base_error")
        base = e.BaseError(
            "base boom",
            error_code="E_BASE",
            context=m.ConfigMap(root={"scope": "demo"}),
            metadata=m.Metadata(attributes={"channel": "example"}),
            correlation_id="corr-base-1",
            auto_correlation=False,
            auto_log=False,
            operation="create",
        )
        self.audit_check("base.class", type(base).__name__)
        self.audit_check("base.message", base.message)
        self.audit_check("base.error_code", base.error_code)
        self.audit_check("base.correlation_id", base.correlation_id or "")
        self.audit_check("base.auto_log", base.auto_log)
        self.audit_check(
            "base.meta.scope", str(base.metadata.attributes.get("scope") or "")
        )
        self.audit_check(
            "base.meta.channel",
            str(base.metadata.attributes.get("channel") or ""),
        )
        self.audit_check(
            "base.meta.operation",
            str(base.metadata.attributes.get("operation") or ""),
        )
        self.audit_check("base.error_message", base.error_message)
        self.audit_check("base.has_timestamp", base.timestamp > 0)
        auto_corr = e.BaseError(
            "auto corr",
            error_code="E_AUTO",
            auto_correlation=True,
            auto_log=False,
        )
        self.audit_check(
            "base.auto_corr.prefix",
            (auto_corr.correlation_id or "").startswith("exc_"),
        )

    def _exercise_factories_and_helpers(self) -> None:
        self.section("factories_helpers")
        self.audit_check(
            "create_error.ValidationError",
            type(e.ValidationError("factory validation", field="test")).__name__,
        )
        self.audit_check(
            "create_error.AttributeError",
            type(
                e.AttributeAccessError("factory attribute", attribute_name="test")
            ).__name__,
        )
        created_dynamic = e.ValidationError(
            "dynamic",
            error_code="E_DYNAMIC",
            field="username",
            value="",
            correlation_id="corr-dyn-1",
        )
        self.audit_check("create.type", type(created_dynamic).__name__)
        self.audit_check("create.error_code", created_dynamic.error_code)
        self.audit_check(
            "create.metadata.caller",
            str(created_dynamic.metadata.attributes.get("caller") or ""),
        )
        self.audit_check("create.correlation_id", created_dynamic.correlation_id or "")
        instance_created = e.ValidationError(
            "from __call__",
            error_code="E_CALL",
            field="title",
            value="",
        )
        self.audit_check("__call__.type", type(instance_created).__name__)
        self.audit_check("__call__.error_code", instance_created.error_code)

    def _exercise_metrics(self) -> None:
        self.section("metrics")
        e.clear_metrics()
        e.record_exception(e.ValidationError)
        e.record_exception(e.ValidationError)
        e.record_exception(e.ConfigurationError)
        metrics = e.resolve_metrics_snapshot()
        self.audit_check("metrics.total_exceptions", metrics.total_exceptions)
        self.audit_check("metrics.has_exception_counts", bool(metrics.exception_counts))
        self.audit_check(
            "metrics.summary_nonempty",
            bool(metrics.exception_counts_summary),
        )
        self.audit_check("metrics.unique_types", metrics.unique_exception_types)
        e.clear_metrics()
        self.audit_check(
            "metrics.cleared_total",
            e.resolve_metrics_snapshot().total_exceptions,
        )


if __name__ == "__main__":
    Ex07FlextExceptions(caller_file=Path(__file__)).run()
