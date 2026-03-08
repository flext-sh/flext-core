"""Golden-file example for FlextContext public API."""

from __future__ import annotations

from typing import override

from flext_core import FlextContainer, FlextContext, FlextRuntime, c, m

from .shared import Examples


class Ex06FlextContext(Examples):
    """Exercise FlextContext API with deterministic output checks."""

    @override
    def exercise(self) -> None:
        self._exercise_core_context_methods()
        self._exercise_container_and_service_methods()
        self._exercise_variables_and_domains()

    def _exercise_core_context_methods(self) -> None:
        self.section("core_context_methods")
        ctx = FlextContext.create(operation_id="op-demo", user_id="user-1")
        ctx.set("meta_from_create", "v")
        self.check("create.instance", type(ctx).__name__)
        self.check("create.has.operation", ctx.has(c.Context.KEY_OPERATION_ID))
        self.check("create.has.user", ctx.has(c.Context.KEY_USER_ID))
        self.check("set.success", ctx.set("k1", "v1").is_success)
        seed = FlextContext.create()
        seed.set("k2", 2)
        seed.set("k3", True)
        payload = seed.iter_scope_vars()[c.Context.SCOPE_GLOBAL].get()
        self.check(
            "set_all.success", ctx.set(payload or m.ConfigMap(root={})).is_success
        )
        self.check("get.k1", ctx.get("k1").unwrap_or("missing"))
        self.check("has.k2", ctx.has("k2"))
        ctx.remove("k2")
        self.check("remove.k2", ctx.has("k2"))
        self.check("keys.count", len(ctx.keys()))
        self.check("values.count", len(ctx.values()))
        self.check("items.count", len(ctx.items()))
        merged = ctx.clone().merge(m.ConfigMap(root={"k4": "merged"}).root)
        self.check("merge.get", merged.get("k4").unwrap_or("missing"))
        self.check("clone.get", ctx.clone().get("k1").unwrap_or("missing"))
        self.check("validate.success", ctx.validate().is_success)
        ctx.set_metadata("meta_key", "meta_value")
        self.check("get_metadata", ctx.get_metadata("meta_key").unwrap_or("missing"))
        exported_min = ctx.export(as_dict=False)
        exported_full = ctx.export(
            include_statistics=True, include_metadata=True, as_dict=False
        )
        self.check("export.min.type", type(exported_min).__name__)
        self.check("export.full.type", type(exported_full).__name__)
        self.check("iter_scope_vars", ",".join(sorted(ctx.iter_scope_vars().keys())))
        ctx.clear()
        self.check("clear.keys", len(ctx.keys()))

    def _exercise_container_and_service_methods(self) -> None:
        self.section("container_and_service")
        self.check("runtime.class", FlextRuntime.__name__)
        container = FlextContainer()
        FlextContext.set_container(container)
        self.check("set_get_container.same", FlextContext.get_container() is container)
        try:
            msg = "boom"
            raise ValueError(msg)
        except ValueError as exc:
            self.check("raise.msg", str(exc))
        self.check(
            "service.register.ok",
            FlextContext.Service.register_service("demo-service", "svc").is_success,
        )
        self.check(
            "service.get.ok",
            FlextContext.Service.get_service("demo-service").unwrap_or("missing"),
        )
        self.check(
            "service.get.missing",
            FlextContext.Service.get_service("missing").is_failure,
        )
        before_service_name = FlextContext.Variables.ServiceName.get()
        with FlextContext.Service.service_context("orders", version="1.2.3"):
            self.check(
                "service_context.name", FlextContext.Variables.ServiceName.get() or ""
            )
            self.check(
                "service_context.version",
                FlextContext.Variables.ServiceVersion.get() or "",
            )
        self.check(
            "service_context.restored",
            FlextContext.Variables.ServiceName.get() == before_service_name,
        )

    def _exercise_variables_and_domains(self) -> None:
        self.section("variables_and_domains")
        FlextContext.create()
        self.check(
            "var.correlation_id",
            type(FlextContext.Variables.Correlation.CORRELATION_ID).__name__,
        )
        self.check(
            "var.parent_correlation_id",
            type(FlextContext.Variables.Correlation.PARENT_CORRELATION_ID).__name__,
        )
        self.check(
            "var.service_name",
            type(FlextContext.Variables.Service.SERVICE_NAME).__name__,
        )
        self.check(
            "var.service_version",
            type(FlextContext.Variables.Service.SERVICE_VERSION).__name__,
        )
        self.check("var.user_id", type(FlextContext.Variables.Request.USER_ID).__name__)
        self.check(
            "var.request_id", type(FlextContext.Variables.Request.REQUEST_ID).__name__
        )
        self.check(
            "var.request_timestamp",
            type(FlextContext.Variables.Request.REQUEST_TIMESTAMP).__name__,
        )
        self.check(
            "var.operation_name",
            type(FlextContext.Variables.Performance.OPERATION_NAME).__name__,
        )
        self.check(
            "var.operation_start",
            type(FlextContext.Variables.Performance.OPERATION_START_TIME).__name__,
        )
        self.check(
            "var.operation_metadata",
            type(FlextContext.Variables.Performance.OPERATION_METADATA).__name__,
        )
        self.check(
            "alias.correlation", type(FlextContext.Variables.CorrelationId).__name__
        )
        self.check(
            "alias.parent_correlation",
            type(FlextContext.Variables.ParentCorrelationId).__name__,
        )
        self.check(
            "alias.service_name", type(FlextContext.Variables.ServiceName).__name__
        )
        self.check(
            "alias.service_version",
            type(FlextContext.Variables.ServiceVersion).__name__,
        )
        self.check("alias.user_id", type(FlextContext.Variables.UserId).__name__)
        self.check("alias.request_id", type(FlextContext.Variables.RequestId).__name__)
        self.check(
            "alias.request_timestamp",
            type(FlextContext.Variables.RequestTimestamp).__name__,
        )
        self.check(
            "alias.operation_name", type(FlextContext.Variables.OperationName).__name__
        )
        self.check(
            "alias.operation_start",
            type(FlextContext.Variables.OperationStartTime).__name__,
        )
        self.check(
            "alias.operation_metadata",
            type(FlextContext.Variables.OperationMetadata).__name__,
        )
        FlextContext.Correlation.set_correlation_id("cid-1")
        self.check(
            "correlation.get_set", FlextContext.Correlation.get_correlation_id() or ""
        )
        with FlextContext.Correlation.new_correlation(
            "cid-2", parent_id="cid-parent"
        ) as corr_id:
            self.check("correlation.new.value", corr_id)
            self.check(
                "correlation.new.current",
                FlextContext.Correlation.get_correlation_id() or "",
            )
        FlextContext.Request.set_operation_name("sync-users")
        self.check("request.get_set", FlextContext.Request.get_operation_name() or "")
        with FlextContext.Performance.timed_operation("bulk-sync") as op_meta:
            self.check(
                "timed_operation.has_start",
                c.Context.METADATA_KEY_START_TIME in op_meta,
            )
            self.check(
                "timed_operation.has_name",
                op_meta.get(c.Context.KEY_OPERATION_NAME) or "",
            )
            full_context = FlextContext.Serialization.get_full_context()
            self.check(
                "serialization.has_correlation_key",
                c.Context.KEY_CORRELATION_ID in full_context,
            )
            self.check(
                "serialization.has_operation_name",
                full_context.get(c.Context.KEY_OPERATION_NAME) or "",
            )
        self.check(
            "timed_operation.has_end", c.Context.METADATA_KEY_END_TIME in op_meta
        )
        self.check(
            "timed_operation.has_duration",
            c.Context.METADATA_KEY_DURATION_SECONDS in op_meta,
        )
        FlextContext.Utilities.clear_context()
        cleared_context = FlextContext.Serialization.get_full_context()
        self.check(
            "utilities.clear_context.correlation",
            cleared_context.get(c.Context.KEY_CORRELATION_ID) or "",
        )
        ensured = FlextContext.Utilities.ensure_correlation_id()
        self.check("utilities.ensure_correlation_id.non_empty", len(ensured) > 0)


if __name__ == "__main__":
    Ex06FlextContext(__file__).run()
