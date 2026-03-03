"""Golden-file example covering FlextContext public APIs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import BaseModel
from shared import Examples

from flext_core import FlextContext, FlextRuntime, FlextSettings, c, r, t

type _ContainerValue = (
    t.ScalarValue
    | BaseModel
    | Path
    | Sequence[_ContainerValue]
    | Mapping[str, _ContainerValue]
)


def _raise_boom(message: str) -> None:
    raise ValueError(message)


class _ContainerStub:
    def __init__(self) -> None:
        self._services: dict[str, t.RegisterableService] = {}
        self._config = FlextSettings()
        self._context = FlextContext.create()

    def _protocol_name(self) -> str:
        return "DI"

    def configure(self, config: Mapping[str, t.ScalarValue]) -> None:
        _ = config

    @property
    def config(self):
        return self._config

    @property
    def context(self):
        return self._context

    def scoped(self, **_kwargs: t.ContainerValue) -> _ContainerStub:
        return self

    def wire_modules(self, **_kwargs: t.ContainerValue) -> None:
        return None

    def get_config(self):
        return self._config.model_dump()

    def has_service(self, name: str) -> bool:
        return name in self._services

    def register(self, name: str, service: t.RegisterableService) -> r[bool]:
        self._services[name] = service
        return r[bool].ok(True)

    def register_factory(
        self,
        name: str,
        factory: t.FactoryCallable,
    ) -> r[bool]:
        self._services[name] = factory()
        return r[bool].ok(True)

    def with_service(self, name: str, service: t.RegisterableService) -> _ContainerStub:
        self._services[name] = service
        return self

    def with_factory(self, name: str, factory: t.FactoryCallable) -> _ContainerStub:
        self._services[name] = factory()
        return self

    def get(self, name: str):
        if name in self._services:
            return r[t.RegisterableService].ok(self._services[name])
        return r[t.RegisterableService].fail("missing")

    def get_typed(self, name: str, type_cls: type) -> r[object]:
        result = self.get(name)
        if result.is_failure:
            return r[type_cls].fail(result.error or "missing")
        value = result.value
        if isinstance(value, type_cls):
            return r[type_cls].ok(value)
        return r[type_cls].fail("wrong-type")

    def list_services(self):
        return list(self._services.keys())

    def clear_all(self) -> None:
        self._services.clear()


class Ex06FlextContext(Examples):
    """Exercise FlextContext public APIs."""

    def exercise(self) -> None:
        """Run all context demos and verify snapshot output."""
        FlextRuntime.configure_structlog(log_level=20)
        self._exercise_core_context_methods()
        self._exercise_container_and_service_methods()
        self._exercise_variables_and_domains()

    def _exercise_core_context_methods(self) -> None:
        """Exercise create/set/get/remove/clone/export and related APIs."""
        self.section("core_context_methods")

        operation_id = f"op-{self.rand_str(6)}"
        user_id = f"user-{self.rand_str(4)}"
        created_meta_key = "meta_from_create"
        created_meta_value = self.rand_str(5)
        ctx = FlextContext.create(operation_id=operation_id, user_id=user_id)
        ctx.set(created_meta_key, created_meta_value)
        self.check(
            "create.operation", ctx.get(c.Context.KEY_OPERATION_ID).unwrap_or("missing")
        )
        self.check("create.user", ctx.get(c.Context.KEY_USER_ID).unwrap_or("missing"))
        self.check("create.meta", ctx.get(created_meta_key).unwrap_or("missing"))
        self.check(
            "create.operation_matches",
            ctx.get(c.Context.KEY_OPERATION_ID).unwrap_or("missing") == operation_id,
        )
        self.check(
            "create.user_matches",
            ctx.get(c.Context.KEY_USER_ID).unwrap_or("missing") == user_id,
        )
        self.check(
            "create.meta_matches",
            ctx.get(created_meta_key).unwrap_or("missing") == created_meta_value,
        )

        key_one = "k1"
        value_one = self.rand_str(6)
        seed_key = "k2"
        seed_int = self.rand_int(1, 99)
        seed_bool_key = "k3"
        seed_bool = self.rand_bool()
        self.check("set.success", ctx.set(key_one, value_one).is_success)
        seed = FlextContext.create()
        seed.set(seed_key, seed_int)
        seed.set(seed_bool_key, seed_bool)
        scope_var = seed.iter_scope_vars()[c.Context.SCOPE_GLOBAL]
        payload = scope_var.get()
        set_all_ok = False
        if payload is not None:
            set_all_ok = ctx.set_all(payload).is_success
        self.check("set_all.success", set_all_ok)
        self.check("get.k1", ctx.get(key_one).unwrap_or("missing"))
        self.check("get.k1_matches", ctx.get(key_one).unwrap_or("missing") == value_one)
        self.check("has.k2", ctx.has(seed_key))
        ctx.remove(seed_key)
        self.check("remove.k2", ctx.has(seed_key))
        self.check("remove.k2_matches", ctx.has(seed_key) is False)

        self.check("keys.count", len(ctx.keys()))
        self.check("values.count", len(ctx.values()))
        self.check("items.count", len(ctx.items()))

        merged_key = "k4"
        merged_value = self.rand_str(6)
        merged = ctx.clone().merge({merged_key: merged_value})
        self.check("merge.get", merged.get(merged_key).unwrap_or("missing"))
        self.check(
            "merge.get_matches",
            merged.get(merged_key).unwrap_or("missing") == merged_value,
        )

        cloned = ctx.clone()
        self.check("clone.get", cloned.get(key_one).unwrap_or("missing"))
        self.check(
            "clone.get_matches",
            cloned.get(key_one).unwrap_or("missing") == value_one,
        )

        self.check("validate.success", ctx.validate().is_success)

        metadata_key = "meta_key"
        metadata_value = self.rand_str(7)
        ctx.set_metadata(metadata_key, metadata_value)
        self.check("get_metadata", ctx.get_metadata(metadata_key).unwrap_or("missing"))
        self.check(
            "get_metadata_matches",
            ctx.get_metadata(metadata_key).unwrap_or("missing") == metadata_value,
        )

        exported_min = ctx.export(as_dict=True)
        exported_full = ctx.export(
            include_statistics=True,
            include_metadata=True,
            as_dict=True,
        )
        exported_min_dict = (
            dict(exported_min) if isinstance(exported_min, Mapping) else {}
        )
        exported_full_dict = (
            dict(exported_full) if isinstance(exported_full, Mapping) else {}
        )
        self.check("export.min.has_global", "global" in exported_min_dict)
        self.check("export.full.has_statistics", "statistics" in exported_full_dict)
        self.check("export.full.has_metadata", "metadata" in exported_full_dict)

        scope_names = sorted(ctx.iter_scope_vars().keys())
        self.check("iter_scope_vars", scope_names)

        ctx.clear()
        self.check("clear.keys", len(ctx.keys()))

    def _exercise_container_and_service_methods(self) -> None:
        """Exercise container integration and service domain methods."""
        self.section("container_and_service")
        self.check("runtime.class", FlextRuntime.__name__)

        container = _ContainerStub()
        FlextContext.set_container(container)
        fetched_container = FlextContext.get_container()
        self.check("set_get_container.same", fetched_container is container)

        boom_message = self.rand_str(8)
        try:
            _raise_boom(boom_message)
        except ValueError as exc:
            self.check("raise.msg", str(exc))
            self.check("raise.msg_matches", str(exc) == boom_message)

        service_name = self.rand_str(8)
        service_value = self.rand_str(4)
        self.check(
            "service.register.ok",
            FlextContext.Service.register_service(
                service_name, service_value
            ).is_success,
        )
        self.check(
            "service.get.ok",
            FlextContext.Service.get_service(service_name).unwrap_or("missing"),
        )
        self.check(
            "service.get.ok_matches",
            FlextContext.Service.get_service(service_name).unwrap_or("missing")
            == service_value,
        )
        self.check(
            "service.get.missing",
            FlextContext.Service.get_service(self.rand_str(8)).is_failure,
        )

        before_service_name = FlextContext.Variables.ServiceName.get()
        scoped_service_name = self.rand_str(6)
        scoped_version = (
            f"{self.rand_int(1, 9)}.{self.rand_int(0, 9)}.{self.rand_int(0, 9)}"
        )
        with FlextContext.Service.service_context(
            scoped_service_name, version=scoped_version
        ):
            self.check(
                "service_context.name",
                FlextContext.Variables.ServiceName.get(),
            )
            self.check(
                "service_context.version",
                FlextContext.Variables.ServiceVersion.get(),
            )
            self.check(
                "service_context.name_matches",
                FlextContext.Variables.ServiceName.get() == scoped_service_name,
            )
            self.check(
                "service_context.version_matches",
                FlextContext.Variables.ServiceVersion.get() == scoped_version,
            )
        self.check(
            "service_context.restored",
            FlextContext.Variables.ServiceName.get() == before_service_name,
        )

    def _exercise_variables_and_domains(self) -> None:
        """Exercise variables constants, correlation, request, performance, and utilities."""
        self.section("variables_and_domains")
        _ = FlextContext.create()

        self.check(
            "var.correlation_id",
            FlextContext.Variables.Correlation.CORRELATION_ID is not None,
        )
        self.check(
            "var.parent_correlation_id",
            FlextContext.Variables.Correlation.PARENT_CORRELATION_ID is not None,
        )
        self.check(
            "var.service_name", FlextContext.Variables.Service.SERVICE_NAME is not None
        )
        self.check(
            "var.service_version",
            FlextContext.Variables.Service.SERVICE_VERSION is not None,
        )
        self.check("var.user_id", FlextContext.Variables.Request.USER_ID is not None)
        self.check(
            "var.request_id", FlextContext.Variables.Request.REQUEST_ID is not None
        )
        self.check(
            "var.request_timestamp",
            FlextContext.Variables.Request.REQUEST_TIMESTAMP is not None,
        )
        self.check(
            "var.operation_name",
            FlextContext.Variables.Performance.OPERATION_NAME is not None,
        )
        self.check(
            "var.operation_start",
            FlextContext.Variables.Performance.OPERATION_START_TIME is not None,
        )
        self.check(
            "var.operation_metadata",
            FlextContext.Variables.Performance.OPERATION_METADATA is not None,
        )

        self.check(
            "alias.correlation", FlextContext.Variables.CorrelationId is not None
        )
        self.check(
            "alias.parent_correlation",
            FlextContext.Variables.ParentCorrelationId is not None,
        )
        self.check("alias.service_name", FlextContext.Variables.ServiceName is not None)
        self.check(
            "alias.service_version", FlextContext.Variables.ServiceVersion is not None
        )
        self.check("alias.user_id", FlextContext.Variables.UserId is not None)
        self.check("alias.request_id", FlextContext.Variables.RequestId is not None)
        self.check(
            "alias.request_timestamp",
            FlextContext.Variables.RequestTimestamp is not None,
        )
        self.check(
            "alias.operation_name", FlextContext.Variables.OperationName is not None
        )
        self.check(
            "alias.operation_start",
            FlextContext.Variables.OperationStartTime is not None,
        )
        self.check(
            "alias.operation_metadata",
            FlextContext.Variables.OperationMetadata is not None,
        )

        correlation_id = self.rand_str(8)
        FlextContext.Correlation.set_correlation_id(correlation_id)
        self.check("correlation.get_set", FlextContext.Correlation.get_correlation_id())
        self.check(
            "correlation.get_set_matches",
            FlextContext.Correlation.get_correlation_id() == correlation_id,
        )
        new_correlation_id = self.rand_str(8)
        parent_correlation_id = self.rand_str(8)
        with FlextContext.Correlation.new_correlation(
            new_correlation_id,
            parent_id=parent_correlation_id,
        ) as corr_id:
            self.check("correlation.new.value", corr_id)
            self.check(
                "correlation.new.current",
                FlextContext.Correlation.get_correlation_id(),
            )
            self.check("correlation.new.value_matches", corr_id == new_correlation_id)
            self.check(
                "correlation.new.current_matches",
                FlextContext.Correlation.get_correlation_id() == new_correlation_id,
            )

        operation_name = self.rand_str(9)
        FlextContext.Request.set_operation_name(operation_name)
        self.check("request.get_set", FlextContext.Request.get_operation_name())
        self.check(
            "request.get_set_matches",
            FlextContext.Request.get_operation_name() == operation_name,
        )

        timed_name = self.rand_str(8)
        with FlextContext.Performance.timed_operation(timed_name) as op_meta:
            self.check(
                "timed_operation.has_start",
                c.Context.METADATA_KEY_START_TIME in op_meta,
            )
            self.check(
                "timed_operation.has_name_matches",
                op_meta.get(c.Context.KEY_OPERATION_NAME) == timed_name,
            )
            full_context = FlextContext.Serialization.get_full_context()
            self.check(
                "serialization.has_correlation_key",
                c.Context.KEY_CORRELATION_ID in full_context,
            )
            self.check(
                "serialization.has_operation_name",
                full_context.get(c.Context.KEY_OPERATION_NAME) == timed_name,
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
            cleared_context.get(c.Context.KEY_CORRELATION_ID),
        )
        ensured = FlextContext.Utilities.ensure_correlation_id()
        self.check("utilities.ensure_correlation_id.non_empty", len(ensured) > 0)


if __name__ == "__main__":
    Ex06FlextContext(__file__).run()
