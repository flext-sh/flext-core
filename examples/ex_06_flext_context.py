"""Golden-file example covering FlextContext public APIs."""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import cast

from flext_core import FlextContext, FlextRuntime, c, p, r, t, u

_RESULTS: list[str] = []


def _check(label: str, value: t.Container) -> None:
    _RESULTS.append(f"{label}: {_ser(value)}")


def _section(name: str) -> None:
    if _RESULTS:
        _RESULTS.append("")
    _RESULTS.append(f"[{name}]")


def _ser(v: t.Container) -> str:
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
            f"{_ser(k)}: {_ser(val)}"
            for k, val in sorted(v.items(), key=lambda kv: str(kv[0]))
        )
        return "{" + pairs + "}"
    if isinstance(v, datetime):
        return v.isoformat()
    if isinstance(v, Path):
        return str(v)
    return type(v).__name__


def _verify() -> None:
    actual = "\n".join(_RESULTS).strip() + "\n"
    me = Path(__file__)
    expected_path = me.with_suffix(".expected")
    checks = sum(1 for line in _RESULTS if ": " in line and not line.startswith("["))

    if expected_path.exists():
        expected = expected_path.read_text(encoding="utf-8")
        if actual == expected:
            sys.stdout.write(f"PASS: {me.stem} ({checks} checks)\n")
            return

        actual_path = me.with_suffix(".actual")
        actual_path.write_text(actual, encoding="utf-8")
        sys.stdout.write(
            f"FAIL: {me.stem} — diff {expected_path.name} {actual_path.name}\n",
        )
        sys.exit(1)

    expected_path.write_text(actual, encoding="utf-8")
    sys.stdout.write(f"GENERATED: {expected_path.name} ({checks} checks)\n")


def _raise_boom() -> None:
    msg = "boom"
    raise ValueError(msg)


class _ContainerStub:
    def __init__(self) -> None:
        self._services: dict[str, t.Container] = {}
        self._config: dict[str, t.Container] = {}
        self._context = FlextContext.create()

    @property
    def config(self) -> Mapping[str, t.Container]:
        return self._config

    @property
    def context(self) -> FlextContext:
        return self._context

    def scoped(self, **_kwargs: object) -> _ContainerStub:
        return self

    def wire_modules(self, **_kwargs: object) -> None:
        return None

    def get_config(self) -> Mapping[str, t.Container]:
        return self._config

    def has_service(self, name: str) -> bool:
        return name in self._services

    def register(self, name: str, service: t.Container) -> r[bool]:
        self._services[name] = service
        return r[bool].ok(True)

    def register_factory(
        self,
        name: str,
        factory: Callable[[], t.Container],
    ) -> r[bool]:
        self._services[name] = factory()
        return r[bool].ok(True)

    def with_service(self, name: str, service: t.Container) -> _ContainerStub:
        self._services[name] = service
        return self

    def with_factory(
        self,
        name: str,
        factory: Callable[[], t.Container],
    ) -> _ContainerStub:
        self._services[name] = factory()
        return self

    def get(self, name: str) -> r[t.Container]:
        if name in self._services:
            return r[t.Container].ok(self._services[name])
        return r[t.Container].fail("missing")

    def get_typed(self, name: str, type_cls: type[object]) -> r[t.Container]:
        result = self.get(name)
        if result.is_failure:
            return r[t.Container].fail(result.error or "missing")
        value = result.value
        if isinstance(value, type_cls):
            return r[t.Container].ok(value)
        return r[t.Container].fail("wrong-type")

    def list_services(self) -> list[str]:
        return list(self._services.keys())

    def clear_all(self) -> None:
        self._services.clear()


def demo_core_context_methods() -> None:
    """Exercise create/set/get/remove/clone/export and related APIs."""
    _section("core_context_methods")

    ctx = FlextContext.create(
        operation_id="op-demo",
        user_id="user-1",
    )
    ctx.set("meta_from_create", "v")
    _check("create.instance", isinstance(ctx, FlextContext))
    _check("create.has.operation", ctx.has(c.Context.KEY_OPERATION_ID))
    _check("create.has.user", ctx.has(c.Context.KEY_USER_ID))

    _check("set.success", ctx.set("k1", "v1").is_success)
    seed = FlextContext.create()
    seed.set("k2", 2)
    seed.set("k3", True)
    scope_var = seed.iter_scope_vars()[c.Context.SCOPE_GLOBAL]
    payload = scope_var.get()
    set_all_ok = False
    if payload is not None:
        set_all_ok = ctx.set(payload).is_success
    _check("set_all.success", set_all_ok)
    _check("get.k1", ctx.get("k1").unwrap_or("missing"))
    _check("has.k2", ctx.has("k2"))
    ctx.remove("k2")
    _check("remove.k2", ctx.has("k2"))

    _check("keys.count", len(ctx.keys()))
    _check("values.count", len(ctx.values()))
    _check("items.count", len(ctx.items()))

    merged = ctx.clone().merge({"k4": "merged"})
    _check("merge.get", merged.get("k4").unwrap_or("missing"))

    cloned = ctx.clone()
    _check("clone.get", cloned.get("k1").unwrap_or("missing"))

    _check("validate.success", ctx.validate().is_success)

    ctx.set_metadata("meta_key", "meta_value")
    _check(
        "get_metadata",
        ctx.get_metadata("meta_key").unwrap_or("missing"),
    )

    exported_min = ctx.export(as_dict=True)
    exported_full = ctx.export(
        include_statistics=True,
        include_metadata=True,
        as_dict=True,
    )
    exported_min_dict = (
        dict(exported_min) if not isinstance(exported_min, dict) else exported_min
    )
    exported_full_dict = (
        dict(exported_full) if not isinstance(exported_full, dict) else exported_full
    )
    _check("export.min.has_global", "global" in exported_min_dict)
    _check("export.full.has_statistics", "statistics" in exported_full_dict)
    _check("export.full.has_metadata", "metadata" in exported_full_dict)

    scope_names = sorted(ctx.iter_scope_vars().keys())
    _check("iter_scope_vars", str(scope_names))

    ctx.clear()
    _check("clear.keys", len(ctx.keys()))


def demo_container_and_service_methods() -> None:
    """Exercise container integration and service domain methods."""
    _section("container_and_service")
    _check("runtime.class", FlextRuntime.__name__)

    container = _ContainerStub()
    FlextContext.set_container(cast("p.DI", container))
    fetched_container = FlextContext.get_container()
    _check("set_get_container.same", fetched_container is container)

    try:
        _raise_boom()
    except ValueError as exc:
        _check("raise.msg", str(exc))

    _check(
        "service.register.ok",
        FlextContext.Service.register_service("demo-service", "svc").is_success,
    )
    _check(
        "service.get.ok",
        FlextContext.Service.get_service("demo-service").unwrap_or("missing"),
    )
    _check(
        "service.get.missing",
        FlextContext.Service.get_service("missing").is_failure,
    )

    before_service_name = FlextContext.Variables.ServiceName.get()
    with FlextContext.Service.service_context("orders", version="1.2.3"):
        _check(
            "service_context.name",
            FlextContext.Variables.ServiceName.get() or "",
        )
        _check(
            "service_context.version",
            FlextContext.Variables.ServiceVersion.get() or "",
        )
    _check(
        "service_context.restored",
        FlextContext.Variables.ServiceName.get() == before_service_name,
    )


def demo_variables_and_domains() -> None:
    """Exercise variables constants, correlation, request, performance, and utilities."""
    _section("variables_and_domains")
    _ = FlextContext.create()

    _check(
        "var.correlation_id",
        FlextContext.Variables.Correlation.CORRELATION_ID is not None,
    )
    _check(
        "var.parent_correlation_id",
        FlextContext.Variables.Correlation.PARENT_CORRELATION_ID is not None,
    )
    _check("var.service_name", FlextContext.Variables.Service.SERVICE_NAME is not None)
    _check(
        "var.service_version",
        FlextContext.Variables.Service.SERVICE_VERSION is not None,
    )
    _check("var.user_id", FlextContext.Variables.Request.USER_ID is not None)
    _check("var.request_id", FlextContext.Variables.Request.REQUEST_ID is not None)
    _check(
        "var.request_timestamp",
        FlextContext.Variables.Request.REQUEST_TIMESTAMP is not None,
    )
    _check(
        "var.operation_name",
        FlextContext.Variables.Performance.OPERATION_NAME is not None,
    )
    _check(
        "var.operation_start",
        FlextContext.Variables.Performance.OPERATION_START_TIME is not None,
    )
    _check(
        "var.operation_metadata",
        FlextContext.Variables.Performance.OPERATION_METADATA is not None,
    )

    _check("alias.correlation", FlextContext.Variables.CorrelationId is not None)
    _check(
        "alias.parent_correlation",
        FlextContext.Variables.ParentCorrelationId is not None,
    )
    _check("alias.service_name", FlextContext.Variables.ServiceName is not None)
    _check("alias.service_version", FlextContext.Variables.ServiceVersion is not None)
    _check("alias.user_id", FlextContext.Variables.UserId is not None)
    _check("alias.request_id", FlextContext.Variables.RequestId is not None)
    _check(
        "alias.request_timestamp",
        FlextContext.Variables.RequestTimestamp is not None,
    )
    _check("alias.operation_name", FlextContext.Variables.OperationName is not None)
    _check(
        "alias.operation_start",
        FlextContext.Variables.OperationStartTime is not None,
    )
    _check(
        "alias.operation_metadata",
        FlextContext.Variables.OperationMetadata is not None,
    )

    FlextContext.Correlation.set_correlation_id("cid-1")
    _check("correlation.get_set", FlextContext.Correlation.get_correlation_id() or "")
    with FlextContext.Correlation.new_correlation(
        "cid-2", parent_id="cid-parent"
    ) as corr_id:
        _check("correlation.new.value", corr_id)
        _check(
            "correlation.new.current",
            FlextContext.Correlation.get_correlation_id() or "",
        )

    FlextContext.Request.set_operation_name("sync-users")
    _check("request.get_set", FlextContext.Request.get_operation_name() or "")

    with FlextContext.Performance.timed_operation("bulk-sync") as op_meta:
        _check(
            "timed_operation.has_start", c.Context.METADATA_KEY_START_TIME in op_meta
        )
        _check(
            "timed_operation.has_name", op_meta.get(c.Context.KEY_OPERATION_NAME) or ""
        )
        full_context = FlextContext.Serialization.get_full_context()
        _check(
            "serialization.has_correlation_key",
            c.Context.KEY_CORRELATION_ID in full_context,
        )
        _check(
            "serialization.has_operation_name",
            full_context.get(c.Context.KEY_OPERATION_NAME) or "",
        )
    _check("timed_operation.has_end", c.Context.METADATA_KEY_END_TIME in op_meta)
    _check(
        "timed_operation.has_duration",
        c.Context.METADATA_KEY_DURATION_SECONDS in op_meta,
    )

    FlextContext.Utilities.clear_context()
    cleared_context = FlextContext.Serialization.get_full_context()
    _check(
        "utilities.clear_context.correlation",
        cleared_context.get(c.Context.KEY_CORRELATION_ID) or "",
    )
    ensured = FlextContext.Utilities.ensure_correlation_id()
    _check("utilities.ensure_correlation_id.non_empty", len(ensured) > 0)


def main() -> None:
    """Run all context demos and verify snapshot output."""
    demo_core_context_methods()
    demo_container_and_service_methods()
    demo_variables_and_domains()
    _verify()


if __name__ == "__main__":
    main()
