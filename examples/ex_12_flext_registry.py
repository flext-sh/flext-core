"""Golden-file example for FlextRegistry public APIs."""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from flext_core import FlextDispatcher, FlextHandlers, FlextRegistry, c, m, r, t, u

_RESULTS: list[str] = []


def _check(label: str, value: object) -> None:
    normalized = _normalize_for_ser(value)
    _RESULTS.append(f"{label}: {_ser(normalized)}")


def _section(name: str) -> None:
    if _RESULTS:
        _RESULTS.append("")
    _RESULTS.append(f"[{name}]")


def _normalize_for_ser(value: object) -> t.ContainerValue | None:
    if value is None:
        return None
    if isinstance(value, bool | int | float | str | datetime | Path):
        return value
    if isinstance(value, BaseModel):
        return str(value.model_dump())
    return str(value)


def _ser(v: t.ContainerValue | None) -> str:
    if v is None:
        return "None"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        return repr(v)
    if (
        u.is_list(v)
        and isinstance(v, Sequence)
        and not isinstance(v, str | bytes | bytearray)
    ):
        return "[" + ", ".join(_ser(item) for item in v) + "]"
    if u.is_dict_like(v) and isinstance(v, Mapping):
        pairs = ", ".join(
            f"{_ser(key)}: {_ser(val)}"
            for key, val in sorted(v.items(), key=lambda kv: str(kv[0]))
        )
        return "{" + pairs + "}"
    return type(v).__name__


def _verify() -> None:
    actual = "\n".join(_RESULTS).strip() + "\n"
    me = Path(__file__)
    expected_path = me.with_suffix(".expected")
    n = sum(1 for line in _RESULTS if ": " in line and not line.startswith("["))
    if expected_path.exists():
        expected = expected_path.read_text(encoding="utf-8")
        if actual == expected:
            sys.stdout.write(f"PASS: {me.stem} ({n} checks)\n")
        else:
            actual_path = me.with_suffix(".actual")
            actual_path.write_text(actual, encoding="utf-8")
            sys.stdout.write(
                f"FAIL: {me.stem} — diff {expected_path.name} {actual_path.name}\n"
            )
            sys.exit(1)
    else:
        expected_path.write_text(actual, encoding="utf-8")
        sys.stdout.write(f"GENERATED: {expected_path.name} ({n} checks)\n")


class _CommandA(m.Command):
    value: str


class _CommandB(m.Command):
    amount: int


class _ProtocolHandler:
    def __init__(self, label: str, message_type: type[object]) -> None:
        self._label = label
        self.message_type = message_type

    def handle(self, message: t.ContainerValue) -> r[t.ContainerValue]:
        value = ""
        if hasattr(message, "value"):
            value = str(getattr(message, "value"))
        if hasattr(message, "amount"):
            value = str(getattr(message, "amount"))
        return r[t.ContainerValue].ok(f"{self._label}:{value}")

    def can_handle(self, message_type: type) -> bool:
        return message_type is self.message_type

    def _protocol_name(self) -> str:
        return f"example-protocol-handler::{self._label}"


@FlextHandlers.handler(_CommandA, priority=3)
def _discovered_handler(_message: t.ContainerValue) -> t.ContainerValue:
    return "decorated"


def _exercise_create_and_service_methods() -> tuple[FlextRegistry, FlextDispatcher]:
    _section("create_and_service_methods")

    dispatcher = FlextDispatcher()
    reg_default = FlextRegistry.create()
    reg_explicit = FlextRegistry.create(dispatcher=None)
    reg_auto_false = FlextRegistry.create(auto_discover_handlers=False)
    reg_auto_true = FlextRegistry.create(auto_discover_handlers=True)

    _check("create.default.type", type(reg_default).__name__)
    _check("create.explicit.type", type(reg_explicit).__name__)
    _check("create.auto_false.type", type(reg_auto_false).__name__)
    _check("create.auto_true.type", type(reg_auto_true).__name__)
    _check(
        "decorated_handler.type",
        type(_discovered_handler(_CommandA(value="d"))).__name__,
    )
    _check("execute.success", reg_explicit.execute().is_success)
    _check(
        "validate_business_rules.success",
        reg_explicit.validate_business_rules().is_success,
    )
    _check("is_valid", reg_explicit.is_valid())
    _check("service_info", reg_explicit.get_service_info())
    _check("result_property", reg_explicit.result)
    _check("runtime.type", type(reg_explicit.runtime).__name__)
    _check("context.type", type(reg_explicit.context).__name__)
    _check("config.type", type(reg_explicit.config).__name__)
    _check("container.type", type(reg_explicit.container).__name__)

    return reg_explicit, dispatcher


def _exercise_summary_and_mixins(registry: FlextRegistry) -> None:
    _section("summary_and_mixins")

    summary_ok = FlextRegistry.Summary()
    summary_fail = FlextRegistry.Summary(errors=["e1"])

    _check("summary.ok.success", summary_ok.is_success)
    _check("summary.ok.failure", summary_ok.is_failure)
    _check("summary.fail.success", summary_fail.is_success)
    _check("summary.fail.failure", summary_fail.is_failure)

    ok_result = registry.ok("value")
    fail_result = registry.fail(
        "boom", error_code="E_REG", error_data=m.ConfigMap(root={"k": "v"})
    )
    _check("mixin.ok.unwrap_or", ok_result.unwrap_or("x"))
    _check("mixin.fail.error", fail_result.error)
    _check("mixin.fail.error_code", fail_result.error_code)

    _check("ensure_result.raw", registry.ensure_result(99).unwrap_or(0))
    _check("ensure_result.existing", registry.ensure_result(r[int].ok(5)).unwrap_or(0))
    _check("to_dict.none", registry.to_dict(None))
    _check("to_dict.mapping", registry.to_dict({"a": 1, "b": "x"}))
    _check(
        "to_dict.basemodel",
        registry.to_dict(m.Handler(handler_name="h", handler_id="id-1")),
    )

    _check("generate_id.len", len(registry.generate_id()))
    _check(
        "generate_prefixed_id",
        registry.generate_prefixed_id("reg", 6).startswith("reg_"),
    )
    _check(
        "generate_datetime_utc.type", type(registry.generate_datetime_utc()).__name__
    )


def _exercise_registration_and_dispatch(
    registry: FlextRegistry,
    dispatcher: FlextDispatcher,
) -> tuple[_ProtocolHandler, _ProtocolHandler]:
    _section("registration_and_dispatch")

    handler_a = _ProtocolHandler("A", _CommandA)
    handler_b = _ProtocolHandler("B", _CommandB)
    handler_mode = FlextHandlers.create_from_callable(
        lambda msg: f"C:{msg}",
        handler_name="mode-handler",
        mode=c.Cqrs.HandlerType.COMMAND,
    )

    reg_one = registry.register_handler(handler_a)
    reg_dup = registry.register_handler(handler_a)
    reg_two = registry.register_handler(handler_b)
    reg_mode = registry.register_handler(handler_a)

    _check("register_handler.a.success", reg_one.is_success)
    _check(
        "register_handler.a.id",
        reg_one.value.registration_id if reg_one.is_success else "",
    )
    _check("register_handler.duplicate.success", reg_dup.is_success)
    _check("register_handler.b.success", reg_two.is_success)
    _check("register_handler.mode.success", reg_mode.is_success)
    _check("create_from_callable.type", type(handler_mode).__name__)

    batch = registry.register_handlers([handler_a, handler_b, handler_a])
    _check("register_handlers.success", batch.is_success)
    _check(
        "register_handlers.registered_len",
        len(batch.value.registered) if batch.is_success else -1,
    )
    _check(
        "register_handlers.errors_len",
        len(batch.value.errors) if batch.is_success else -1,
    )

    cmd_a = _CommandA(value="alpha")
    dispatch_a = dispatcher.dispatch(cmd_a)
    _check("dispatch.a.success", dispatch_a.is_success)
    _check("dispatch.a.value", dispatch_a.unwrap_or(""))

    cmd_b = _CommandB(amount=7)
    dispatch_b = dispatcher.dispatch(cmd_b)
    _check("dispatch.b.success", dispatch_b.is_success)
    _check("dispatch.b.value", dispatch_b.unwrap_or(""))

    return handler_a, handler_b


def _exercise_bindings_and_plugin_apis(
    registry: FlextRegistry,
    handler_a: _ProtocolHandler,
    handler_b: _ProtocolHandler,
) -> None:
    _section("bindings_and_plugins")

    bindings_result = registry.register_bindings({
        _CommandA: handler_a,
        "custom-binding": handler_b,
    })
    _check("register_bindings.success", bindings_result.is_success)
    _check(
        "register_bindings.registered_len",
        len(bindings_result.value.registered) if bindings_result.is_success else -1,
    )

    plugin_ok = registry.register_plugin("transports", "http", "plugin-http")
    plugin_dup = registry.register_plugin("transports", "http", "plugin-http")
    plugin_empty = registry.register_plugin("transports", "", "plugin-http")
    plugin_validated = registry.register_plugin(
        "transports",
        "grpc",
        "plugin-grpc",
        validate=lambda pval: r[bool].ok(bool(pval)),
    )
    plugin_validate_fail = registry.register_plugin(
        "transports",
        "bad",
        "x",
        validate=lambda _pval: r[bool].fail("invalid"),
    )
    plugin_validate_exc = registry.register_plugin(
        "transports",
        "explode",
        "x",
        validate=lambda _pval: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    _check("register_plugin.ok", plugin_ok.is_success)
    _check("register_plugin.dup", plugin_dup.is_success)
    _check("register_plugin.empty_name", plugin_empty.is_failure)
    _check("register_plugin.validated", plugin_validated.is_success)
    _check("register_plugin.validate_fail", plugin_validate_fail.is_failure)
    _check("register_plugin.validate_exc", plugin_validate_exc.is_failure)

    plugin_get_ok = registry.get_plugin("transports", "http")
    plugin_get_missing = registry.get_plugin("transports", "missing")
    plugin_list = registry.list_plugins("transports")
    plugin_unreg_ok = registry.unregister_plugin("transports", "http")
    plugin_unreg_missing = registry.unregister_plugin("transports", "missing")

    _check("get_plugin.ok", plugin_get_ok.unwrap_or(""))
    _check("get_plugin.missing", plugin_get_missing.is_failure)
    _check("list_plugins.transports", sorted(plugin_list.unwrap_or([])))
    _check("unregister_plugin.ok", plugin_unreg_ok.is_success)
    _check("unregister_plugin.missing", plugin_unreg_missing.is_failure)

    class_ok = registry.register_class_plugin("auth", "jwt", "class-jwt")
    class_dup = registry.register_class_plugin("auth", "jwt", "class-jwt")
    class_empty = registry.register_class_plugin("auth", "", "class-jwt")
    class_get_ok = registry.get_class_plugin("auth", "jwt")
    class_get_missing = registry.get_class_plugin("auth", "missing")
    class_list = registry.list_class_plugins("auth")
    class_unreg_ok = registry.unregister_class_plugin("auth", "jwt")
    class_unreg_missing = registry.unregister_class_plugin("auth", "missing")

    _check("register_class_plugin.ok", class_ok.is_success)
    _check("register_class_plugin.dup", class_dup.is_success)
    _check("register_class_plugin.empty_name", class_empty.is_failure)
    _check("get_class_plugin.ok", class_get_ok.unwrap_or(""))
    _check("get_class_plugin.missing", class_get_missing.is_failure)
    _check("list_class_plugins.auth", class_list.unwrap_or([]))
    _check("unregister_class_plugin.ok", class_unreg_ok.is_success)
    _check("unregister_class_plugin.missing", class_unreg_missing.is_failure)


def _exercise_register_method_and_tracking(registry: FlextRegistry) -> None:
    _section("register_method_and_tracking")

    meta_dict = m.ConfigMap(root={"team": "core", "version": "1"})
    meta_model = m.Metadata(attributes={"owner": "registry", "enabled": True})

    reg_plain = registry.register("svc.plain", "value")
    reg_meta_dict = registry.register("svc.dict", "payload", metadata=meta_dict)
    reg_meta_model = registry.register(
        "svc.meta",
        lambda: "callable-service",
        metadata=meta_model,
    )
    reg_bad = registry.register("", "bad")

    _check("register.service.plain", reg_plain.is_success)
    _check("register.service.meta_dict", reg_meta_dict.is_success)
    _check("register.service.meta_model", reg_meta_model.is_success)
    _check("register.service.bad", reg_bad.is_failure)

    with registry.track("example_track") as metrics:
        _check("track.has_operation_count", "operation_count" in metrics)
        _check("track.operation_count", metrics.get("operation_count", -1))


def main() -> None:
    registry, dispatcher = _exercise_create_and_service_methods()
    _exercise_summary_and_mixins(registry)
    handler_a, handler_b = _exercise_registration_and_dispatch(registry, dispatcher)
    _exercise_bindings_and_plugin_apis(registry, handler_a, handler_b)
    _exercise_register_method_and_tracking(registry)
    _verify()


if __name__ == "__main__":
    main()
