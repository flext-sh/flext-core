"""Golden-file example for FlextContainer public APIs."""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType

from flext_core import FlextContainer, FlextContext, FlextRuntime, c, r, t, u

_RESULTS: list[str] = []


def _check(label: str, value: t.ContainerValue | None) -> None:
    """Record a labeled value for golden-file verification."""
    _RESULTS.append(f"{label}: {_ser(value)}")


def _section(name: str) -> None:
    """Start a new output section in the golden stream."""
    if _RESULTS:
        _RESULTS.append("")
    _RESULTS.append(f"[{name}]")


def _ser(v: t.ContainerValue | None) -> str:
    """Serialize values deterministically for stable golden files."""
    if v is None:
        return "None"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        return repr(v)
    if (
        isinstance(v, Sequence)
        and not isinstance(v, (str, bytes, bytearray))
        and u.is_list(v)
    ):
        return "[" + ", ".join(_ser(item) for item in v) + "]"
    if isinstance(v, Mapping):
        if not u.is_dict_like(v):
            return type(v).__name__
        pairs = ", ".join(
            f"{_ser(key)}: {_ser(val)}"
            for key, val in sorted(v.items(), key=lambda kv: str(kv[0]))
        )
        return "{" + pairs + "}"
    if isinstance(v, type):
        return v.__name__
    return type(v).__name__


def _verify() -> None:
    """Compare current output with .expected and report result."""
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


class _WireProbe:
    """Probe class used to exercise wire_modules(classes=...)."""


def _exercise_singleton_and_creation() -> FlextContainer:
    """Exercise get_global/create/Builder creation paths and singleton semantics."""
    _section("singleton_and_creation")

    FlextContainer.reset_singleton_for_testing()
    root_context = FlextContext()
    root = FlextContainer.get_global(context=root_context)

    _check("get_global.type", type(root).__name__)
    _check("get_global.context.type", type(root.context).__name__)
    _check("get_global.config.type", type(root.config).__name__)
    _check("get_global.same_instance", root is FlextContainer.get_global())

    created_false = FlextContainer.create(auto_register_factories=False)
    created_true = FlextContainer.create(auto_register_factories=True)
    builder_false = FlextContainer.Builder.create(auto_register_factories=False)
    builder_true = FlextContainer.Builder.create(auto_register_factories=True)

    _check("create.false.same_instance", created_false is root)
    _check("create.true.same_instance", created_true is root)
    _check("builder.create.false.same_instance", builder_false is root)
    _check("builder.create.true.same_instance", builder_true is root)
    _check("result.ok.unwrap_or", r[int].ok(7).unwrap_or(0))
    _check("runtime.normalize.bool", FlextRuntime.normalize_to_general_value(True))
    _check("constants.default_max_services", c.Container.DEFAULT_MAX_SERVICES)

    return root


def _exercise_registration_and_resolution(container: FlextContainer) -> None:
    """Exercise register APIs plus get/get_typed/list/has checks."""
    _section("registration_and_resolution")

    register_ok = container.register("svc.number", 41)
    register_dup = container.register("svc.number", 42)
    register_empty = container.register("", 5)

    _check("register.service.success", register_ok.is_success)
    _check("register.service.value", register_ok.unwrap_or(False))
    _check("register.service.duplicate_failure", register_dup.is_failure)
    _check("register.service.empty_name_failure", register_empty.is_failure)

    factory_calls = {"count": 0}

    def _factory_counter() -> int:
        factory_calls["count"] = int(factory_calls["count"]) + 1
        return int(factory_calls["count"])

    register_factory_ok = container.register_factory(
        "factory.counter", _factory_counter
    )
    register_factory_dup = container.register_factory(
        "factory.counter", _factory_counter
    )

    def _factory_raises() -> int:
        error_message = "factory boom"
        raise RuntimeError(error_message)

    register_factory_bad = container.register_factory("factory.bad", _factory_raises)

    _check("register.factory.success", register_factory_ok.is_success)
    _check("register.factory.duplicate_failure", register_factory_dup.is_failure)
    _check("register.factory.bad_registration_success", register_factory_bad.is_success)

    resource_calls = {"count": 0}

    def _resource_data() -> dict[str, int]:
        resource_calls["count"] = int(resource_calls["count"]) + 1
        return {"resource": int(resource_calls["count"])}

    register_resource_ok = container.register_resource("resource.data", _resource_data)
    register_resource_dup = container.register_resource("resource.data", _resource_data)

    _check("register.resource.success", register_resource_ok.is_success)
    _check("register.resource.duplicate_failure", register_resource_dup.is_failure)

    get_service = container.get("svc.number")
    get_factory = container.get("factory.counter")
    get_resource = container.get("resource.data")
    get_missing = container.get("missing.name")
    get_bad_factory = container.get("factory.bad")

    _check("get.service.success", get_service.is_success)
    _check("get.service.value", container.get_typed("svc.number", int).unwrap_or(-1))
    _check("get.factory.success", get_factory.is_success)
    _check(
        "get.factory.value", container.get_typed("factory.counter", int).unwrap_or(-1)
    )
    _check("get.resource.success", get_resource.is_success)
    _check("get.resource.call_count", resource_calls["count"])
    _check("get.missing.failure", get_missing.is_failure)
    _check("get.bad_factory.failure", get_bad_factory.is_failure)

    get_typed_service = container.get_typed("svc.number", int)
    get_typed_service_bad = container.get_typed("svc.number", str)
    get_typed_factory = container.get_typed("factory.counter", int)
    get_typed_missing = container.get_typed("missing.name", int)

    _check("get_typed.service.success", get_typed_service.is_success)
    _check("get_typed.service.value", get_typed_service.unwrap_or(-1))
    _check("get_typed.service.type_mismatch_failure", get_typed_service_bad.is_failure)
    _check("get_typed.factory.success", get_typed_factory.is_success)
    _check(
        "get_typed.resource_via_get.success", container.get("resource.data").is_success
    )
    _check("get_typed.missing.failure", get_typed_missing.is_failure)

    _check("has_service.service.true", container.has_service("svc.number"))
    _check("has_service.factory.true", container.has_service("factory.counter"))
    _check("has_service.resource.true", container.has_service("resource.data"))
    _check("has_service.missing.false", container.has_service("missing.name"))

    service_list = list(container.list_services())
    _check("list_services.contains.svc_number", "svc.number" in service_list)
    _check("list_services.contains.factory_counter", "factory.counter" in service_list)
    _check("list_services.contains.resource_data", "resource.data" in service_list)


def _exercise_fluent_and_config(container: FlextContainer) -> None:
    """Exercise fluent registration/configuration APIs."""
    _section("fluent_and_config")

    with_service_result = container.with_service("fluent.service", "ready")
    with_factory_result = container.with_factory(
        "fluent.factory", lambda: "from-factory"
    )
    with_resource_result = container.with_resource(
        "fluent.resource",
        lambda: "from-resource",
    )
    with_config_result = container.with_config({"max_factories": 111})

    _check("with_service.returns_self", with_service_result is container)
    _check("with_factory.returns_self", with_factory_result is container)
    _check("with_resource.returns_self", with_resource_result is container)
    _check("with_config.returns_self", with_config_result is container)

    container.configure({"max_services": 222, "enable_factory_caching": True})
    config_map = container.get_config()
    max_services = config_map["max_services"]
    enable_factory_caching = config_map["enable_factory_caching"]

    max_services_num = max_services if isinstance(max_services, int) else -1
    factory_cache_flag = (
        enable_factory_caching if isinstance(enable_factory_caching, bool) else False
    )

    _check("configure.get_config.max_services", max_services_num)
    _check(
        "configure.get_config.enable_factory_caching",
        factory_cache_flag,
    )
    _check(
        "with_service.get.value",
        container.get_typed("fluent.service", str).unwrap_or(""),
    )
    _check(
        "with_factory.get.value",
        container.get_typed("fluent.factory", str).unwrap_or(""),
    )
    _check(
        "with_resource.get.value",
        container.get_typed("fluent.resource", str).unwrap_or(""),
    )


def _exercise_wiring_and_scoped(container: FlextContainer) -> FlextContainer:
    """Exercise wire_modules and scoped with all supported parameter styles."""
    _section("wiring_and_scoped")

    this_module: ModuleType = sys.modules[__name__]
    container.wire_modules(modules=[this_module])
    container.wire_modules(packages=[])
    container.wire_modules(classes=[_WireProbe])
    _check("wire_modules.calls_completed", True)

    scoped_default = container.scoped()
    scoped_subproject = container.scoped(subproject="alpha")
    explicit_context = FlextContext()
    explicit_config = container.config.model_copy(
        update={"app_name": "scoped.explicit"}
    )
    scoped_full = container.scoped(
        config=explicit_config,
        context=explicit_context,
        subproject="beta",
        services={"scoped.service": "scoped"},
        factories={"scoped.factory": lambda: 700},
        resources={"scoped.resource": lambda: {"res": "ok"}},
    )

    _check("scoped.default.new_instance", scoped_default is not container)
    _check("scoped.default.inherits_service", scoped_default.has_service("svc.number"))
    _check(
        "scoped.default.get_typed_service",
        scoped_default.get_typed("svc.number", int).unwrap_or(-1),
    )
    _check(
        "scoped.subproject.app_name_suffix",
        scoped_subproject.config.app_name.endswith(".alpha"),
    )
    _check("scoped.full.new_instance", scoped_full is not container)
    _check("scoped.full.config_app_name", scoped_full.config.app_name)
    _check("scoped.full.uses_explicit_context", scoped_full.context is explicit_context)
    _check("scoped.full.has_service", scoped_full.has_service("scoped.service"))
    _check("scoped.full.has_factory", scoped_full.has_service("scoped.factory"))
    _check("scoped.full.has_resource", scoped_full.has_service("scoped.resource"))
    _check(
        "scoped.full.get_service",
        scoped_full.get_typed("scoped.service", str).unwrap_or(""),
    )
    _check(
        "scoped.full.get_factory",
        scoped_full.get_typed("scoped.factory", int).unwrap_or(-1),
    )
    _check(
        "scoped.full.get_resource.success",
        scoped_full.get("scoped.resource").is_success,
    )

    return scoped_full


def _exercise_internal_and_cleanup(
    container: FlextContainer, root: FlextContainer
) -> None:
    """Exercise internal/public lifecycle helpers and cleanup APIs."""
    _section("internal_and_cleanup")

    container.initialize_di_components()
    _check("initialize_di_components.bridge_exists", hasattr(container, "_di_bridge"))
    _check(
        "initialize_di_components.container_exists",
        hasattr(container, "_di_container"),
    )

    container.initialize_registrations(
        config=root.config.model_copy(deep=True), context=FlextContext()
    )
    _check(
        "initialize_registrations.list_services_empty", len(container.list_services())
    )

    container.sync_config_to_di()
    container.register_existing_providers()
    container.register_core_services()

    _check("sync_config_to_di.service_config_present", container.has_service("config"))
    _check("register_core_services.logger_present", container.has_service("logger"))
    _check(
        "register_core_services.command_bus_present",
        container.has_service("command_bus"),
    )

    logger_default = container.create_module_logger()
    logger_custom = container.create_module_logger("examples.ex_08")
    _check("create_module_logger.default.type", type(logger_default).__name__)
    _check("create_module_logger.custom.type", type(logger_custom).__name__)

    _ = container.register("tmp.remove", 999)
    unregister_ok = container.unregister("tmp.remove")
    unregister_missing = container.unregister("tmp.missing")
    _check("unregister.existing.success", unregister_ok.is_success)
    _check("unregister.missing.failure", unregister_missing.is_failure)

    container.clear_all()
    _check("clear_all.count", len(container.list_services()))

    before_reset = root
    FlextContainer.reset_singleton_for_testing()
    after_reset = FlextContainer.get_global(context=FlextContext())
    _check("reset_singleton.new_instance", before_reset is not after_reset)
    _check(
        "reset_singleton.get_global.same_after_reset",
        after_reset is FlextContainer.get_global(),
    )


def main() -> None:
    """Run all sections and verify against the golden file."""
    root = _exercise_singleton_and_creation()
    _exercise_registration_and_resolution(root)
    _exercise_fluent_and_config(root)
    scoped_full = _exercise_wiring_and_scoped(root)
    _exercise_internal_and_cleanup(scoped_full, root)
    _verify()


if __name__ == "__main__":
    main()
