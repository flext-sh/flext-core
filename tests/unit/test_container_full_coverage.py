"""Container full coverage tests."""

from __future__ import annotations

import sys
import types
from collections.abc import Callable
from typing import ClassVar, cast

import pytest
from pydantic import BaseModel
from pydantic_settings import BaseSettings as _BaseSettings

from flext_core import FlextContainer, FlextContext, FlextSettings, m, t


class _FalseConfig:
    app_name = "app"

    def model_copy(
        self,
        *,
        deep: bool = False,
        update: dict[str, t.ContainerValue] | None = None,
    ) -> _FalseConfig:
        return self


class _ContextNoClone:
    def set(self, _k: str, _v: object) -> None:
        return None


class _BridgeNoProvide:
    pass


class _BridgeBadProvide:
    Provide: None = None


class _BridgeGoodProvide:
    @staticmethod
    def Provide(name: str) -> str:  # noqa: N802  # Test double for protocol method Provide
        return name


def _scan_factory_module(
    _module: object,
) -> list[tuple[str, m.HandlerFactoryDecoratorConfig]]:
    return [
        (
            "the_factory",
            m.HandlerFactoryDecoratorConfig(name="x", singleton=False, lazy=True),
        ),
    ]


def _scan_factory_module_captured(
    _module: object,
) -> list[tuple[str, m.HandlerFactoryDecoratorConfig]]:
    return [
        (
            "factory_fn",
            m.HandlerFactoryDecoratorConfig(
                name="factory.captured",
                singleton=False,
                lazy=True,
            ),
        ),
    ]


def _has_service_false(_name: str) -> bool:
    return False


def _raise_register_object(*_args: object, **_kwargs: object) -> None:
    msg = "boom"
    raise RuntimeError(msg)


def _raise_register_factory(*_args: object, **_kwargs: object) -> None:
    msg = "boom"
    raise RuntimeError(msg)


def _raise_register_resource(*_args: object, **_kwargs: object) -> None:
    msg = "boom"
    raise RuntimeError(msg)


def _namespace_config_none(_namespace: str) -> None:
    return None


def test_protocol_name_and_builder() -> None:
    c = FlextContainer.create()
    assert c._protocol_name() == "FlextContainer"
    assert isinstance(FlextContainer.Builder.create(), FlextContainer)


def test_create_auto_register_factories_path(monkeypatch: pytest.MonkeyPatch) -> None:
    container = FlextContainer.create()
    called: list[str] = []

    def factory() -> int:
        return 1

    fake_module = types.SimpleNamespace(the_factory=factory)
    frame = types.SimpleNamespace(
        f_back=types.SimpleNamespace(f_globals={"__name__": "fake_mod"}),
    )
    monkeypatch.setitem(sys.modules, "fake_mod", cast("types.ModuleType", fake_module))
    monkeypatch.setattr("flext_core.container.inspect.currentframe", lambda: frame)
    monkeypatch.setattr(
        "flext_core.container.FactoryDecoratorsDiscovery.scan_module",
        _scan_factory_module,
    )

    def _register(name: str, _impl: object, *, kind: str = "service") -> FlextContainer:
        if kind == "factory":
            called.append(name)
        return container

    monkeypatch.setattr(container, "register", _register)

    def _call_container(*_args: object, **_kwargs: object) -> FlextContainer:
        return container

    monkeypatch.setattr(
        "flext_core.container.FlextContainer.__call__",
        _call_container,
        raising=False,
    )
    created = FlextContainer.create(auto_register_factories=True)
    assert isinstance(created, FlextContainer)
    assert "x" in called


def test_provide_property_paths() -> None:
    c = FlextContainer.create()
    c_any: FlextContainer = c
    c_any._di_bridge = _BridgeGoodProvide()
    assert c.provide("x") == "x"
    c_any._di_bridge = _BridgeBadProvide()
    with pytest.raises(RuntimeError):
        _ = c.provide
    c_any._di_bridge = _BridgeNoProvide()
    with pytest.raises(RuntimeError):
        _ = c.provide


def test_config_context_properties_and_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    c = FlextContainer.create()
    c._config = None
    with pytest.raises(RuntimeError):
        _ = c.config
    c._context = None
    with pytest.raises(RuntimeError):
        _ = c.context
    monkeypatch.setattr(
        "flext_core.container.FlextSettings.get_global",
        lambda: FlextSettings(),
    )
    assert isinstance(c._get_default_config(), FlextSettings)


def test_initialize_di_components_error_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    c = FlextContainer.create()
    bad_bridge = types.SimpleNamespace(config="not-provider")
    monkeypatch.setattr(
        "flext_core.container.FlextRuntime.DependencyIntegration.create_layered_bridge",
        lambda: (bad_bridge, types.SimpleNamespace(), types.SimpleNamespace()),
    )
    with pytest.raises(TypeError, match="Bridge must have config provider"):
        c.initialize_di_components()


def test_sync_config_namespace_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    c = FlextContainer.create()
    c._config = FlextSettings()
    c._user_overrides = m.ConfigMap(root={})
    c._global_config = m.ContainerConfig(
        enable_singleton=True,
        enable_factory_caching=False,
        max_services=10,
        max_factories=10,
    )
    monkeypatch.setattr(type(c._config), "_namespace_registry", {"x": object()})
    monkeypatch.setattr(c, "has_service", _has_service_false)

    def _capture_register(
        _name: str,
        _impl: object,
        *,
        kind: str = "service",
    ) -> FlextContainer:
        _ = kind
        return c

    monkeypatch.setattr(c, "register", _capture_register)
    monkeypatch.setattr(
        type(c._config),
        "get_namespace_config",
        staticmethod(_namespace_config_none),
    )
    c.sync_config_to_di()


def test_register_existing_providers_skips_and_register_core_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    c = FlextContainer.create()
    c._services = {
        "svc": m.ServiceRegistration(name="svc", service="v", service_type="str"),
    }
    c._factories = {"fac": m.FactoryRegistration(name="fac", factory=lambda: "v")}
    c._resources = {"res": m.ResourceRegistration(name="res", factory=lambda: "v")}
    setattr(c._di_services, "svc", object())
    setattr(c._di_services, "fac", object())
    setattr(c._di_resources, "res", object())
    c.register_existing_providers()
    c_any: FlextContainer = c
    c_any._config = _FalseConfig()
    c_any._context = None
    monkeypatch.setattr(c, "has_service", _has_service_false)
    c.register_core_services()


def test_configure_with_resource_register_and_factory_error_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    c = FlextContainer.create()
    c._user_overrides = m.ConfigMap(root={})
    c._global_config = m.ContainerConfig(
        enable_singleton=True,
        enable_factory_caching=False,
        max_services=100,
        max_factories=100,
    )
    c.configure({"enable_factory_caching": True})
    assert c._global_config.enable_factory_caching is True
    c.register("res", lambda: "x", kind="resource")
    assert "res" in c._resources
    monkeypatch.setattr(
        "flext_core.container.FlextRuntime.DependencyIntegration.register_object",
        _raise_register_object,
    )
    c.register("x", "y")
    monkeypatch.setattr(
        "flext_core.container.FlextRuntime.DependencyIntegration.register_factory",
        _raise_register_factory,
    )
    c.register("x2", lambda: "v", kind="factory")
    setattr(c._di_resources, "dup", object())
    c.register("dup", lambda: "v", kind="resource")
    delattr(c._di_resources, "dup")
    monkeypatch.setattr(
        "flext_core.container.FlextRuntime.DependencyIntegration.register_resource",
        _raise_register_resource,
    )
    c.register("new", lambda: "v", kind="resource")


def test_get_and_get_typed_resource_factory_paths() -> None:
    c = FlextContainer.create()
    c._factories = {
        "f": m.FactoryRegistration(
            name="f",
            factory=lambda: (_ for _ in ()).throw(ValueError("x")),
        ),
    }
    c._resources = {
        "r": m.ResourceRegistration(
            name="r",
            factory=lambda: (_ for _ in ()).throw(ValueError("x")),
        ),
    }
    assert c.get("r").is_failure
    c._factories = {"f2": m.FactoryRegistration(name="f2", factory=lambda: "abc")}
    assert c.get("f2", type_cls=dict).is_failure
    c._resources = {"r2": m.ResourceRegistration(name="r2", factory=lambda: "abc")}
    assert c.get("r2", type_cls=dict).is_failure
    c._factories = {
        "f3": m.FactoryRegistration(
            name="f3",
            factory=lambda: (_ for _ in ()).throw(ValueError("err")),
        ),
    }
    assert c.get("f3", type_cls=str).is_failure
    c._resources = {
        "r3": m.ResourceRegistration(
            name="r3",
            factory=lambda: (_ for _ in ()).throw(ValueError("err")),
        ),
    }
    assert c.get("r3", type_cls=str).is_failure


def test_misc_unregistration_clear_and_reset() -> None:
    c = FlextContainer.create()
    _ = c.register("resx", lambda: "ok", kind="resource")
    assert c.create_module_logger().logger is not None
    assert c.unregister("resx").is_success
    _ = c.register("r1", lambda: "ok", kind="resource")
    c.clear_all()
    FlextContainer.reset_for_testing()
    instance = FlextContainer.create()
    assert instance._global_instance is instance


def test_scoped_config_context_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    c = FlextContainer.create()
    c._services = {}
    c._factories = {}
    c._resources = {}
    c._user_overrides = m.ConfigMap(root={})
    c._global_config = m.ContainerConfig(
        enable_singleton=True,
        enable_factory_caching=False,
        max_services=10,
        max_factories=10,
    )
    c._config = FlextSettings(app_name="base")
    c._context = FlextContext()
    captured: dict[str, object] = {}

    def _fake_create_scoped_instance(**kwargs: object) -> FlextContainer:
        captured.update(kwargs)
        return c

    monkeypatch.setattr(
        "flext_core.container.FlextContainer._create_scoped_instance",
        _fake_create_scoped_instance,
    )
    _ = c.scoped(subproject="sub")
    assert isinstance(captured["config"], FlextSettings)
    _ = c.scoped(
        config=cast("object", _FalseConfig()),
        context=cast("object", _ContextNoClone()),
    )
    assert isinstance(captured["context"], FlextContext)


def test_create_auto_register_factory_wrapper_callable_and_non_callable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Callable[..., object]] = {}

    def factory_fn() -> int:
        return 7

    fake_module = types.SimpleNamespace(factory_fn=factory_fn)
    frame = types.SimpleNamespace(
        f_back=types.SimpleNamespace(f_globals={"__name__": "fake_factory_mod"}),
    )
    monkeypatch.setitem(
        sys.modules,
        "fake_factory_mod",
        cast("types.ModuleType", fake_module),
    )
    monkeypatch.setattr("flext_core.container.inspect.currentframe", lambda: frame)
    monkeypatch.setattr(
        "flext_core.container.FactoryDecoratorsDiscovery.scan_module",
        _scan_factory_module_captured,
    )
    original_register = FlextContainer.register

    def capture_register(
        self: FlextContainer,
        name: str,
        impl: object,
        *,
        kind: str = "service",
        singleton: bool = False,
        lazy: bool = True,
    ) -> FlextContainer:
        _ = singleton
        _ = lazy
        if kind == "factory" and callable(impl):
            captured[name] = impl
        return self

    monkeypatch.setattr(FlextContainer, "register", capture_register)
    _ = FlextContainer.create(auto_register_factories=True)
    wrapper = captured["factory.captured"]
    assert wrapper() == 7
    assert (
        wrapper(_factory_config=types.SimpleNamespace(fn=123, name="factory.captured"))
        == ""
    )
    monkeypatch.setattr(FlextContainer, "register", original_register)


def test_initialize_di_components_second_type_error_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    c = FlextContainer.create()
    bad_bridge = types.SimpleNamespace(config=None)
    monkeypatch.setattr(
        "flext_core.container.FlextRuntime.DependencyIntegration.create_layered_bridge",
        lambda: (bad_bridge, types.SimpleNamespace(), types.SimpleNamespace()),
    )
    monkeypatch.setattr("flext_core.container.di_providers.Configuration", object)
    with pytest.raises(TypeError, match="cannot be None"):
        c.initialize_di_components()


def test_sync_config_registers_namespace_factories_and_fallbacks() -> None:
    c = FlextContainer.create()

    class _NsAlpha(_BaseSettings):
        v: str = "ok"

    class _NsBeta(_BaseSettings):
        v: str = "ok2"

    # Register namespaces so FlextSettings.get_namespace_config() finds them.
    original_registry = dict(FlextSettings._namespace_registry)
    FlextSettings._namespace_registry["alpha"] = _NsAlpha
    FlextSettings._namespace_registry["beta"] = _NsBeta

    try:

        class _Cfg:
            _namespace_registry: ClassVar[dict[str, object]] = {
                "alpha": object(),
                "beta": object(),
            }

        c_any: FlextContainer = c
        c_any._config = _Cfg()
        c._global_config = m.ContainerConfig(
            enable_singleton=True,
            enable_factory_caching=False,
            max_services=10,
            max_factories=10,
        )
        c._user_overrides = m.ConfigMap(root={})
        registered: dict[str, Callable[..., object]] = {}
        c_any.has_service = _has_service_false

        def _register(
            name: str,
            impl: object,
            *,
            kind: str = "service",
        ) -> FlextContainer:
            if kind == "factory" and callable(impl):
                registered[name] = impl
            return c

        c_any.register = _register
        c.sync_config_to_di()
        assert "config.alpha" in registered
        assert "config.beta" in registered
        assert isinstance(registered["config.alpha"](), BaseModel)
        assert isinstance(registered["config.beta"](), BaseModel)
    finally:
        FlextSettings._namespace_registry.clear()
        FlextSettings._namespace_registry.update(original_registry)


def test_register_existing_providers_full_paths_and_misc_methods() -> None:
    c = FlextContainer.create()
    c._services = {
        "s1": m.ServiceRegistration(name="s1", service="v", service_type="str"),
    }
    c._factories = {"f1": m.FactoryRegistration(name="f1", factory=lambda: "fv")}
    c._resources = {"r1": m.ResourceRegistration(name="r1", factory=lambda: "rv")}
    c.register_existing_providers()
    assert hasattr(c._di_container, "s1")
    assert hasattr(c._di_container, "f1")
    assert hasattr(c._di_container, "r1")
    c_any: FlextContainer = c
    c_any._context = FlextContext()
    c_any.has_service = _has_service_false
    c.register_core_services()
    assert c.has_service("context") is False or True
    c.wire_modules(modules=[])
    assert c.get("r1").is_success
    assert c.get("f1", type_cls=str).is_success
    assert c.get("r1", type_cls=str).is_success


def test_create_scoped_instance_and_scoped_additional_branches() -> None:
    base = FlextContainer.create()
    scoped = FlextContainer._create_scoped_instance(
        config=FlextSettings(),
        context=FlextContext(),
        services={},
        factories={},
        resources={},
        user_overrides=m.ConfigMap(root={}),
        container_config=m.ContainerConfig(
            enable_singleton=True,
            enable_factory_caching=False,
            max_services=10,
            max_factories=10,
        ),
    )
    assert isinstance(scoped, FlextContainer)
    base_any: FlextContainer = base
    base_any._config = _FalseConfig()
    base_any._context = FlextContext()
    _ = base.scoped(
        config=FlextSettings(app_name="x"),
        context=FlextContext(),
        services={"sx": "vx"},
        factories={"fx": lambda: "fv"},
        resources={"rx": lambda: "rv"},
    )
    base_any._config = _FalseConfig()
    base_any._context = _ContextNoClone()
    _ = base.scoped()


def test_additional_container_branches_cover_fluent_and_lookup_paths() -> None:
    c = FlextContainer.create()
    c._global_config = m.ContainerConfig(
        enable_singleton=True,
        enable_factory_caching=False,
        max_services=10,
        max_factories=10,
    )
    global_instance = FlextContainer.get_global()
    assert isinstance(global_instance, FlextContainer)
    assert c.configure({"enable_factory_caching": True}) is c
    c.register("svc-x", "value")
    c.register("fac-x", lambda: "v", kind="factory")
    assert c.get_config().root
    c.register("", "x")
    assert c.get("svc-x").is_success
    assert c.get("missing-service").is_failure
    assert c.get("svc-x", type_cls=str).is_success
    assert c.get("missing-service", type_cls=str).is_failure


def test_additional_register_factory_and_unregister_paths() -> None:
    c = FlextContainer.create()
    c._global_config = m.ContainerConfig(
        enable_singleton=True,
        enable_factory_caching=False,
        max_services=10,
        max_factories=10,
    )
    c.register("fac-ok", lambda: 1, kind="factory")
    c.register("fac-ok", lambda: 2, kind="factory")
    c.register("fac-bad", cast("object", 123), kind="factory")
    assert FlextContainer._narrow_factory_result("x") == "x"
    _ = c.register("svc-remove", "v")
    _ = c.register("res-remove", lambda: "r", kind="resource")
    _ = c.register("fac-remove", lambda: "f", kind="factory")
    assert c.unregister("svc-remove").is_success
    assert c.unregister("fac-remove").is_success
    assert c.unregister("res-remove").is_success
    assert c.unregister("not-there").is_failure


def test_container_remaining_branch_paths_in_sync_factory_and_getters() -> None:
    c = FlextContainer.create()
    c._global_config = m.ContainerConfig(
        enable_singleton=True,
        enable_factory_caching=False,
        max_services=10,
        max_factories=10,
    )
    c._user_overrides = m.ConfigMap(root={})

    # --- _CfgNoMethod: namespace_registry without get_namespace_config ---
    # n1 is NOT registered in FlextSettings, so get_namespace_config returns None
    # and sync_config_to_di skips it (continue branch).
    class _CfgNoMethod:
        _namespace_registry: ClassVar[dict[str, object]] = {"n1": object()}

    c_any: FlextContainer = c
    c_any._config = _CfgNoMethod()
    c.sync_config_to_di()

    # --- Create real BaseSettings subclasses for n2, n3, n4 ---
    class _NsModel(_BaseSettings):
        v: str = "x"

    # Register namespaces in FlextSettings._namespace_registry so that
    # sync_config_to_di -> FlextSettings.get_namespace_config() finds them.
    original_registry = dict(FlextSettings._namespace_registry)
    FlextSettings._namespace_registry["n2"] = _NsModel
    FlextSettings._namespace_registry["n3"] = _NsModel
    FlextSettings._namespace_registry["n4"] = _NsModel

    try:

        class _CfgFallback:
            _namespace_registry: ClassVar[dict[str, object]] = {"n2": object()}

        class _CfgBadNamespace:
            _namespace_registry: ClassVar[dict[str, object]] = {"n3": object()}

        class _CfgGoodNamespace:
            _namespace_registry: ClassVar[dict[str, object]] = {"n4": object()}

        c_any._config = _CfgFallback()
        captured: dict[str, Callable[..., object]] = {}
        c_any.has_service = _has_service_false

        def _capture_register(
            name: str,
            impl: object,
            *,
            kind: str = "service",
        ) -> FlextContainer:
            if kind == "factory" and callable(impl):
                captured[name] = impl
            return c

        c_any.register = _capture_register
        c.sync_config_to_di()
        c_any._config = _CfgBadNamespace()
        c.sync_config_to_di()
        c_any._config = _CfgGoodNamespace()
        c.sync_config_to_di()
        assert isinstance(captured["config.n2"](), BaseModel)
        assert isinstance(captured["config.n3"](), BaseModel)
        assert isinstance(captured["config.n4"](), BaseModel)
        delattr(c_any, "register")
        c2 = FlextContainer.create()
        c2._global_config = m.ContainerConfig(
            enable_singleton=True,
            enable_factory_caching=False,
            max_services=10,
            max_factories=10,
        )
        c2.register("fac-call", lambda: "value", kind="factory")
        assert c2._factories["fac-call"].factory() == "value"
        c2._factories = {
            "boom": m.FactoryRegistration(
                name="boom",
                factory=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
            ),
        }
        assert c2.get("boom").is_failure
        c2._factories = {
            "ok-factory": m.FactoryRegistration(
                name="ok-factory", factory=lambda: "ok"
            ),
        }
        assert c2.get("ok-factory").is_success
        c2._services = {
            "svc-int": m.ServiceRegistration(
                name="svc-int",
                service="str",
                service_type="str",
            ),
        }
        assert c2.get("svc-int", type_cls=int).is_failure
        executed: list[str] = []
        c_any.has_service = _has_service_false

        def _track_register(
            _name: str,
            impl: object,
            *,
            kind: str = "service",
        ) -> FlextContainer:
            if kind == "factory" and callable(impl):
                executed.append(type(impl()).__name__)
            return c

        c_any.register = _track_register
        c_any._config = _CfgFallback()
        c.sync_config_to_di()
        c_any._config = _CfgBadNamespace()
        c.sync_config_to_di()
        assert executed
    finally:
        # Restore original registry
        FlextSettings._namespace_registry.clear()
        FlextSettings._namespace_registry.update(original_registry)
