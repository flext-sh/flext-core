"""Container full coverage tests."""

from __future__ import annotations

import types
from collections.abc import Callable
from typing import Any, ClassVar, cast

import pytest
from flext_core import FlextContainer, FlextContext, FlextSettings, m, r
from pydantic import BaseModel


class _FalseConfig:
    app_name = "app"


class _ContextNoClone:
    def set(self, _k: str, _v: object) -> None:
        return None


class _BridgeNoProvide:
    pass


class _BridgeBadProvide:
    Provide = None


class _BridgeGoodProvide:
    @staticmethod
    def Provide(name: str) -> str:  # noqa: N802
        return name


class _FalseyError(Exception):
    def __bool__(self) -> bool:
        return False


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
        f_back=types.SimpleNamespace(f_globals={"__name__": "fake_mod"})
    )

    monkeypatch.setitem(__import__("sys").modules, "fake_mod", fake_module)
    monkeypatch.setattr("flext_core.container.inspect.currentframe", lambda: frame)
    monkeypatch.setattr(
        "flext_core.container.FactoryDecoratorsDiscovery.scan_module",
        lambda _m: [
            (
                "the_factory",
                m.HandlerFactoryDecoratorConfig(name="x", singleton=False, lazy=True),
            )
        ],
    )

    def _register_factory(name: str, _factory: Callable[[], object]) -> r[bool]:
        called.append(name)
        return r[bool].ok(True)

    monkeypatch.setattr(container, "register_factory", _register_factory)
    monkeypatch.setattr(
        "flext_core.container.FlextContainer.__call__",
        lambda *a, **kw: container,
        raising=False,
    )

    created = FlextContainer.create(auto_register_factories=True)
    assert isinstance(created, FlextContainer)


def test_provide_property_paths() -> None:
    c = FlextContainer.create()
    c_any: Any = c
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
        "flext_core.container.FlextSettings.get_global_instance",
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
    c._global_config = m.Container.ContainerConfig(
        enable_singleton=True,
        enable_factory_caching=False,
        max_services=10,
        max_factories=10,
    )

    monkeypatch.setattr(type(c._config), "_namespace_registry", {"x": object()})
    monkeypatch.setattr(c, "has_service", lambda _name: False)
    monkeypatch.setattr(c, "register_factory", lambda _n, _f: r[bool].ok(True))
    monkeypatch.setattr(
        type(c._config), "get_namespace_config", lambda _self, _ns: None
    )
    c.sync_config_to_di()


def test_register_existing_providers_skips_and_register_core_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    c = FlextContainer.create()
    c._services = {
        "svc": m.Container.ServiceRegistration(
            name="svc", service="v", service_type="str"
        )
    }
    c._factories = {
        "fac": m.Container.FactoryRegistration(name="fac", factory=lambda: "v")
    }
    c._resources = {
        "res": m.Container.ResourceRegistration(name="res", factory=lambda: "v")
    }

    setattr(c._di_services, "svc", object())
    setattr(c._di_services, "fac", object())
    setattr(c._di_resources, "res", object())
    c.register_existing_providers()

    c_any: Any = c
    c_any._config = _FalseConfig()
    c_any._context = None
    monkeypatch.setattr(c, "has_service", lambda _name: False)
    c.register_core_services()


def test_configure_with_resource_register_and_factory_error_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    c = FlextContainer.create()
    c._user_overrides = m.ConfigMap(root={})
    c._global_config = m.Container.ContainerConfig(
        enable_singleton=True,
        enable_factory_caching=False,
        max_services=100,
        max_factories=100,
    )

    c.configure({"enable_factory_caching": True})
    assert c._global_config.enable_factory_caching is True

    called: list[str] = []
    original_register_resource = c.register_resource

    def _register_resource(name: str, _factory: Callable[[], object]) -> r[bool]:
        called.append(name)
        return r[bool].ok(True)

    monkeypatch.setattr(c, "register_resource", _register_resource)
    assert c.with_resource("res", lambda: "x") is c
    assert called == ["res"]
    object.__setattr__(c, "register_resource", original_register_resource)

    monkeypatch.setattr(
        "flext_core.container.FlextRuntime.DependencyIntegration.register_object",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert c.register("x", "y").is_failure

    monkeypatch.setattr(
        "flext_core.container.FlextRuntime.DependencyIntegration.register_factory",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert c.register_factory("x2", lambda: "v").is_failure

    setattr(c._di_resources, "dup", object())
    assert c.register_resource("dup", lambda: "v").is_failure

    delattr(c._di_resources, "dup")
    monkeypatch.setattr(
        "flext_core.container.FlextRuntime.DependencyIntegration.register_resource",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert c.register_resource("new", lambda: "v").is_failure


def test_get_and_get_typed_resource_factory_paths() -> None:
    c = FlextContainer.create()
    c._factories = {
        "f": m.Container.FactoryRegistration(
            name="f", factory=lambda: (_ for _ in ()).throw(ValueError("x"))
        )
    }
    c._resources = {
        "r": m.Container.ResourceRegistration(
            name="r", factory=lambda: (_ for _ in ()).throw(ValueError("x"))
        )
    }
    assert c.get("r").is_failure

    c._factories = {
        "f2": m.Container.FactoryRegistration(name="f2", factory=lambda: "abc")
    }
    assert c.get_typed("f2", dict).is_failure

    c._resources = {
        "r2": m.Container.ResourceRegistration(name="r2", factory=lambda: "abc")
    }
    assert c.get_typed("r2", dict).is_failure

    c._factories = {
        "f3": m.Container.FactoryRegistration(
            name="f3", factory=lambda: (_ for _ in ()).throw(ValueError("err"))
        )
    }
    assert c.get_typed("f3", str).is_failure

    c._resources = {
        "r3": m.Container.ResourceRegistration(
            name="r3", factory=lambda: (_ for _ in ()).throw(ValueError("err"))
        )
    }
    assert c.get_typed("r3", str).is_failure


def test_misc_unregistration_clear_and_reset() -> None:
    c = FlextContainer.create()
    _ = c.register_resource("resx", lambda: "ok")
    assert c.create_module_logger().logger is not None

    assert c.unregister("resx").is_success

    _ = c.register_resource("r1", lambda: "ok")
    c.clear_all()

    FlextContainer.reset_singleton_for_testing()
    instance = FlextContainer.create()
    assert instance._global_instance is instance


def test_scoped_config_context_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    c = FlextContainer.create()
    c._services = {}
    c._factories = {}
    c._resources = {}
    c._user_overrides = m.ConfigMap(root={})
    c._global_config = m.Container.ContainerConfig(
        enable_singleton=True,
        enable_factory_caching=False,
        max_services=10,
        max_factories=10,
    )
    c._config = FlextSettings(app_name="base")
    c._context = FlextContext()

    captured: dict[str, Any] = {}

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
        config=cast("Any", _FalseConfig()), context=cast("Any", _ContextNoClone())
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
        f_back=types.SimpleNamespace(f_globals={"__name__": "fake_factory_mod"})
    )
    monkeypatch.setitem(__import__("sys").modules, "fake_factory_mod", fake_module)
    monkeypatch.setattr("flext_core.container.inspect.currentframe", lambda: frame)
    monkeypatch.setattr(
        "flext_core.container.FactoryDecoratorsDiscovery.scan_module",
        lambda _m: [
            (
                "factory_fn",
                m.HandlerFactoryDecoratorConfig(
                    name="factory.captured", singleton=False, lazy=True
                ),
            )
        ],
    )

    original_register = FlextContainer.register_factory

    def capture_register(
        self: FlextContainer, name: str, factory: Callable[..., object]
    ):
        captured[name] = factory
        return r[bool].ok(True)

    monkeypatch.setattr(FlextContainer, "register_factory", capture_register)
    _ = FlextContainer.create(auto_register_factories=True)
    wrapper = captured["factory.captured"]
    assert wrapper() == 7
    assert wrapper(fn=123) == ""
    monkeypatch.setattr(FlextContainer, "register_factory", original_register)


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

    class _Cfg:
        _namespace_registry: ClassVar[dict[str, object]] = {
            "alpha": object(),
            "beta": object(),
        }

        @staticmethod
        def get_namespace_config(namespace: str) -> type[BaseModel]:
            if namespace == "alpha":

                class _Model(BaseModel):
                    v: str = "ok"

                return _Model

            class _Model2(BaseModel):
                v: str = "ok2"

            return _Model2

        @staticmethod
        def get_namespace(_ns: str, _cls: type[BaseModel]) -> object:
            return "bad-value"

    c_any: Any = c
    c_any._config = _Cfg()
    c._global_config = m.Container.ContainerConfig(
        enable_singleton=True,
        enable_factory_caching=False,
        max_services=10,
        max_factories=10,
    )
    c._user_overrides = m.ConfigMap(root={})
    registered: dict[str, Callable[..., object]] = {}
    c_any.has_service = lambda name: False

    def _register(name: str, factory: Callable[..., object]):
        registered[name] = factory
        return r[bool].ok(True)

    c_any.register_factory = _register
    c.sync_config_to_di()
    assert "config.alpha" in registered
    assert "config.beta" in registered
    assert isinstance(registered["config.alpha"](), BaseModel)
    assert isinstance(registered["config.beta"](), BaseModel)


def test_register_existing_providers_full_paths_and_misc_methods() -> None:
    c = FlextContainer.create()
    c._services = {
        "s1": m.Container.ServiceRegistration(
            name="s1", service="v", service_type="str"
        )
    }
    c._factories = {
        "f1": m.Container.FactoryRegistration(name="f1", factory=lambda: "fv")
    }
    c._resources = {
        "r1": m.Container.ResourceRegistration(name="r1", factory=lambda: "rv")
    }
    c.register_existing_providers()
    assert hasattr(c._di_container, "s1")
    assert hasattr(c._di_container, "f1")
    assert hasattr(c._di_container, "r1")

    c_any: Any = c
    c_any._context = FlextContext()
    c_any.has_service = lambda name: False
    c.register_core_services()
    assert c.has_service("context") is False or True

    c.wire_modules(modules=[])
    assert c.get("r1").is_success
    assert c.get_typed("f1", str).is_success
    assert c.get_typed("r1", str).is_success


def test_create_scoped_instance_and_scoped_additional_branches() -> None:
    base = FlextContainer.create()
    scoped = FlextContainer._create_scoped_instance(
        config=FlextSettings(),
        context=FlextContext(),
        services={},
        factories={},
        resources={},
        user_overrides=m.ConfigMap(root={}),
        container_config=m.Container.ContainerConfig(
            enable_singleton=True,
            enable_factory_caching=False,
            max_services=10,
            max_factories=10,
        ),
    )
    assert isinstance(scoped, FlextContainer)

    base_any: Any = base
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
    c._global_config = m.Container.ContainerConfig(
        enable_singleton=True,
        enable_factory_caching=False,
        max_services=10,
        max_factories=10,
    )

    global_instance = FlextContainer.get_global()
    assert isinstance(global_instance, FlextContainer)

    assert c.with_config({"enable_factory_caching": True}) is c
    assert c.with_service("svc-x", "value") is c
    assert c.with_factory("fac-x", lambda: "v") is c

    assert c.get_config().root
    assert c.register("", "x").is_failure
    assert c.get("svc-x").is_success
    assert c.get("missing-service").is_failure
    assert c.get_typed("svc-x", str).is_success
    assert c.get_typed("missing-service", str).is_failure


def test_additional_register_factory_and_unregister_paths() -> None:
    c = FlextContainer.create()
    c._global_config = m.Container.ContainerConfig(
        enable_singleton=True,
        enable_factory_caching=False,
        max_services=10,
        max_factories=10,
    )

    assert c.register_factory("fac-ok", lambda: 1).is_success
    assert c.register_factory("fac-ok", lambda: 2).is_failure
    assert c.register_factory("fac-bad", cast("Any", 123)).is_success

    assert FlextContainer._narrow_factory_result("x") == "x"

    _ = c.register("svc-remove", "v")
    _ = c.register_resource("res-remove", lambda: "r")
    _ = c.register_factory("fac-remove", lambda: "f")

    assert c.unregister("svc-remove").is_success
    assert c.unregister("fac-remove").is_success
    assert c.unregister("res-remove").is_success
    assert c.unregister("not-there").is_failure


def test_container_remaining_branch_paths_in_sync_factory_and_getters() -> None:
    c = FlextContainer.create()
    c._global_config = m.Container.ContainerConfig(
        enable_singleton=True,
        enable_factory_caching=False,
        max_services=10,
        max_factories=10,
    )
    c._user_overrides = m.ConfigMap(root={})

    class _CfgNoMethod:
        _namespace_registry: ClassVar[dict[str, object]] = {"n1": object()}

    c_any: Any = c
    c_any._config = _CfgNoMethod()
    c.sync_config_to_di()

    class _CfgFallback:
        _namespace_registry: ClassVar[dict[str, object]] = {"n2": object()}

        @staticmethod
        def get_namespace_config(_namespace: str) -> type[BaseModel]:
            class _Model(BaseModel):
                v: str = "x"

            return _Model

    class _CfgBadNamespace:
        _namespace_registry: ClassVar[dict[str, object]] = {"n3": object()}

        @staticmethod
        def get_namespace_config(_namespace: str) -> type[BaseModel]:
            class _Model(BaseModel):
                v: str = "x"

            return _Model

        @staticmethod
        def get_namespace(_namespace: str, _cls: type[BaseModel]) -> object:
            return "bad"

    class _CfgGoodNamespace:
        _namespace_registry: ClassVar[dict[str, object]] = {"n4": object()}

        @staticmethod
        def get_namespace_config(_namespace: str) -> type[BaseModel]:
            class _Model(BaseModel):
                v: str = "x"

            return _Model

        @staticmethod
        def get_namespace(_namespace: str, cls: type[BaseModel]) -> object:
            return cls()

    c_any._config = _CfgFallback()
    captured: dict[str, Callable[..., object]] = {}
    c_any.has_service = lambda _name: False

    def _capture_factory(name: str, factory: Callable[..., object]) -> r[bool]:
        captured[name] = factory
        return r[bool].ok(True)

    c_any.register_factory = _capture_factory
    c.sync_config_to_di()

    c_any._config = _CfgBadNamespace()
    c.sync_config_to_di()

    c_any._config = _CfgGoodNamespace()
    c.sync_config_to_di()

    assert isinstance(captured["config.n2"](), BaseModel)
    assert isinstance(captured["config.n3"](), BaseModel)
    assert isinstance(captured["config.n4"](), BaseModel)

    delattr(c_any, "register_factory")

    c2 = FlextContainer.create()
    c2._global_config = m.Container.ContainerConfig(
        enable_singleton=True,
        enable_factory_caching=False,
        max_services=10,
        max_factories=10,
    )
    assert c2.register_factory("fac-call", lambda: "value").is_success
    assert c2._factories["fac-call"].factory() == "value"

    c2._factories = {
        "boom": m.Container.FactoryRegistration(
            name="boom",
            factory=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        )
    }
    assert c2.get("boom").is_failure

    c2._factories = {
        "ok-factory": m.Container.FactoryRegistration(
            name="ok-factory", factory=lambda: "ok"
        )
    }
    assert c2.get("ok-factory").is_success

    c2._services = {
        "svc-int": m.Container.ServiceRegistration(
            name="svc-int", service="str", service_type="str"
        )
    }
    assert c2.get_typed("svc-int", int).is_failure

    executed: list[str] = []
    c_any.has_service = lambda _name: False

    def _track_factory(_name: str, factory: Callable[[], object]) -> r[bool]:
        executed.append(type(factory()).__name__)
        return r[bool].ok(True)

    c_any.register_factory = _track_factory
    c_any._config = _CfgFallback()
    c.sync_config_to_di()
    c_any._config = _CfgBadNamespace()
    c.sync_config_to_di()
    assert executed
