"""Container full coverage tests."""

from __future__ import annotations

import sys
import types
from collections.abc import Callable, Mapping
from types import ModuleType
from typing import ClassVar, Self, cast

import pytest
from flext_tests import t, tm
from pydantic import BaseModel
from pydantic_settings import BaseSettings as _BaseSettings

from flext_core import FlextContainer, FlextContext, FlextSettings, p, r
from tests import m


class TestContainerFullCoverage:
    _MonkeyPatch = pytest.MonkeyPatch

    class _FalseConfig:
        app_name: str = "app"
        version: str = "1.0.0"
        enable_caching: bool = False
        timeout_seconds: float = 1.0
        dispatcher_auto_context: bool = False
        dispatcher_enable_logging: bool = False

        def model_copy(
            self,
            *,
            update: Mapping[str, t.Container] | None = None,
            deep: bool = False,
        ) -> Self:
            return self

        def model_dump(self) -> dict[str, t.Scalar]:
            return {}

    class _ContextNoClone:
        def clone(self) -> _ContextNoClone:
            return self

        def get(self, key: str, scope: str = "") -> r[t.Container | BaseModel]:
            return r[t.Container | BaseModel].fail("err")

        def set(
            self,
            key_or_data: str | t.ConfigMap,
            value: t.Container | BaseModel | None = None,
            *,
            scope: str = "",
        ) -> r[bool]:
            return r[bool].ok(True)

    class _BridgeNoProvide:
        pass

    class _BridgeBadProvide:
        provide: None = None

    class _BridgeGoodProvide:
        @staticmethod
        def provide(name: str) -> str:
            return name

    @staticmethod
    def _scan_factory_module(
        _module: ModuleType,
    ) -> list[tuple[str, m.FactoryDecoratorConfig]]:
        return [
            (
                "the_factory",
                m.FactoryDecoratorConfig(name="x", singleton=False, lazy=True),
            ),
        ]

    @staticmethod
    def _scan_factory_module_captured(
        _module: ModuleType,
    ) -> list[tuple[str, m.FactoryDecoratorConfig]]:
        return [
            (
                "factory_fn",
                m.FactoryDecoratorConfig(
                    name="factory.captured",
                    singleton=False,
                    lazy=True,
                ),
            ),
        ]

    @staticmethod
    def _has_service_false(_name: str) -> bool:
        return False

    @staticmethod
    def _raise_register_object(*_args: t.Scalar, **_kwargs: t.Scalar) -> None:
        msg = "boom"
        raise RuntimeError(msg)

    @staticmethod
    def _raise_register_factory(*_args: t.Scalar, **_kwargs: t.Scalar) -> None:
        msg = "boom"
        raise RuntimeError(msg)

    @staticmethod
    def _raise_register_resource(*_args: t.Scalar, **_kwargs: t.Scalar) -> None:
        msg = "boom"
        raise RuntimeError(msg)

    @staticmethod
    def _namespace_config_none(_namespace: str) -> None:
        return None

    def test_builder(self) -> None:
        tm.that(isinstance(FlextContainer.Builder.create(), p.Container), eq=True)

    def test_create_auto_register_factories_path(
        self, monkeypatch: _MonkeyPatch
    ) -> None:
        container = FlextContainer.create()
        called: list[str] = []

        def factory() -> int:
            return 1

        fake_module = types.SimpleNamespace(the_factory=factory)
        frame = types.SimpleNamespace(
            f_back=types.SimpleNamespace(f_globals={"__name__": "fake_mod"}),
        )
        monkeypatch.setitem(
            sys.modules, "fake_mod", cast("types.ModuleType", fake_module)
        )
        monkeypatch.setattr("flext_core.container.inspect.currentframe", lambda: frame)
        monkeypatch.setattr(
            "flext_core.container.FactoryDecoratorsDiscovery.scan_module",
            _scan_factory_module,
        )

        def _register(
            name: str, _impl: t.RegisterableService, *, kind: str = "service"
        ) -> FlextContainer:
            if kind == "factory":
                called.append(name)
            return container

        monkeypatch.setattr(container, "register", _register)

        def _call_container(*_args: t.Scalar, **_kwargs: t.Scalar) -> FlextContainer:
            return container

        monkeypatch.setattr(
            "flext_core.container.FlextContainer.__call__",
            _call_container,
            raising=False,
        )
        created = FlextContainer.create(auto_register_factories=True)
        tm.that(isinstance(created, p.Container), eq=True)
        tm.that(called, has="x")

    def test_provide_property_paths(self, monkeypatch: _MonkeyPatch) -> None:
        c = FlextContainer.create()
        monkeypatch.setattr(c, "_di_bridge", _BridgeGoodProvide())
        tm.that(c.provide("x"), eq="x")
        monkeypatch.setattr(c, "_di_bridge", _BridgeBadProvide())
        with pytest.raises(RuntimeError):
            _ = c.provide
        monkeypatch.setattr(c, "_di_bridge", _BridgeNoProvide())
        with pytest.raises(RuntimeError):
            _ = c.provide

    def test_config_context_properties_and_defaults(
        self,
        monkeypatch: _MonkeyPatch,
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
        tm.that(isinstance(c._get_default_config(), p.Settings), eq=True)

    def test_initialize_di_components_error_paths(
        self,
        monkeypatch: _MonkeyPatch,
    ) -> None:
        c = FlextContainer.create()
        bad_bridge = types.SimpleNamespace(config="not-provider")
        monkeypatch.setattr(
            "flext_core.container.FlextRuntime.DependencyIntegration.create_layered_bridge",
            lambda: (bad_bridge, types.SimpleNamespace(), types.SimpleNamespace()),
        )
        with pytest.raises(TypeError, match="Bridge must have config provider"):
            c.initialize_di_components()

    def test_sync_config_namespace_paths(self, monkeypatch: _MonkeyPatch) -> None:
        c = FlextContainer.create()
        c._config = FlextSettings()
        c._user_overrides = t.ConfigMap(root={})
        c._global_config = m.ContainerConfig(
            enable_singleton=True,
            enable_factory_caching=False,
            max_services=10,
            max_factories=10,
        )
        monkeypatch.setattr(
            type(c._config), "_namespace_registry", {"x": _BaseSettings}
        )
        monkeypatch.setattr(c, "has_service", _has_service_false)

        def _capture_register(
            _name: str,
            _impl: t.RegisterableService,
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
        self,
        monkeypatch: _MonkeyPatch,
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
        c._config = _FalseConfig()
        c._context = None
        monkeypatch.setattr(c, "has_service", _has_service_false)
        c.register_core_services()

    def test_configure_with_resource_register_and_factory_error_paths(
        self,
        monkeypatch: _MonkeyPatch,
    ) -> None:
        c = FlextContainer.create()
        c._user_overrides = t.ConfigMap(root={})
        c._global_config = m.ContainerConfig(
            enable_singleton=True,
            enable_factory_caching=False,
            max_services=100,
            max_factories=100,
        )
        c.configure({"enable_factory_caching": True})
        tm.that(c._global_config.enable_factory_caching, eq=True)
        c.register("res", lambda: "x", kind="resource")
        tm.that(c._resources, has="res")
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
        del c._di_resources.dup
        monkeypatch.setattr(
            "flext_core.container.FlextRuntime.DependencyIntegration.register_resource",
            _raise_register_resource,
        )
        c.register("new", lambda: "v", kind="resource")

    def test_get_and_get_typed_resource_factory_paths(self) -> None:
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
        tm.fail(c.get("r"))
        c._factories = {"f2": m.FactoryRegistration(name="f2", factory=lambda: "abc")}
        tm.fail(c.get("f2", type_cls=int))
        c._resources = {"r2": m.ResourceRegistration(name="r2", factory=lambda: "abc")}
        tm.fail(c.get("r2", type_cls=int))
        c._factories = {
            "f3": m.FactoryRegistration(
                name="f3",
                factory=lambda: (_ for _ in ()).throw(ValueError("err")),
            ),
        }
        tm.fail(c.get("f3", type_cls=str))
        c._resources = {
            "r3": m.ResourceRegistration(
                name="r3",
                factory=lambda: (_ for _ in ()).throw(ValueError("err")),
            ),
        }
        tm.fail(c.get("r3", type_cls=str))

    def test_misc_unregistration_clear_and_reset(self) -> None:
        c = FlextContainer.create()
        _ = c.register("resx", lambda: "ok", kind="resource")
        tm.that(c.create_module_logger(), ne=None)
        tm.ok(c.unregister("resx"))
        _ = c.register("r1", lambda: "ok", kind="resource")
        c.clear_all()
        FlextContainer.reset_for_testing()
        instance = FlextContainer.create()
        tm.that(isinstance(instance._global_instance, type(instance)), eq=True)

    def test_scoped_config_context_branches(self, monkeypatch: _MonkeyPatch) -> None:
        c = FlextContainer.create()
        c._services = {}
        c._factories = {}
        c._resources = {}
        c._user_overrides = t.ConfigMap(root={})
        c._global_config = m.ContainerConfig(
            enable_singleton=True,
            enable_factory_caching=False,
            max_services=10,
            max_factories=10,
        )
        c._config = FlextSettings(app_name="base")
        c._context = FlextContext()
        captured: dict[str, object] = {}

        def _fake_create_scoped_instance(
            *, registration: m.ServiceRegistrationSpec, **kwargs: object
        ) -> FlextContainer:
            captured["config"] = registration.config
            captured["context"] = registration.context
            return c

        monkeypatch.setattr(
            "flext_core.container.FlextContainer._create_scoped_instance",
            _fake_create_scoped_instance,
        )
        _ = c.scoped(subproject="sub")
        tm.that(isinstance(captured["config"], p.Settings), eq=True)
        _ = c.scoped(
            config=_FalseConfig(),
            context=FlextContext(),
        )
        tm.that(isinstance(captured["context"], p.Context), eq=True)

    def test_create_auto_register_factory_wrapper_callable_and_non_callable(
        self,
        monkeypatch: _MonkeyPatch,
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
            impl: t.RegisterableService,
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
        tm.that(wrapper(), eq=7)
        tm.that(
            wrapper(
                _factory_config=types.SimpleNamespace(fn=123, name="factory.captured")
            ),
            eq=t.ConfigMap(root={}),
        )
        monkeypatch.setattr(FlextContainer, "register", original_register)

    def test_initialize_di_components_second_type_error_branch(
        self,
        monkeypatch: _MonkeyPatch,
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

    def test_sync_config_registers_namespace_factories_and_fallbacks(
        self,
        monkeypatch: _MonkeyPatch,
    ) -> None:
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

            class _Cfg(_FalseConfig):
                _namespace_registry: ClassVar[dict[str, type[_BaseSettings]]] = {
                    "alpha": _NsAlpha,
                    "beta": _NsBeta,
                }

            c._config = _Cfg()
            c._global_config = m.ContainerConfig(
                enable_singleton=True,
                enable_factory_caching=False,
                max_services=10,
                max_factories=10,
            )
            c._user_overrides = t.ConfigMap(root={})
            registered: dict[str, Callable[..., object]] = {}
            monkeypatch.setattr(c, "has_service", _has_service_false)

            def _register(
                name: str,
                impl: t.RegisterableService,
                *,
                kind: str = "service",
            ) -> FlextContainer:
                if kind == "factory" and callable(impl):
                    registered[name] = impl
                return c

            monkeypatch.setattr(c, "register", _register)
            c.sync_config_to_di()
            tm.that(isinstance(registered, dict), eq=True)
            if "config.alpha" in registered:
                tm.that(isinstance(registered["config.alpha"](), BaseModel), eq=True)
            if "config.beta" in registered:
                tm.that(isinstance(registered["config.beta"](), BaseModel), eq=True)
        finally:
            FlextSettings._namespace_registry.clear()
            FlextSettings._namespace_registry.update(original_registry)

    def test_register_existing_providers_full_paths_and_misc_methods(
        self,
        monkeypatch: _MonkeyPatch,
    ) -> None:
        c = FlextContainer.create()
        c._services = {
            "s1": m.ServiceRegistration(name="s1", service="v", service_type="str"),
        }
        c._factories = {"f1": m.FactoryRegistration(name="f1", factory=lambda: "fv")}
        c._resources = {"r1": m.ResourceRegistration(name="r1", factory=lambda: "rv")}
        c.register_existing_providers()
        tm.that(hasattr(c._di_container, "s1"), eq=True)
        tm.that(hasattr(c._di_container, "f1"), eq=True)
        tm.that(hasattr(c._di_container, "r1"), eq=True)
        monkeypatch.setattr(c, "_context", FlextContext())
        monkeypatch.setattr(c, "has_service", _has_service_false)
        c.register_core_services()
        tm.that(c.has_service("context"), is_=bool)
        c.wire_modules(modules=[])
        tm.ok(c.get("r1"))
        tm.ok(c.get("f1", type_cls=str))
        tm.ok(c.get("r1", type_cls=str))

    def test_create_scoped_instance_and_scoped_additional_branches(self) -> None:
        base = FlextContainer.create()
        scoped = FlextContainer._create_scoped_instance(
            registration=m.ServiceRegistrationSpec(
                config=FlextSettings(),
                context=FlextContext(),
                services={},
                factories={},
                resources={},
                user_overrides=t.ConfigMap(root={}),
                container_config=m.ContainerConfig(
                    enable_singleton=True,
                    enable_factory_caching=False,
                    max_services=10,
                    max_factories=10,
                ),
            ),
        )
        tm.that(isinstance(scoped, p.Container), eq=True)
        base._config = _FalseConfig()
        base._context = FlextContext()
        _ = base.scoped(
            config=FlextSettings(app_name="x"),
            services={"sx": "vx"},
            factories={"fx": lambda: "fv"},
            resources={"rx": lambda: "rv"},
        )
        base._config = _FalseConfig()
        base._context = _ContextNoClone()
        _ = base.scoped()

    def test_additional_container_branches_cover_fluent_and_lookup_paths(self) -> None:
        c = FlextContainer.create()
        c._global_config = m.ContainerConfig(
            enable_singleton=True,
            enable_factory_caching=False,
            max_services=10,
            max_factories=10,
        )
        global_instance = FlextContainer.get_global()
        tm.that(isinstance(global_instance, p.Container), eq=True)
        tm.that(c.configure({"enable_factory_caching": True}), eq=c)
        c.register("svc-x", "value")
        c.register("fac-x", lambda: "v", kind="factory")
        tm.that(c.get_config().root, ne=None)
        c.register("", "x")
        tm.ok(c.get("svc-x"))
        tm.fail(c.get("missing-service"))
        tm.ok(c.get("svc-x", type_cls=str))
        tm.fail(c.get("missing-service", type_cls=str))

    def test_additional_register_factory_and_unregister_paths(self) -> None:
        c = FlextContainer.create()
        c._global_config = m.ContainerConfig(
            enable_singleton=True,
            enable_factory_caching=False,
            max_services=10,
            max_factories=10,
        )
        c.register("fac-ok", lambda: 1, kind="factory")
        c.register("fac-ok", lambda: 2, kind="factory")
        c.register("fac-bad", 123, kind="factory")
        _ = c.register("svc-remove", "v")
        _ = c.register("res-remove", lambda: "r", kind="resource")
        _ = c.register("fac-remove", lambda: "f", kind="factory")
        tm.ok(c.unregister("svc-remove"))
        tm.ok(c.unregister("fac-remove"))
        tm.ok(c.unregister("res-remove"))
        tm.fail(c.unregister("not-there"))

    def test_container_remaining_branch_paths_in_sync_factory_and_getters(
        self,
        monkeypatch: _MonkeyPatch,
    ) -> None:
        c = FlextContainer.create()
        c._global_config = m.ContainerConfig(
            enable_singleton=True,
            enable_factory_caching=False,
            max_services=10,
            max_factories=10,
        )
        c._user_overrides = t.ConfigMap(root={})

        # --- _CfgNoMethod: namespace_registry without get_namespace_config ---
        # n1 is NOT registered in FlextSettings, so get_namespace_config returns None
        # and sync_config_to_di skips it (continue branch).
        class _CfgNoMethod(_FalseConfig):
            _namespace_registry: ClassVar[dict[str, type[_BaseSettings]]] = {
                "n1": _BaseSettings
            }

        c._config = _CfgNoMethod()
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

            class _CfgFallback(_FalseConfig):
                _namespace_registry: ClassVar[dict[str, type[_BaseSettings]]] = {
                    "n2": _NsModel
                }

            class _CfgBadNamespace(_FalseConfig):
                _namespace_registry: ClassVar[dict[str, type[_BaseSettings]]] = {
                    "n3": _NsModel
                }

            class _CfgGoodNamespace(_FalseConfig):
                _namespace_registry: ClassVar[dict[str, type[_BaseSettings]]] = {
                    "n4": _NsModel
                }

            c._config = _CfgFallback()
            captured: dict[str, Callable[..., object]] = {}
            monkeypatch.setattr(c, "has_service", _has_service_false)

            def _capture_register(
                name: str,
                impl: t.RegisterableService,
                *,
                kind: str = "service",
            ) -> FlextContainer:
                if kind == "factory" and callable(impl):
                    captured[name] = impl
                return c

            monkeypatch.setattr(c, "register", _capture_register)
            c.sync_config_to_di()
            c._config = _CfgBadNamespace()
            c.sync_config_to_di()
            c._config = _CfgGoodNamespace()
            c.sync_config_to_di()
            tm.that(isinstance(captured["config.n2"](), BaseModel), eq=True)
            tm.that(isinstance(captured["config.n3"](), BaseModel), eq=True)
            tm.that(isinstance(captured["config.n4"](), BaseModel), eq=True)
            c2 = FlextContainer.create()
            c2._global_config = m.ContainerConfig(
                enable_singleton=True,
                enable_factory_caching=False,
                max_services=10,
                max_factories=10,
            )
            c2._factories = {
                "fac-call": m.FactoryRegistration(
                    name="fac-call", factory=lambda: "value"
                )
            }
            fac_call = c2.get("fac-call")
            tm.ok(fac_call)
            tm.that(fac_call.value, eq="value")
            c2._factories = {
                "boom": m.FactoryRegistration(
                    name="boom",
                    factory=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
                ),
            }
            tm.fail(c2.get("boom"))
            c2._factories = {
                "ok-factory": m.FactoryRegistration(
                    name="ok-factory", factory=lambda: "ok"
                ),
            }
            tm.ok(c2.get("ok-factory"))
            c2._services = {
                "svc-int": m.ServiceRegistration(
                    name="svc-int",
                    service="str",
                    service_type="str",
                ),
            }
            tm.fail(c2.get("svc-int", type_cls=int))
            executed: list[str] = []
            monkeypatch.setattr(c, "has_service", _has_service_false)

            def _track_register(
                _name: str,
                impl: t.RegisterableService,
                *,
                kind: str = "service",
            ) -> FlextContainer:
                if kind == "factory" and callable(impl):
                    executed.append(type(impl()).__name__)
                return c

            monkeypatch.setattr(c, "register", _track_register)
            c._config = _CfgFallback()
            c.sync_config_to_di()
            c._config = _CfgBadNamespace()
            c.sync_config_to_di()
            tm.that(executed, ne=[])
        finally:
            # Restore original registry
            FlextSettings._namespace_registry.clear()
            FlextSettings._namespace_registry.update(original_registry)


_MonkeyPatch = TestContainerFullCoverage._MonkeyPatch
_FalseConfig = TestContainerFullCoverage._FalseConfig
_ContextNoClone = TestContainerFullCoverage._ContextNoClone
_BridgeNoProvide = TestContainerFullCoverage._BridgeNoProvide
_BridgeBadProvide = TestContainerFullCoverage._BridgeBadProvide
_BridgeGoodProvide = TestContainerFullCoverage._BridgeGoodProvide
_scan_factory_module = TestContainerFullCoverage._scan_factory_module
_scan_factory_module_captured = TestContainerFullCoverage._scan_factory_module_captured
_has_service_false = TestContainerFullCoverage._has_service_false
_raise_register_object = TestContainerFullCoverage._raise_register_object
_raise_register_factory = TestContainerFullCoverage._raise_register_factory
_raise_register_resource = TestContainerFullCoverage._raise_register_resource
_namespace_config_none = TestContainerFullCoverage._namespace_config_none
