"""Container full coverage tests."""

from __future__ import annotations

import types
from collections.abc import Callable, Mapping, MutableMapping, MutableSequence, Sequence
from types import ModuleType
from typing import ClassVar, cast

import pytest
from dependency_injector import containers as di_containers
from pydantic import BaseModel
from pydantic_settings import BaseSettings as _BaseSettings

import flext_core as _discovery_mod
import flext_core as core_container
from flext_core import FlextContainer, FlextContext, FlextSettings
from flext_tests import tm
from tests import m, p, r, t


class TestContainerFullCoverage:
    class TypedValue(BaseModel):
        value: str = ""

    @staticmethod
    def _typed_value_cls() -> type[t.RegisterableService]:
        return cast(
            "type[t.RegisterableService]",
            TestContainerFullCoverage.TypedValue,
        )

    @staticmethod
    def _get_with_type(
        container: p.Container,
        name: str,
        type_cls: type[t.RegisterableService],
    ) -> r[t.RegisterableService]:
        typed_get = cast("Callable[..., r[t.RegisterableService]]", container.get)
        return typed_get(name, type_cls=type_cls)

    @staticmethod
    def _scan_factory_module(
        _module: ModuleType,
    ) -> Sequence[tuple[str, m.FactoryDecoratorConfig]]:
        return [
            (
                "the_factory",
                m.FactoryDecoratorConfig(name="x", singleton=False, lazy=True),
            ),
        ]

    @staticmethod
    def _scan_factory_module_captured(
        _module: ModuleType,
    ) -> Sequence[tuple[str, m.FactoryDecoratorConfig]]:
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

    def test_create(self) -> None:
        tm.that(FlextContainer.create(), is_=p.Container)

    def test_create_auto_register_factories_path(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        called: MutableSequence[str] = []
        FlextContainer.reset_for_testing()

        def factory() -> int:
            return 1

        class FactoryModule(ModuleType):
            the_factory: Callable[[], int]

        def _resolve_fake_module(_frame: types.FrameType | None) -> ModuleType:
            return fake_module

        fake_module = FactoryModule("fake_mod")
        fake_module.the_factory = factory
        monkeypatch.setattr(
            _discovery_mod.FlextUtilitiesDiscovery,
            "scan_module",
            self._scan_factory_module,
        )
        monkeypatch.setattr(
            FlextContainer,
            "_resolve_caller_module",
            staticmethod(_resolve_fake_module),
        )

        def _register(
            self: p.Container,
            name: str,
            _impl: t.RegisterableService,
            *,
            kind: str = "service",
        ) -> p.Container:
            if kind == "factory":
                called.append(name)
            return self

        monkeypatch.setattr(
            FlextContainer,
            "register",
            _register,
        )
        created = FlextContainer.create(auto_register_factories=True)
        tm.that(created, is_=p.Container)
        tm.that(called, has="x")

    def test_provide_property_paths(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _ = monkeypatch
        c = FlextContainer.create()

        class ProvideBridge:
            @staticmethod
            def provide(name: str) -> str:
                return name

        class NoProvideBridge:
            provide: ClassVar[None] = None

        c._di_bridge = cast(
            "di_containers.DeclarativeContainer",
            cast("t.RecursiveContainer", ProvideBridge()),
        )
        result = c.provide("x")
        assert result == "x"
        c._di_bridge = cast(
            "di_containers.DeclarativeContainer",
            cast("t.RecursiveContainer", NoProvideBridge()),
        )
        with pytest.raises(RuntimeError, match="Provide helper not initialized"):
            _ = c.provide

    def test_config_context_properties_and_defaults(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        c = FlextContainer.create()
        c._config = None
        with pytest.raises(RuntimeError):
            _ = c.settings
        c._context = None
        with pytest.raises(RuntimeError):
            _ = c.context
        monkeypatch.setattr(
            core_container.FlextSettings,
            "fetch_global",
            lambda: FlextSettings(),
        )
        tm.that(c._get_default_config(), is_=p.Settings)

    def test_initialize_di_components_error_paths(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        c = FlextContainer.create()
        bad_bridge = types.SimpleNamespace(settings="not-provider")
        monkeypatch.setattr(
            core_container.u.DependencyIntegration,
            "create_layered_bridge",
            lambda: (bad_bridge, types.SimpleNamespace(), types.SimpleNamespace()),
        )
        with pytest.raises(TypeError, match="Bridge must have settings provider"):
            c.initialize_di_components()

    def test_sync_config_namespace_paths(self, monkeypatch: pytest.MonkeyPatch) -> None:
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
            type(c._config),
            "_namespace_registry",
            {"x": cast("type[p.Settings]", _BaseSettings)},
        )

        monkeypatch.setattr(
            type(c._config),
            "resolve_namespace_settings",
            staticmethod(self._namespace_config_none),
        )
        c.sync_config_to_di()

    def test_register_existing_providers_skips_and_register_core_fallback(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        c = FlextContainer.create()
        c._services = {
            "svc": m.ServiceRegistration(name="svc", service="v", service_type="str"),
        }
        c._factories = {"fac": m.FactoryRegistration(name="fac", factory=lambda: "v")}
        c._resources = {"res": m.ResourceRegistration(name="res", factory=lambda: "v")}
        setattr(c._di_services, "svc", "normalized")
        setattr(c._di_services, "fac", "normalized")
        setattr(c._di_resources, "res", "normalized")
        c.register_existing_providers()
        c._config = cast("p.Settings", m.Core.Tests.FalseSettings())
        c._context = None
        c.register_core_services()

    def test_configure_with_resource_register_and_factory_error_paths(
        self,
        monkeypatch: pytest.MonkeyPatch,
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
            core_container.u.DependencyIntegration,
            "register_object",
            self._raise_register_object,
        )
        c.register("x", "y")
        monkeypatch.setattr(
            core_container.u.DependencyIntegration,
            "register_factory",
            self._raise_register_factory,
        )
        c.register("x2", lambda: "v", kind="factory")
        setattr(c._di_resources, "dup", "normalized")
        c.register("dup", lambda: "v", kind="resource")
        del c._di_resources.dup
        monkeypatch.setattr(
            core_container.u.DependencyIntegration,
            "register_resource",
            self._raise_register_resource,
        )
        c.register("new", lambda: "v", kind="resource")

    def test_get_and_get_typed_resource_factory_paths(self) -> None:
        c = FlextContainer.create()
        typed_value_cls = self._typed_value_cls()
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
        tm.fail(self._get_with_type(c, "f2", typed_value_cls))
        c._resources = {"r2": m.ResourceRegistration(name="r2", factory=lambda: "abc")}
        tm.fail(self._get_with_type(c, "r2", typed_value_cls))
        c._factories = {
            "f3": m.FactoryRegistration(
                name="f3",
                factory=lambda: (_ for _ in ()).throw(ValueError("err")),
            ),
        }
        tm.fail(self._get_with_type(c, "f3", typed_value_cls))
        c._resources = {
            "r3": m.ResourceRegistration(
                name="r3",
                factory=lambda: (_ for _ in ()).throw(ValueError("err")),
            ),
        }
        tm.fail(self._get_with_type(c, "r3", typed_value_cls))

    def test_misc_unregistration_clear_and_reset(self) -> None:
        c = FlextContainer.create()
        _ = c.register("resx", lambda: "ok", kind="resource")
        tm.that(c.create_module_logger(), ne=None)
        tm.ok(c.unregister("resx"))
        _ = c.register("r1", lambda: "ok", kind="resource")
        c.clear_all()
        FlextContainer.reset_for_testing()
        instance = FlextContainer.create()
        tm.that(instance._global_instance, is_=type(instance))

    def test_scoped_config_context_branches(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
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
        captured: MutableMapping[str, p.Settings | p.Context | None] = {}

        def _fake_create_scoped_instance(
            *,
            registration: m.ServiceRegistrationSpec,
            **kwargs: t.RecursiveContainer,
        ) -> p.Container:
            captured["settings"] = registration.settings
            captured["context"] = registration.context
            return c

        monkeypatch.setattr(
            FlextContainer, "_create_scoped_instance", _fake_create_scoped_instance
        )
        _ = c.scoped(subproject="sub")
        tm.that(captured["settings"], is_=p.Settings)
        _ = c.scoped(
            settings=cast("p.Settings", m.Core.Tests.FalseSettings()),
            context=FlextContext(),
        )
        tm.that(captured["context"], is_=p.Context)

    def test_create_auto_register_factory_wrapper_callable_and_non_callable(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: MutableMapping[str, t.RegisterableService] = {}

        def factory_fn() -> int:
            return 7

        class FactoryModule(ModuleType):
            factory_fn: Callable[[], int]

        def _resolve_fake_module(_frame: types.FrameType | None) -> ModuleType:
            return fake_module

        fake_module = FactoryModule("fake_factory_mod")
        fake_module.factory_fn = factory_fn
        monkeypatch.setattr(
            _discovery_mod.FlextUtilitiesDiscovery,
            "scan_module",
            self._scan_factory_module_captured,
        )
        monkeypatch.setattr(
            FlextContainer,
            "_resolve_caller_module",
            staticmethod(_resolve_fake_module),
        )

        def capture_register(
            self: p.Container,
            name: str,
            impl: t.RegisterableService,
            *,
            kind: str = "service",
            singleton: bool = False,
            lazy: bool = True,
        ) -> p.Container:
            _ = singleton
            _ = lazy
            if kind == "factory" and callable(impl):
                captured[name] = impl
            return self

        monkeypatch.setattr(FlextContainer, "register", capture_register)
        _ = FlextContainer.create(auto_register_factories=True)
        wrapper = cast(
            "Callable[..., t.RecursiveContainer]", captured["factory.captured"]
        )
        assert wrapper() == 7
        assert wrapper(
            _factory_config=types.SimpleNamespace(fn=123, name="factory.captured"),
        ) == t.ConfigMap(root={})

    def test_initialize_di_components_exposes_valid_runtime_providers(self) -> None:
        c = FlextContainer.create()
        c.initialize_di_components()
        assert c._config_provider is not None
        assert c._base_config_provider is not None
        assert c._user_config_provider is not None

    def test_sync_config_registers_namespace_factories_and_fallbacks(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        c = FlextContainer.create()

        class _NsAlpha(_BaseSettings):
            v: str = "ok"

        class _NsBeta(_BaseSettings):
            v: str = "ok2"

        # Register namespaces so FlextSettings.resolve_namespace_settings() finds them.
        original_registry = dict(FlextSettings._namespace_registry)
        FlextSettings._namespace_registry["alpha"] = cast("type[p.Settings]", _NsAlpha)
        FlextSettings._namespace_registry["beta"] = cast("type[p.Settings]", _NsBeta)

        try:

            class _Cfg(m.Core.Tests.FalseSettings):
                _namespace_registry: ClassVar[Mapping[str, type[p.Settings]]] = {
                    "alpha": cast("type[p.Settings]", _NsAlpha),
                    "beta": cast("type[p.Settings]", _NsBeta),
                }

            c._config = cast("p.Settings", _Cfg())
            c._global_config = m.ContainerConfig(
                enable_singleton=True,
                enable_factory_caching=False,
                max_services=10,
                max_factories=10,
            )
            c._user_overrides = t.ConfigMap(root={})
            registered: MutableMapping[str, t.RegisterableService] = {}

            def _register(
                name: str,
                impl: t.RegisterableService,
                *,
                kind: str = "service",
            ) -> p.Container:
                if kind == "factory" and callable(impl):
                    registered[name] = impl
                return c

            monkeypatch.setattr(c, "register", _register)
            c.sync_config_to_di()
            alpha_factory = registered["settings.alpha"]
            assert callable(alpha_factory) and isinstance(
                alpha_factory(),
                BaseModel,
            )
            beta_factory = registered["settings.beta"]
            assert callable(beta_factory) and isinstance(beta_factory(), BaseModel)
        finally:
            FlextSettings._namespace_registry.clear()
            FlextSettings._namespace_registry.update(original_registry)

    def test_register_existing_providers_full_paths_and_misc_methods(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        typed_value_cls = self._typed_value_cls()
        c = FlextContainer.create()
        c._services = {
            "s1": m.ServiceRegistration(name="s1", service="v", service_type="str"),
        }
        c._factories = {
            "f1": m.FactoryRegistration(
                name="f1",
                factory=lambda: self.TypedValue(value="fv"),
            )
        }
        c._resources = {
            "r1": m.ResourceRegistration(
                name="r1",
                factory=lambda: self.TypedValue(value="rv"),
            )
        }
        c.register_existing_providers()
        c.register_core_services()
        tm.that(c.has_service("context"), is_=bool)
        c.wire_modules(modules=[])
        assert c.get("r1").success
        tm.ok(self._get_with_type(c, "f1", typed_value_cls))
        tm.ok(self._get_with_type(c, "r1", typed_value_cls))

    def test_create_scoped_instance_and_scoped_additional_branches(self) -> None:
        base = FlextContainer.create()
        scoped = FlextContainer._create_scoped_instance(
            registration=m.ServiceRegistrationSpec(
                settings=FlextSettings(),
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
        tm.that(scoped, is_=p.Container)
        base._config = cast("p.Settings", m.Core.Tests.FalseSettings())
        base._context = FlextContext()
        _ = base.scoped(
            settings=FlextSettings(app_name="x"),
            services={"sx": "vx"},
            factories={"fx": lambda: "fv"},
            resources={"rx": lambda: "rv"},
        )
        base._config = cast("p.Settings", m.Core.Tests.FalseSettings())
        base._context = FlextContext()
        _ = base.scoped()

    def test_additional_container_branches_cover_fluent_and_lookup_paths(self) -> None:
        typed_value_cls = self._typed_value_cls()
        c = FlextContainer.create()
        c._global_config = m.ContainerConfig(
            enable_singleton=True,
            enable_factory_caching=False,
            max_services=10,
            max_factories=10,
        )
        global_instance = FlextContainer.fetch_global()
        tm.that(global_instance, is_=p.Container)
        tm.that(c.configure({"enable_factory_caching": True}), eq=c)
        c.register("svc-x", self.TypedValue(value="value"))
        c.register("fac-x", lambda: "v", kind="factory")
        tm.that(c.resolve_settings().root, ne=None)
        c.register("", "x")
        assert c.get("svc-x").success
        tm.fail(c.get("missing-service"))
        tm.ok(self._get_with_type(c, "svc-x", typed_value_cls))
        tm.fail(self._get_with_type(c, "missing-service", typed_value_cls))

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
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        c = FlextContainer.create()
        c._global_config = m.ContainerConfig(
            enable_singleton=True,
            enable_factory_caching=False,
            max_services=10,
            max_factories=10,
        )
        c._user_overrides = t.ConfigMap(root={})

        # --- _CfgNoMethod: namespace_registry without resolve_namespace_settings ---
        # n1 is NOT registered in FlextSettings, so resolve_namespace_settings returns None
        # and sync_config_to_di skips it (continue branch).
        class _CfgNoMethod(m.Core.Tests.FalseSettings):
            _namespace_registry: ClassVar[Mapping[str, type[p.Settings]]] = {
                "n1": cast("type[p.Settings]", _BaseSettings),
            }

        c._config = cast("p.Settings", _CfgNoMethod())
        c.sync_config_to_di()

        # --- Create real BaseSettings subclasses for n2, n3, n4 ---
        class _NsModel(_BaseSettings):
            v: str = "x"

        # Register namespaces in FlextSettings._namespace_registry so that
        # sync_config_to_di -> FlextSettings.resolve_namespace_settings() finds them.
        original_registry = dict(FlextSettings._namespace_registry)
        FlextSettings._namespace_registry["n2"] = cast("type[p.Settings]", _NsModel)
        FlextSettings._namespace_registry["n3"] = cast("type[p.Settings]", _NsModel)
        FlextSettings._namespace_registry["n4"] = cast("type[p.Settings]", _NsModel)

        try:

            class _CfgFallback(m.Core.Tests.FalseSettings):
                _namespace_registry: ClassVar[Mapping[str, type[p.Settings]]] = {
                    "n2": cast("type[p.Settings]", _NsModel),
                }

            class _CfgBadNamespace(m.Core.Tests.FalseSettings):
                _namespace_registry: ClassVar[Mapping[str, type[p.Settings]]] = {
                    "n3": cast("type[p.Settings]", _NsModel),
                }

            class _CfgGoodNamespace(m.Core.Tests.FalseSettings):
                _namespace_registry: ClassVar[Mapping[str, type[p.Settings]]] = {
                    "n4": cast("type[p.Settings]", _NsModel),
                }

            c._config = cast("p.Settings", _CfgFallback())
            captured: MutableMapping[str, t.RegisterableService] = {}

            def _capture_register(
                name: str,
                impl: t.RegisterableService,
                *,
                kind: str = "service",
            ) -> p.Container:
                if kind == "factory" and callable(impl):
                    captured[name] = impl
                return c

            monkeypatch.setattr(c, "register", _capture_register)
            c.sync_config_to_di()
            c._config = cast("p.Settings", _CfgBadNamespace())
            c.sync_config_to_di()
            c._config = cast("p.Settings", _CfgGoodNamespace())
            c.sync_config_to_di()
            assert isinstance(
                cast("Callable[[], BaseModel]", captured["settings.n2"])(), BaseModel
            )
            assert isinstance(
                cast("Callable[[], BaseModel]", captured["settings.n3"])(), BaseModel
            )
            assert isinstance(
                cast("Callable[[], BaseModel]", captured["settings.n4"])(), BaseModel
            )
            c2 = FlextContainer.create()
            c2._global_config = m.ContainerConfig(
                enable_singleton=True,
                enable_factory_caching=False,
                max_services=10,
                max_factories=10,
            )
            c2._factories = {
                "fac-call": m.FactoryRegistration(
                    name="fac-call",
                    factory=lambda: "value",
                ),
            }
            fac_call = c2.get("fac-call", type_cls=str)
            tm.ok(fac_call)
            assert fac_call.value == "value"
            c2._factories = {
                "boom": m.FactoryRegistration(
                    name="boom",
                    factory=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
                ),
            }
            tm.fail(c2.get("boom"))
            c2._factories = {
                "ok-factory": m.FactoryRegistration(
                    name="ok-factory",
                    factory=lambda: "ok",
                ),
            }
            assert c2.get("ok-factory").success
            c2._services = {
                "svc-int": m.ServiceRegistration(
                    name="svc-int",
                    service="str",
                    service_type="str",
                ),
            }
            tm.fail(self._get_with_type(c2, "svc-int", self._typed_value_cls()))

            c._config = cast("p.Settings", _CfgFallback())
            c.sync_config_to_di()
            c._config = cast("p.Settings", _CfgBadNamespace())
            c.sync_config_to_di()
        finally:
            # Restore original registry
            FlextSettings._namespace_registry.clear()
            FlextSettings._namespace_registry.update(original_registry)
