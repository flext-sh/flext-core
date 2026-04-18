"""Container full coverage tests."""

from __future__ import annotations

import types
from collections.abc import Callable, Mapping, MutableMapping, MutableSequence, Sequence
from types import ModuleType
from typing import ClassVar

import pytest

import flext_core as _discovery_mod
import flext_core as core_container
from flext_core import FlextContainer, FlextContext, FlextSettings
from flext_tests import tm
from tests import m, p, t, u


class TestContainerFullCoverage:
    class TypedValue(m.BaseModel):
        value: str = ""

    @staticmethod
    def _typed_value_cls() -> type[t.RegisterableService]:
        cls: type[t.RegisterableService] = TestContainerFullCoverage.TypedValue
        return cls

    @staticmethod
    def _get_with_type(
        container: p.Container,
        name: str,
        type_cls: type[t.RegisterableService],
    ) -> p.Result[t.RegisterableService]:
        typed_get: Callable[..., p.Result[t.RegisterableService]] = container.resolve
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
        tm.that(FlextContainer.shared(), is_=p.Container)

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

        def _factory(
            self: p.Container,
            name: str,
            _impl: t.RegisterableService,
        ) -> p.Container:
            called.append(name)
            return self

        monkeypatch.setattr(
            FlextContainer,
            "factory",
            _factory,
        )
        created = FlextContainer.shared(auto_register_factories=True)
        tm.that(created, is_=p.Container)
        tm.that(called, has="x")

    def test_provide_property_paths(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _ = monkeypatch
        c = FlextContainer.shared()

        class ProvideBridge:
            @staticmethod
            def provide(name: str) -> str:
                return name

        class NoProvideBridge:
            provide: ClassVar[None] = None

        setattr(c, "_di_bridge", ProvideBridge())
        result = c.provide("x")
        assert result == "x"
        setattr(c, "_di_bridge", NoProvideBridge())
        with pytest.raises(TypeError, match="Provide helper not initialized"):
            _ = c.provide

    def test_config_context_properties_and_defaults(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        c = FlextContainer.shared()
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
        tm.that(core_container.FlextSettings.fetch_global(), is_=p.Settings)

    def test_initialize_di_components_error_paths(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        c = FlextContainer.shared()
        bad_bridge = types.SimpleNamespace(settings="not-provider")
        monkeypatch.setattr(
            core_container.u.DependencyIntegration,
            "create_layered_bridge",
            lambda: (bad_bridge, types.SimpleNamespace(), types.SimpleNamespace()),
        )
        with pytest.raises(TypeError, match="Bridge must have settings provider"):
            c.initialize_di_components()

    def test_sync_config_namespace_paths(self, monkeypatch: pytest.MonkeyPatch) -> None:
        c = FlextContainer.shared()
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
            {"x": FlextSettings},
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
        c = FlextContainer.shared()
        c._services = {
            "svc": m.ServiceRegistration(name="svc", service="v", service_type="str"),
        }
        c._factories = {"fac": m.FactoryRegistration(name="fac", factory=lambda: "v")}
        c._resources = {"res": m.ResourceRegistration(name="res", factory=lambda: "v")}
        setattr(c._di_services, "svc", "normalized")
        setattr(c._di_services, "fac", "normalized")
        setattr(c._di_resources, "res", "normalized")
        c.register_existing_providers()
        c._config = m.Core.Tests.FalseSettings()
        c._context = None
        c.register_core_services()

    def test_configure_with_resource_register_and_factory_error_paths(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        FlextContainer.reset_for_testing()
        c = FlextContainer.shared()
        c._user_overrides = t.ConfigMap(root={})
        c._global_config = m.ContainerConfig(
            enable_singleton=True,
            enable_factory_caching=False,
            max_services=100,
            max_factories=100,
        )
        c.apply({"enable_factory_caching": True})
        tm.that(c._global_config.enable_factory_caching, eq=True)
        c.resource("res", lambda: "x")
        tm.that(c._resources, has="res")
        monkeypatch.setattr(
            core_container.u.DependencyIntegration,
            "register_object",
            self._raise_register_object,
        )
        c.bind("x", "y")
        monkeypatch.setattr(
            core_container.u.DependencyIntegration,
            "register_factory",
            self._raise_register_factory,
        )
        c.factory("x2", lambda: "v")
        setattr(c._di_resources, "dup", "normalized")
        c.resource("dup", lambda: "v")
        del c._di_resources.dup
        monkeypatch.setattr(
            core_container.u.DependencyIntegration,
            "register_resource",
            self._raise_register_resource,
        )
        c.resource("new", lambda: "v")

    def test_get_and_get_typed_resource_factory_paths(self) -> None:
        c = FlextContainer.shared()
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
        tm.fail(c.resolve("r"))
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
        c = FlextContainer.shared()
        _ = c.resource("resx", lambda: "ok")
        tm.that(c.logger(), ne=None)
        tm.ok(c.drop("resx"))
        _ = c.resource("r1", lambda: "ok")
        c.clear()
        FlextContainer.reset_for_testing()
        instance = FlextContainer.shared()
        tm.that(instance._global_instance, is_=type(instance))

    def test_scoped_config_context_branches(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        c = FlextContainer.shared()
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
        _ = c.scope(subproject="sub")
        tm.that(captured["settings"], is_=p.Settings)
        false_settings: p.Settings = m.Core.Tests.FalseSettings()
        _ = c.scope(
            settings=false_settings,
            context=FlextContext(),
        )
        tm.that(captured["context"], is_=p.Context)

    def test_create_auto_register_factory_wrapper_callable_and_non_callable(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: MutableMapping[str, t.RegisterableService] = {}
        FlextContainer.reset_for_testing()

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

        def capture_factory(
            self: p.Container,
            name: str,
            impl: t.RegisterableService,
        ) -> p.Container:
            if callable(impl):
                captured[name] = impl
            return self

        monkeypatch.setattr(FlextContainer, "factory", capture_factory)
        _ = FlextContainer.shared(auto_register_factories=True)
        wrapper_raw = captured["factory.captured"]
        assert callable(wrapper_raw)
        wrapper: Callable[..., t.RegisterableService] = wrapper_raw
        assert wrapper() == 7
        assert (
            wrapper(
                _factory_config=types.SimpleNamespace(fn=123, name="factory.captured")
            )
            == 7
        )

    def test_initialize_di_components_exposes_valid_runtime_providers(self) -> None:
        c = FlextContainer.shared()
        c.initialize_di_components()
        assert c._config_provider is not None
        assert c._base_config_provider is not None
        assert c._user_config_provider is not None

    def test_sync_config_registers_namespace_factories_and_fallbacks(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        container = FlextContainer.shared()

        class _NsAlpha(FlextSettings):
            v: str = "ok"

        class _NsBeta(FlextSettings):
            v: str = "ok2"

        # Register namespaces so FlextSettings.resolve_namespace_settings() finds them.
        original_registry = dict(FlextSettings._namespace_registry)
        FlextSettings._namespace_registry["alpha"] = _NsAlpha
        FlextSettings._namespace_registry["beta"] = _NsBeta

        try:

            class _Cfg(m.Core.Tests.FalseSettings):
                _namespace_registry: ClassVar[Mapping[str, type[p.Settings]]] = {
                    "alpha": _NsAlpha,
                    "beta": _NsBeta,
                }

            container._config = _Cfg()
            container._global_config = m.ContainerConfig(
                enable_singleton=True,
                enable_factory_caching=False,
                max_services=10,
                max_factories=10,
            )
            container._user_overrides = t.ConfigMap(root={})
            registered: MutableMapping[str, t.RegisterableService] = {}

            def _factory(
                name: str,
                impl: t.RegisterableService,
            ) -> p.Container:
                if callable(impl):
                    registered[name] = impl
                return container

            monkeypatch.setattr(container, "factory", _factory)
            container.sync_config_to_di()
            alpha_factory = registered["settings.alpha"]
            assert callable(alpha_factory)
            with pytest.raises(TypeError, match="must be a Pydantic model"):
                _ = alpha_factory()
            beta_factory = registered["settings.beta"]
            assert callable(beta_factory)
            with pytest.raises(TypeError, match="must be a Pydantic model"):
                _ = beta_factory()
        finally:
            FlextSettings._namespace_registry.clear()
            FlextSettings._namespace_registry.update(original_registry)

    def test_register_existing_providers_full_paths_and_misc_methods(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        typed_value_cls = self._typed_value_cls()
        c = FlextContainer.shared()
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
        tm.that(c.has("context"), is_=bool)
        c.wire(modules=[])
        assert c.resolve("r1").success
        tm.ok(self._get_with_type(c, "f1", typed_value_cls))
        tm.ok(self._get_with_type(c, "r1", typed_value_cls))

    def test_create_scoped_instance_and_scoped_additional_branches(self) -> None:
        base = FlextContainer.shared()
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
        base._config = m.Core.Tests.FalseSettings()
        base._context = FlextContext()
        _ = base.scope(
            settings=FlextSettings(app_name="x"),
            services={"sx": "vx"},
            factories={"fx": lambda: "fv"},
            resources={"rx": lambda: "rv"},
        )
        base._config = m.Core.Tests.FalseSettings()
        base._context = FlextContext()
        _ = base.scope()

    def test_additional_container_branches_cover_fluent_and_lookup_paths(self) -> None:
        typed_value_cls = self._typed_value_cls()
        c = FlextContainer.shared()
        c._global_config = m.ContainerConfig(
            enable_singleton=True,
            enable_factory_caching=False,
            max_services=10,
            max_factories=10,
        )
        global_instance = FlextContainer.shared()
        tm.that(global_instance, is_=p.Container)
        tm.that(c.apply({"enable_factory_caching": True}), eq=c)
        c.bind("svc-x", self.TypedValue(value="value"))
        c.factory("fac-x", lambda: "v")
        tm.that(c.snapshot().root, ne=None)
        c.bind("", "x")
        assert c.resolve("svc-x").success
        tm.fail(c.resolve("missing-service"))
        tm.ok(self._get_with_type(c, "svc-x", typed_value_cls))
        tm.fail(self._get_with_type(c, "missing-service", typed_value_cls))

    def test_additional_register_factory_and_unregister_paths(self) -> None:
        c = FlextContainer.shared()
        c._global_config = m.ContainerConfig(
            enable_singleton=True,
            enable_factory_caching=False,
            max_services=10,
            max_factories=10,
        )
        c.factory("fac-ok", lambda: 1)
        c.factory("fac-ok", lambda: 2)
        # Defensive path: non-callable impl is rejected by u.factory guard
        # (container.factory is a no-op when the guard narrows False).
        assert u.factory(123) is False
        _ = c.bind("svc-remove", "v")
        _ = c.resource("res-remove", lambda: "r")
        _ = c.factory("fac-remove", lambda: "f")
        tm.ok(c.drop("svc-remove"))
        tm.ok(c.drop("fac-remove"))
        tm.ok(c.drop("res-remove"))
        tm.fail(c.drop("not-there"))

    def test_container_remaining_branch_paths_in_sync_factory_and_getters(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        container = FlextContainer.shared()
        container._global_config = m.ContainerConfig(
            enable_singleton=True,
            enable_factory_caching=False,
            max_services=10,
            max_factories=10,
        )
        container._user_overrides = t.ConfigMap(root={})

        # --- _CfgNoMethod: namespace_registry without resolve_namespace_settings ---
        # n1 is NOT registered in FlextSettings, so resolve_namespace_settings returns None
        # and sync_config_to_di skips it (continue branch).
        class _CfgNoMethod(m.Core.Tests.FalseSettings):
            _namespace_registry: ClassVar[Mapping[str, type[p.Settings]]] = {
                "n1": FlextSettings,
            }

        container._config = _CfgNoMethod()
        container.sync_config_to_di()

        # --- Create real BaseSettings subclasses for n2, n3, n4 ---
        class _NsModel(FlextSettings):
            v: str = "x"

        # Register namespaces in FlextSettings._namespace_registry so that
        # sync_config_to_di -> FlextSettings.resolve_namespace_settings() finds them.
        original_registry = dict(FlextSettings._namespace_registry)
        FlextSettings._namespace_registry["n2"] = _NsModel
        FlextSettings._namespace_registry["n3"] = _NsModel
        FlextSettings._namespace_registry["n4"] = _NsModel

        try:

            class _CfgFallback(m.Core.Tests.FalseSettings):
                _namespace_registry: ClassVar[Mapping[str, type[p.Settings]]] = {
                    "n2": _NsModel,
                }

            class _CfgBadNamespace(m.Core.Tests.FalseSettings):
                _namespace_registry: ClassVar[Mapping[str, type[p.Settings]]] = {
                    "n3": _NsModel,
                }

            class _CfgGoodNamespace(m.Core.Tests.FalseSettings):
                _namespace_registry: ClassVar[Mapping[str, type[p.Settings]]] = {
                    "n4": _NsModel,
                }

            container._config = _CfgFallback()
            captured: MutableMapping[str, t.RegisterableService] = {}

            def _capture_factory(
                name: str,
                impl: t.RegisterableService,
            ) -> p.Container:
                if callable(impl):
                    captured[name] = impl
                return container

            monkeypatch.setattr(container, "factory", _capture_factory)
            container.sync_config_to_di()
            container._config = _CfgBadNamespace()
            container.sync_config_to_di()
            container._config = _CfgGoodNamespace()
            container.sync_config_to_di()
            for namespace_key in ("settings.n2", "settings.n3", "settings.n4"):
                ns_factory_raw = captured[namespace_key]
                assert callable(ns_factory_raw)
                namespace_factory: Callable[..., t.RegisterableService] = ns_factory_raw
                with pytest.raises(TypeError, match="must be a Pydantic model"):
                    _ = namespace_factory()
            c2 = FlextContainer.shared()
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
            fac_call = c2.resolve("fac-call", type_cls=str)
            tm.ok(fac_call)
            assert fac_call.value == "value"
            c2._factories = {
                "boom": m.FactoryRegistration(
                    name="boom",
                    factory=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
                ),
            }
            tm.fail(c2.resolve("boom"))
            c2._factories = {
                "ok-factory": m.FactoryRegistration(
                    name="ok-factory",
                    factory=lambda: "ok",
                ),
            }
            assert c2.resolve("ok-factory").success
            c2._services = {
                "svc-int": m.ServiceRegistration(
                    name="svc-int",
                    service="str",
                    service_type="str",
                ),
            }
            tm.fail(self._get_with_type(c2, "svc-int", self._typed_value_cls()))

            container._config = _CfgFallback()
            container.sync_config_to_di()
            container._config = _CfgBadNamespace()
            container.sync_config_to_di()
        finally:
            # Restore original registry
            FlextSettings._namespace_registry.clear()
            FlextSettings._namespace_registry.update(original_registry)
