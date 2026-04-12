"""Tests for Mixins."""

from __future__ import annotations

from collections.abc import (
    MutableSequence,
)
from types import SimpleNamespace
from typing import cast, override

import pytest

import flext_core as core_mixins
from flext_core import r, x
from flext_tests import tm
from tests import m, p, t


class TestMixinsFullCoverage:
    @staticmethod
    def _noop(*_a: t.Scalar, **_k: t.Scalar) -> None:
        """Typed no-op for protocol stubs."""
        return

    @staticmethod
    def _return_true(*_a: t.Scalar, **_k: t.Scalar) -> bool:
        """Typed return-True for protocol stubs."""
        return True

    @staticmethod
    def _return_true_no_args() -> bool:
        """Typed callable for process/validate stubs."""
        return True

    @staticmethod
    def _mock_register_fail(_name: str) -> r[bool]:
        """Mock _register_in_container that returns failure."""
        return cast(
            "r[bool]",
            cast("t.RecursiveContainer", SimpleNamespace(failure=True, error=None)),
        )

    @staticmethod
    def _validation_ok_true(v: t.RecursiveContainer) -> r[bool]:
        """Validator that always returns ok(True)."""
        _ = v
        return r[bool].ok(True)

    @staticmethod
    def _validation_ok_false(v: t.RecursiveContainer) -> r[bool]:
        """Validator that always returns ok(False)."""
        _ = v
        return r[bool].ok(False)

    @staticmethod
    def _validation_fail_no(v: t.RecursiveContainer) -> r[bool]:
        """Validator that always returns fail('no')."""
        _ = v
        return r[bool].fail("no")

    class _RuntimeContainer:
        def __init__(self) -> None:
            super().__init__()
            self.configured: t.RecursiveContainerMapping | None = None
            self.wired: t.RecursiveContainerMapping | None = None

        def scoped(
            self,
            **_kwargs: t.Scalar,
        ) -> TestMixinsFullCoverage._RuntimeContainer:
            return self

        def configure(self, overrides: t.RecursiveContainerMapping) -> None:
            self.configured = overrides

        def wire_modules(self, **kwargs: t.Scalar) -> None:
            self.wired = dict(kwargs)

        def list_services(self) -> t.StrSequence:
            return list[str]()

        def has_service(self, _name: str) -> bool:
            return False

        def register(
            self,
            _name: str,
            _value: t.RecursiveContainer,
            *,
            kind: str = "service",
        ) -> TestMixinsFullCoverage._RuntimeContainer:
            _ = kind
            return self

        def clear_all(self) -> None:
            return None

        def resolve_settings(self) -> t.ConfigMap:
            return t.ConfigMap(root={})

        def get(self, _key: str, **_kwargs: t.Scalar) -> r[t.RecursiveContainer]:
            return r[t.RecursiveContainer].fail("not implemented")

        @property
        def context(self) -> None:
            return None

        @property
        def settings(self) -> None:
            return None

    class _ContainerForLogger:
        def __init__(
            self,
            success: bool,
            logger: t.RuntimeAtomic | None = None,
        ) -> None:
            super().__init__()
            self.success: bool = success
            self.logger: t.RuntimeAtomic | None = logger
            self.factories: t.MutableRecursiveContainerMapping = dict[
                str, t.RecursiveContainer
            ]()
            self.register_calls: MutableSequence[tuple[str, str]] = []

        def get_typed(
            self,
            _key: str,
            _tp: type[t.RuntimeAtomic],
        ) -> r[t.RuntimeAtomic]:
            if self.success:
                return r[t.RuntimeAtomic].ok(self.logger or "logger")
            return r[t.RuntimeAtomic].fail("missing")

        def get(
            self,
            _key: str,
            *,
            type_cls: type[t.RuntimeAtomic] | None = None,
        ) -> r[t.RuntimeAtomic]:
            _ = type_cls
            if self.success:
                return r[t.RuntimeAtomic].ok(self.logger or "logger")
            return r[t.RuntimeAtomic].fail("missing")

        def register_factory(self, key: str, factory: t.FactoryCallable) -> r[bool]:
            _ = key
            _ = factory
            msg = "register_factory path should not be used"
            raise AssertionError(msg)

        def register(
            self,
            name: str,
            value: t.RecursiveContainer,
            *,
            kind: str = "service",
        ) -> TestMixinsFullCoverage._ContainerForLogger:
            self.register_calls.append((name, kind))
            if kind == "factory":
                self.factories[name] = value
            _ = value
            return self

    def test_mixins_runtime_bootstrap_and_track_paths(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime_container = self._RuntimeContainer()
        monkeypatch.setattr(
            core_mixins.FlextContainer,
            "create",
            staticmethod(lambda: runtime_container),
        )

        class _Service(x):
            @classmethod
            def _runtime_bootstrap_options(cls) -> p.RuntimeBootstrapOptions:
                return cast(
                    "p.RuntimeBootstrapOptions",
                    cast(
                        "t.RecursiveContainer",
                        SimpleNamespace(
                            settings_type=None,
                            settings_overrides=None,
                            context=None,
                            services=None,
                            wire_modules=["pkg"],
                            resources={"res": lambda: "x"},
                            factories={"fac": lambda: "y"},
                        ),
                    ),
                )

        service = _Service(
            settings_type=None,
            settings_overrides=None,
            initial_context=None,
        )
        runtime = service._get_runtime()
        assert runtime is not None
        tm.that(runtime_container.wired, none=False)
        with service.track("op") as metrics:
            cast("t.MutableRecursiveContainerMapping", metrics)["duration_ms"] = 2.0
        tm.that(service._operation_stats, has="op")
        try:
            with service.track("op_fail"):
                msg = "boom"
                raise RuntimeError(msg)
        except RuntimeError:
            pass

    def test_mixins_container_registration_and_logger_paths(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class _Service(x):
            pass

        service = _Service(
            settings_type=None,
            settings_overrides=None,
            initial_context=None,
        )
        ok_register = service._register_in_container("svc")
        tm.ok(ok_register)

        class _AlreadyContainer:
            def has_service(self, _name: str) -> bool:
                return True

            def register(
                self,
                _name: str,
                _value: t.RecursiveContainer,
            ) -> _AlreadyContainer:
                return self

        monkeypatch.setattr(
            _Service,
            "container",
            property(lambda _self: _AlreadyContainer()),
        )
        tm.ok(service._register_in_container("svc"))

        class _FailContainer:
            def has_service(self, _name: str) -> bool:
                return False

            def register(
                self, _name: str, _value: t.RecursiveContainer
            ) -> _FailContainer:
                return self

        monkeypatch.setattr(
            _Service,
            "container",
            property(lambda _self: _FailContainer()),
        )
        service._init_service("svc")
        x._logger_cache.clear()
        monkeypatch.setattr(
            core_mixins.FlextContainer,
            "create",
            staticmethod(
                lambda: self._ContainerForLogger(True, logger="l"),
            ),
        )
        logger_from_di = _Service._get_or_create_logger()
        tm.that(logger_from_di, none=False)
        monkeypatch.setattr(
            core_mixins.FlextContainer,
            "create",
            staticmethod(lambda: self._ContainerForLogger(False)),
        )
        logger_created = _Service._get_or_create_logger()
        tm.that(logger_created, none=False)
        monkeypatch.setattr(
            core_mixins.FlextContainer,
            "create",
            staticmethod(lambda: (_ for _ in ()).throw(RuntimeError("no container"))),
        )
        logger_fallback = _Service._get_or_create_logger()
        tm.that(logger_fallback, none=False)

    def test_mixins_context_logging_and_cqrs_paths(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class _LocalLogger:
            def info(self, *_args: t.Scalar, **_kwargs: t.Scalar) -> None:
                return None

            def warning(self, *_args: t.Scalar, **_kwargs: t.Scalar) -> None:
                return None

        class _Service(x):
            @override
            @classmethod
            def _get_or_create_logger(cls) -> p.Logger:
                return cast("p.Logger", cast("t.RecursiveContainer", _LocalLogger()))

        service = _Service(
            settings_type=None,
            settings_overrides=None,
            initial_context=None,
        )
        settings = t.ConfigMap(root={"k": "v"})
        service._log_settings_once(settings, message="cfg")
        service._with_operation_context(
            "run",
            params="k=v",
            stack_trace="s",
            normal="n",
        )
        service._clear_operation_context()
        mt = m.MetricsTracker()
        result_record = mt.record_metric("k", 1)
        tm.ok(result_record)
        result_metrics = mt.metrics
        tm.that(result_metrics["k"], eq=1)
        cs = m.ContextStack()
        result_push1 = cs.push_context({"handler_name": "h", "handler_mode": "query"})
        tm.ok(result_push1)
        result_push2 = cs.push_context({"x": "y"})
        tm.ok(result_push2)
        result_pop = cs.pop_context()
        tm.ok(result_pop)
        result_pop2 = cs.pop_context()
        tm.ok(result_pop2)
        tm.that(cs.current_context(), none=True)
        result_push3 = cs.push_context({
            "handler_name": "h2",
            "handler_mode": "command",
        })
        tm.ok(result_push3)
        tm.that(cs.current_context(), none=False)

    def test_mixins_remaining_branch_paths(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        runtime_container = self._RuntimeContainer()

        def _create_runtime_container() -> TestMixinsFullCoverage._RuntimeContainer:
            return runtime_container

        monkeypatch.setattr(
            core_mixins.FlextContainer,
            "create",
            staticmethod(_create_runtime_container),
        )

        class _WireService(x):
            @classmethod
            def _runtime_bootstrap_options(cls) -> p.RuntimeBootstrapOptions:
                return cast(
                    "p.RuntimeBootstrapOptions",
                    cast("t.RecursiveContainer", {"wire_packages": ["pkg", 1]}),
                )

        _ = _WireService(
            settings_type=None,
            settings_overrides=None,
            initial_context=None,
        )._get_runtime()
        tm.that(runtime_container.wired, none=True)

        captured: t.MutableRecursiveContainerMapping = {}

        class _ModelService(x):
            pass

        model_service = _ModelService(
            settings_type=None,
            settings_overrides=None,
            initial_context=None,
        )

        class _RegContainer:
            def __init__(self) -> None:
                super().__init__()
                self._services: t.MutableRecursiveContainerMapping = dict[
                    str, t.RecursiveContainer
                ]()

            def has_service(self, name: str) -> bool:
                return name in self._services

            def register(self, name: str, value: t.RecursiveContainer) -> _RegContainer:
                self._services[name] = value
                captured["name"] = name
                captured["value"] = value
                return self

        def _container_getter(_self: _ModelService) -> _RegContainer:
            return _RegContainer()

        monkeypatch.setattr(
            _ModelService,
            "container",
            property(_container_getter),
        )
        tm.ok(model_service._register_in_container("svc_model"))
        tm.that(captured["value"], is_=_ModelService)

        class _WarnLogger:
            def warning(self, *_args: t.Scalar, **_kwargs: t.Scalar) -> None:
                return None

        class _WarnService(x):
            @override
            @classmethod
            def _get_or_create_logger(cls) -> p.Logger:
                return cast("p.Logger", cast("t.RecursiveContainer", _WarnLogger()))

        class _LoggerService(x):
            pass

        warn_service = _WarnService(
            settings_type=None,
            settings_overrides=None,
            initial_context=None,
        )
        monkeypatch.setattr(
            warn_service,
            "_register_in_container",
            self._mock_register_fail,
        )
        warn_service._init_service("svc_warn")
        tracker = m.MetricsTracker()
        tm.ok(tracker.record_metric("a", 1))
        tm.that(tracker.metrics["a"], eq=1)
        tracker2 = m.MetricsTracker()
        tm.ok(tracker2.record_metric("b", 2))
        tm.that(tracker2.metrics["b"], eq=2)
        stack = m.ContextStack()
        tm.ok(stack.push_context({"handler_name": "h", "handler_mode": "event"}))
        popped_dict = stack.pop_context()
        tm.ok(popped_dict)
        tm.that(
            popped_dict.value,
            eq={"handler_name": "h", "handler_mode": "event"},
        )
        tm.that(stack.current_context(), none=True)
        stack2 = m.ContextStack()
        tm.ok(stack2.push_context({"handler_name": "h2", "handler_mode": "query"}))
        popped2 = stack2.pop_context()
        tm.ok(popped2)
        tm.that(stack2.current_context(), none=True)
        x._logger_cache.clear()
        logger_obj = _LoggerService._get_or_create_logger()
        tm.that(logger_obj, none=False)

        class _BrokenContainer:
            def get_typed(
                self,
                _key: str,
                _tp: type[t.RuntimeAtomic],
            ) -> r[t.RuntimeAtomic]:
                msg = "boom"
                raise RuntimeError(msg)

        def _create_broken_container() -> _BrokenContainer:
            return _BrokenContainer()

        x._logger_cache.clear()
        monkeypatch.setattr(
            core_mixins.FlextContainer,
            "create",
            staticmethod(_create_broken_container),
        )
        fallback_logger = _LoggerService._get_or_create_logger()
        tm.that(fallback_logger, none=False)

    def test_mixins_context_stack_pop_empty_stack_returns_empty(self) -> None:
        _ = self
        stack = m.ContextStack()
        popped = stack.pop_context()
        tm.ok(popped)
        tm.that(popped.value, eq={})
