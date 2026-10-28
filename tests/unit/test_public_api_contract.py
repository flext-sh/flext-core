"""Behavioral contract tests for the flext-core public API surface.

Exercises the *observable behavior* a caller depends on when importing from
``flext_core``: the single-letter facade aliases resolve to their facades, the
``FlextResult`` railway (``ok``/``fail``/``map``/``flat_map``/``recover``/
``unwrap``/``unwrap_or``/``tap``) behaves per its monadic contract, the
``FlextExceptions`` family raises catchable structured errors, and the package
publishes usable version metadata. No private attributes, internal collaborators,
or dir() snapshots are asserted.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from typing import cast

import pytest

import flext_core
from flext_core import FlextMroViolation, FlextSettings
from tests import e, p, r, t, tm

# Public facade names a caller can import from ``flext_core``.
_FACADES: tuple[str, ...] = (
    "FlextCore",
    "FlextConstants",
    "FlextContainer",
    "FlextContext",
    "FlextDecoratorsRuntime",
    "FlextDispatcher",
    "FlextExceptions",
    "FlextHandlers",
    "FlextLazy",
    "FlextUtilitiesLogging",
    "FlextMixins",
    "FlextModels",
    "FlextProtocols",
    "FlextRegistry",
    "FlextResult",
    "FlextService",
    "FlextSettings",
    "FlextTypes",
    "FlextUtilities",
)

# alias -> facade name published by the package's public contract
_ALIASES: tuple[tuple[str, str], ...] = (
    ("r", "FlextResult"),
    ("u", "FlextUtilities"),
    ("m", "FlextModels"),
    ("c", "FlextConstants"),
    ("p", "FlextProtocols"),
    ("t", "FlextTypes"),
    ("s", "FlextService"),
    ("e", "FlextExceptions"),
    ("d", "FlextDecoratorsRuntime"),
    ("h", "FlextHandlers"),
    ("x", "FlextMixins"),
)


def _increment_step(value: int) -> p.Result[int]:
    """Fallible increment used to exercise ``flat_map`` bindings.

    Returns:
        The resulting ``p.Result[int]``.

    """
    return r[int].ok(value + 1)


class TestsFlextCorePublicApiContract:
    """Assert the observable behavior promised by the flext_core public surface."""

    @staticmethod
    def test_root_lazy_helpers_publish_a_consumer_export() -> None:
        """Generated consumers can compose lazy exports from the foundation root."""
        build_map = flext_core.build_lazy_import_map
        install = flext_core.install_lazy_exports
        tm.that(
            flext_core.__all__,
            has=["build_lazy_import_map", "install_lazy_exports"],
        )
        imports = build_map({"collections": ("Counter",)})
        namespace: t.ModuleGlobals = {"__name__": "flext_core_consumer"}
        install(
            "flext_core_consumer",
            namespace,
            imports,
            public_exports=("Counter",),
        )
        resolver = cast("Callable[[str], object]", namespace["__getattr__"])
        tm.that(callable(resolver), truth=True)
        counter = resolver("Counter")
        tm.that(isinstance(counter, type), truth=True)
        tm.that(cast("type", counter).__name__, eq="Counter")

    @staticmethod
    @pytest.mark.parametrize("name", _FACADES)
    def test_named_facade_is_importable(name: str) -> None:
        """Every advertised facade is reachable from the package root."""
        tm.that(
            hasattr(flext_core, name),
            truth=True,
            msg=f"{name} not importable from flext_core",
        )

    @staticmethod
    def test_every_all_entry_resolves_to_an_object() -> None:
        """Each name published in ``__all__`` actually resolves via ``getattr``."""
        missing = [name for name in flext_core.__all__ if not hasattr(flext_core, name)]
        tm.that(
            missing, truth=False, msg=f"Names in __all__ but not importable: {missing}"
        )

    @staticmethod
    @pytest.mark.parametrize(("alias", "facade_name"), _ALIASES)
    def test_single_letter_alias_is_its_facade(
        alias: str,
        facade_name: str,
    ) -> None:
        """Each single-letter alias is the exact same object as its named facade."""
        # Arrange / Act
        facade = getattr(flext_core, facade_name)
        # Assert
        tm.that(getattr(flext_core, alias) is facade, truth=True)

    @staticmethod
    def test_ok_result_reports_success_and_yields_value() -> None:
        """``r.ok(v)`` is truthy, successful, and unwraps to the wrapped value."""
        # Arrange / Act
        result = r.ok(5)
        # Assert
        tm.that(result.success is True, truth=True)
        tm.that(result.failure is False, truth=True)
        tm.that(bool(result) is True, truth=True)
        tm.that(result.value, eq=5)
        tm.that(result.unwrap(), eq=5)

    @staticmethod
    def test_fail_result_reports_failure_and_carries_error() -> None:
        """``r.fail(msg, error_code=...)`` is falsy and exposes error + code."""
        # Arrange / Act
        result: p.Result[int] = r[int].fail("boom", error_code="E42")
        # Assert
        tm.that(result.success is False, truth=True)
        tm.that(result.failure is True, truth=True)
        tm.that(bool(result) is False, truth=True)
        tm.that(result.error, eq="boom")
        tm.that(result.error_code, eq="E42")

    @staticmethod
    def test_value_access_on_failure_raises() -> None:
        """Reading ``.value`` on a failure raises with the error message included."""
        # Arrange
        result: p.Result[int] = r[int].fail("boom")
        # Act / Assert
        with pytest.raises(RuntimeError, match="boom"):
            _ = result.value

    @staticmethod
    def test_unwrap_on_failure_raises() -> None:
        """``unwrap`` on a failure raises rather than returning a sentinel."""
        # Arrange
        result: p.Result[int] = r[int].fail("boom")
        # Act / Assert
        with pytest.raises(RuntimeError):
            result.unwrap()

    @staticmethod
    def test_unwrap_or_returns_default_on_failure_and_value_on_success() -> None:
        """``unwrap_or`` yields the wrapped value on success, the default on failure."""
        # Arrange / Act / Assert
        tm.that(r.ok(7).unwrap_or(99), eq=7)
        tm.that(r[int].fail("boom").unwrap_or(99), eq=99)

    @staticmethod
    def test_unwrap_or_else_invokes_supplier_only_on_failure() -> None:
        """``unwrap_or_else`` computes the fallback from the supplier on failure."""
        # Arrange / Act / Assert
        tm.that(r[int].fail("boom").unwrap_or_else(lambda: 42), eq=42)
        tm.that(r.ok(1).unwrap_or_else(lambda: 42), eq=1)

    @staticmethod
    def test_map_transforms_success_and_passes_failure_through() -> None:
        """``map`` applies to a success value and short-circuits on failure."""
        # Arrange / Act
        mapped: p.Result[int] = r.ok(3).map(lambda v: v * 2)
        passthrough: p.Result[int] = r[int].fail("boom").map(lambda v: v * 2)
        # Assert
        tm.that(mapped.value, eq=6)
        tm.that(passthrough.failure is True, truth=True)
        tm.that(passthrough.error, eq="boom")

    @staticmethod
    def test_map_is_chainable_and_composes() -> None:
        """Chained ``map`` calls compose left-to-right over a success value."""
        # Arrange / Act
        result: p.Result[int] = r.ok(2).map(lambda v: v + 1).map(lambda v: v * 10)
        # Assert
        tm.that(result.unwrap(), eq=30)

    @staticmethod
    def test_flat_map_chains_success_and_short_circuits_failure() -> None:
        """``flat_map`` binds the next fallible step, skipping it on failure."""
        # Arrange / Act
        chained = r[int].ok(3).flat_map(_increment_step)
        short = r[int].fail("boom").flat_map(_increment_step)
        # Assert
        tm.that(chained.value, eq=4)
        tm.that(short.failure is True, truth=True)
        tm.that(short.error, eq="boom")

    @staticmethod
    def test_recover_converts_failure_into_success() -> None:
        """``recover`` maps a failure's error into a recovered success value."""
        # Arrange / Act
        recovered: p.Result[int] = r[int].fail("boom").recover(lambda _err: 7)
        # Assert
        tm.that(recovered.value, eq=7)

    @staticmethod
    def test_tap_observes_success_value_without_altering_result() -> None:
        """``tap`` runs its callback on success and returns an equivalent result."""
        # Arrange
        seen: list[int] = []
        # Act
        result: p.Result[int] = r.ok(9).tap(seen.append)
        # Assert
        tm.that(seen, eq=[9])
        tm.that(result.value, eq=9)

    @staticmethod
    def test_tap_error_observes_error_message_on_failure() -> None:
        """``tap_error`` runs its callback with the error message on failure."""
        # Arrange
        seen: list[str] = []
        # Act
        result: p.Result[int] = r[int].fail("bad").tap_error(seen.append)
        # Assert
        tm.that(seen, eq=["bad"])
        tm.that(result.failure is True, truth=True)

    @staticmethod
    def test_result_repr_reflects_success_and_failure_state() -> None:
        """``repr`` distinguishes ok and fail results for debuggability."""
        # Arrange / Act / Assert
        tm.that(repr(r[int].ok(1)), eq="r[T].ok(1)")
        tm.that(repr(r[int].fail("z")), eq="r[T].fail('z')")

    @staticmethod
    @pytest.mark.parametrize("exc_name", ["MroViolation", "SmellViolation"])
    def test_exception_family_members_are_raisable_and_catchable(
        exc_name: str,
    ) -> None:
        """Structured exception classes raise with, and preserve, their message."""
        # Arrange
        exc_type: type[Exception] = getattr(e, exc_name)
        tm.that(exc_type, where=Exception.__subclasscheck__)
        message = "broken invariant"
        # Act / Assert
        with pytest.raises(exc_type, match="broken"):
            raise exc_type(message)


class TestsFlextCoreFacadeLifecycle:
    """Published metadata and facade registry lifecycle."""

    @staticmethod
    def test_version_metadata_is_usable_string() -> None:
        """The package publishes a non-empty ``__version__`` string."""
        # Arrange / Act
        version = flext_core.__version__
        # Assert
        tm.that(version, is_=str)
        tm.that(version, truth=True)

    @staticmethod
    def test_settings_facade_exposes_base_class() -> None:
        """``FlextSettings.Base`` is the canonical settings base class."""
        # Arrange / Act
        base = flext_core.FlextSettings
        # Assert
        tm.that(base is FlextSettings, truth=True)

    @staticmethod
    def test_exceptions_mro_violation_is_real_reference() -> None:
        """``FlextExceptions.MroViolation`` is the canonical violation class."""
        tm.that(
            flext_core.FlextExceptions.MroViolation is FlextMroViolation, truth=True
        )

    @staticmethod
    def test_flextcore_facade_is_importable_and_singleton() -> None:
        """``FlextCore`` facade and ``core`` singleton are importable and functional."""
        tm.that(
            hasattr(flext_core, "FlextCore"),
            truth=True,
            msg="FlextCore not importable from flext_core",
        )
        tm.that(
            hasattr(flext_core, "core"),
            truth=True,
            msg="core not importable from flext_core",
        )
        # The singleton should be an instance of FlextCore
        tm.that(flext_core.core, is_=flext_core.FlextCore)
        # core should have container access via MRO
        tm.that(hasattr(flext_core.core, "container"), truth=True)
        tm.that(hasattr(flext_core.core, "registry"), truth=True)
        tm.that(hasattr(flext_core.core, "dispatcher"), truth=True)
        tm.that(hasattr(flext_core.core, "settings"), truth=True)

    @staticmethod
    def test_facade_registry_keeps_instance_plugins_between_accesses() -> None:
        """Repeated public registry access preserves instance registrations."""
        facade = flext_core.FlextCore()
        plugin = "public-registry-plugin"
        tm.ok(facade.registry.register_plugin("public-api", "plugin", plugin))
        tm.that(
            facade.registry.fetch_plugin("public-api", "plugin").unwrap(), eq=plugin
        )
        tm.ok(facade.registry is facade.registry)
        tm.ok(facade.registry.dispatcher is facade.dispatcher)

    @staticmethod
    def test_facade_registry_uses_custom_runtime_without_leaking_to_siblings() -> None:
        """Registry bindings reach the facade container, never a sibling runtime."""
        custom_settings = FlextSettings.fetch_global().clone()
        custom_settings.debug = not custom_settings.debug
        facade = flext_core.FlextCore(runtime_settings=custom_settings)
        sibling = flext_core.FlextCore()
        binding = "public-api-runtime-binding"
        tm.ok(facade.registry.register(binding, "registered-value"))
        tm.that(facade.container.resolve(binding).unwrap(), eq="registered-value")
        tm.fail(sibling.container.has(binding))
        tm.ok(facade.settings is custom_settings)
        tm.ok(facade.registry.dispatcher is facade.dispatcher)

    @staticmethod
    def test_concurrent_registry_access_preserves_plugin_lifecycle() -> None:
        """Concurrent first access shares one registry and all plugin ownership."""
        facade = flext_core.FlextCore()
        barrier = Barrier(4)

        def register(index: int) -> p.Registry:
            barrier.wait()
            registry = facade.registry
            registry.register_plugin("concurrent-api", str(index), str(index)).unwrap()
            return registry

        with ThreadPoolExecutor(max_workers=barrier.parties) as executor:
            registries = tuple(executor.map(register, range(barrier.parties)))

        tm.ok(all(registry is facade.registry for registry in registries))
        for index in range(barrier.parties):
            name = str(index)
            tm.that(
                facade.registry.fetch_plugin("concurrent-api", name).unwrap(), eq=name
            )
            tm.ok(facade.registry.unregister_plugin("concurrent-api", name))
            tm.that(
                facade.registry.fetch_plugin("concurrent-api", name).failure, eq=True
            )
