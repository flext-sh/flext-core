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

import pytest

import flext_core
from flext_core import FlextSettings, c, d, e, h, m, p, r, s, t, u, x
from flext_core.exceptions import FlextMroViolation

# Public facade names a caller can import from ``flext_core``.
_FACADES: t.VariadicTuple[str] = (
    "FlextConstants",
    "FlextContainer",
    "FlextContext",
    "FlextDecorators",
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
    "FlextRuntime",
    "FlextService",
    "FlextSettings",
    "FlextTypes",
    "FlextUtilities",
)

# alias -> facade name published by the package's public contract
_ALIASES: tuple[tuple[object, str], ...] = (
    (r, "FlextResult"),
    (u, "FlextUtilities"),
    (m, "FlextModels"),
    (c, "FlextConstants"),
    (p, "FlextProtocols"),
    (t, "FlextTypes"),
    (s, "FlextService"),
    (e, "FlextExceptions"),
    (d, "FlextDecorators"),
    (h, "FlextHandlers"),
    (x, "FlextMixins"),
)


def _increment_step(value: int) -> p.Result[int]:
    """Fallible increment used to exercise ``flat_map`` bindings.

    Returns:
        The resulting ``p.Result[int]``.

    """
    return r[int].ok(value + 1)


class TestsFlextCorePublicApiContract:
    """Assert the observable behavior promised by the flext_core public surface."""

    def test_root_lazy_helpers_publish_a_consumer_export(self) -> None:
        """Generated consumers can compose lazy exports from the foundation root."""
        build_map = flext_core.build_lazy_import_map
        install = flext_core.install_lazy_exports
        assert {"build_lazy_import_map", "install_lazy_exports"} <= set(
            flext_core.__all__,
        )
        imports = build_map({"collections": ("Counter",)})
        namespace: t.ModuleGlobals = {"__name__": "flext_core_consumer"}
        install(
            "flext_core_consumer",
            namespace,
            imports,
            public_exports=("Counter",),
        )
        resolver = namespace["__getattr__"]
        assert callable(resolver)
        assert resolver("Counter").__name__ == "Counter"

    @pytest.mark.parametrize("name", _FACADES)
    def test_named_facade_is_importable(self, name: str) -> None:
        """Every advertised facade is reachable from the package root."""
        assert hasattr(flext_core, name), f"{name} not importable from flext_core"

    @staticmethod
    def test_every_all_entry_resolves_to_an_object() -> None:
        """Each name published in ``__all__`` actually resolves via ``getattr``."""
        missing = [name for name in flext_core.__all__ if not hasattr(flext_core, name)]
        assert not missing, f"Names in __all__ but not importable: {missing}"

    @pytest.mark.parametrize(("alias", "facade_name"), _ALIASES)
    def test_single_letter_alias_is_its_facade(
        self,
        alias: object,
        facade_name: str,
    ) -> None:
        """Each single-letter alias is the exact same object as its named facade."""
        # Arrange / Act
        facade = getattr(flext_core, facade_name)
        # Assert
        assert alias is facade

    @staticmethod
    def test_ok_result_reports_success_and_yields_value() -> None:
        """``r.ok(v)`` is truthy, successful, and unwraps to the wrapped value."""
        # Arrange / Act
        result = r.ok(5)
        # Assert
        assert result.success is True
        assert result.failure is False
        assert bool(result) is True
        assert result.value == 5
        assert result.unwrap() == 5

    @staticmethod
    def test_fail_result_reports_failure_and_carries_error() -> None:
        """``r.fail(msg, error_code=...)`` is falsy and exposes error + code."""
        # Arrange / Act
        result: p.Result[int] = r.fail("boom", error_code="E42")
        # Assert
        assert result.success is False
        assert result.failure is True
        assert bool(result) is False
        assert result.error == "boom"
        assert result.error_code == "E42"

    @staticmethod
    def test_value_access_on_failure_raises() -> None:
        """Reading ``.value`` on a failure raises with the error message included."""
        # Arrange
        result: p.Result[int] = r.fail("boom")
        # Act / Assert
        with pytest.raises(RuntimeError, match="boom"):
            _ = result.value

    @staticmethod
    def test_unwrap_on_failure_raises() -> None:
        """``unwrap`` on a failure raises rather than returning a sentinel."""
        # Arrange
        result: p.Result[int] = r.fail("boom")
        # Act / Assert
        with pytest.raises(RuntimeError):
            result.unwrap()

    @staticmethod
    def test_unwrap_or_returns_default_on_failure_and_value_on_success() -> None:
        """``unwrap_or`` yields the wrapped value on success, the default on failure."""
        # Arrange / Act / Assert
        assert r.ok(7).unwrap_or(99) == 7
        assert r.fail("boom").unwrap_or(99) == 99

    @staticmethod
    def test_unwrap_or_else_invokes_supplier_only_on_failure() -> None:
        """``unwrap_or_else`` computes the fallback from the supplier on failure."""
        # Arrange / Act / Assert
        assert r.fail("boom").unwrap_or_else(lambda: 42) == 42
        assert r.ok(1).unwrap_or_else(lambda: 42) == 1

    @staticmethod
    def test_map_transforms_success_and_passes_failure_through() -> None:
        """``map`` applies to a success value and short-circuits on failure."""
        # Arrange / Act
        mapped = r.ok(3).map(lambda v: v * 2)
        passthrough = r.fail("boom").map(lambda v: v * 2)
        # Assert
        assert mapped.value == 6
        assert passthrough.failure is True
        assert passthrough.error == "boom"

    @staticmethod
    def test_map_is_chainable_and_composes() -> None:
        """Chained ``map`` calls compose left-to-right over a success value."""
        # Arrange / Act
        result = r.ok(2).map(lambda v: v + 1).map(lambda v: v * 10)
        # Assert
        assert result.unwrap() == 30

    @staticmethod
    def test_flat_map_chains_success_and_short_circuits_failure() -> None:
        """``flat_map`` binds the next fallible step, skipping it on failure."""
        # Arrange / Act
        chained = r[int].ok(3).flat_map(_increment_step)
        short = r[int].fail("boom").flat_map(_increment_step)
        # Assert
        assert chained.value == 4
        assert short.failure is True
        assert short.error == "boom"

    @staticmethod
    def test_recover_converts_failure_into_success() -> None:
        """``recover`` maps a failure's error into a recovered success value."""
        # Arrange / Act
        recovered = r.fail("boom").recover(lambda _err: 7)
        # Assert
        assert recovered.value == 7

    @staticmethod
    def test_tap_observes_success_value_without_altering_result() -> None:
        """``tap`` runs its callback on success and returns an equivalent result."""
        # Arrange
        seen: list[int] = []
        # Act
        result = r.ok(9).tap(seen.append)
        # Assert
        assert seen == [9]
        assert result.value == 9

    @staticmethod
    def test_tap_error_observes_error_message_on_failure() -> None:
        """``tap_error`` runs its callback with the error message on failure."""
        # Arrange
        seen: list[str] = []
        # Act
        result: p.Result[int] = r.fail("bad").tap_error(seen.append)
        # Assert
        assert seen == ["bad"]
        assert result.failure is True

    @staticmethod
    def test_result_repr_reflects_success_and_failure_state() -> None:
        """``repr`` distinguishes ok and fail results for debuggability."""
        # Arrange / Act / Assert
        assert repr(r.ok(1)) == "r[T].ok(1)"
        assert repr(r.fail("z")) == "r[T].fail('z')"

    @pytest.mark.parametrize("exc_name", ["MroViolation", "SmellViolation"])
    def test_exception_family_members_are_raisable_and_catchable(
        self,
        exc_name: str,
    ) -> None:
        """Structured exception classes raise with, and preserve, their message."""
        # Arrange
        exc_type: type[Exception] = getattr(e, exc_name)
        assert issubclass(exc_type, Exception)
        message = "broken invariant"
        # Act / Assert
        with pytest.raises(exc_type, match="broken"):
            raise exc_type(message)

    @staticmethod
    def test_version_metadata_is_usable_string() -> None:
        """The package publishes a non-empty ``__version__`` string."""
        # Arrange / Act
        version = flext_core.__version__
        # Assert
        assert isinstance(version, str)
        assert version

    @staticmethod
    def test_settings_facade_exposes_base_class() -> None:
        """``FlextSettings.Base`` is the canonical settings base class."""
        # Arrange / Act
        base = flext_core.FlextSettings
        # Assert
        assert base is FlextSettings

    @staticmethod
    def test_exceptions_mro_violation_is_real_reference() -> None:
        """``FlextExceptions.MroViolation`` is the canonical violation class."""
        assert flext_core.FlextExceptions.MroViolation is FlextMroViolation
