"""Contract test — flext-core 1.0.0 public API surface lock.

Asserts that:
- ``flext_core.__all__`` is exactly the 38-symbol frozen set.
- Every symbol in ``__all__`` is importable via ``getattr(flext_core, name)``.
- Single-letter alias → facade identity is preserved.
- ``FlextSettings.Base`` and ``FlextExceptions.MroViolation`` are real references.
- Each of the 19 facade classes' public attribute surface matches the golden snapshot.
- Dropped symbols (sub-facades removed from ``__all__``) are still reachable via
  lazy import (i.e. ``from flext_core import FlextModelsBase`` still works).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path
from typing import ClassVar

import flext_core
from flext_core._constants.enforcement import FlextMroViolation
from flext_core._settings.base import FlextSettingsBase

_GOLDEN_PATH = Path(__file__).parent / "_golden_public_api.json"

_FROZEN: frozenset[str] = frozenset({
    # version metadata (8)
    "__author__",
    "__author_email__",
    "__description__",
    "__license__",
    "__title__",
    "__url__",
    "__version__",
    "__version_info__",
    # facade classes (19)
    "FlextConstants",
    "FlextContainer",
    "FlextContext",
    "FlextDecorators",
    "FlextDispatcher",
    "FlextExceptions",
    "FlextHandlers",
    "FlextLazy",
    "FlextLogger",
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
    # single-letter aliases (11)
    "c",
    "d",
    "e",
    "h",
    "m",
    "p",
    "r",
    "s",
    "t",
    "u",
    "x",
})

_ALIAS_MAP: dict[str, str] = {
    "r": "FlextResult",
    "u": "FlextUtilities",
    "m": "FlextModels",
    "c": "FlextConstants",
    "p": "FlextProtocols",
    "t": "FlextTypes",
    "s": "FlextService",
    "e": "FlextExceptions",
    "d": "FlextDecorators",
    "h": "FlextHandlers",
    "x": "FlextMixins",
}

_FACADES: tuple[str, ...] = (
    "FlextConstants",
    "FlextContainer",
    "FlextContext",
    "FlextDecorators",
    "FlextDispatcher",
    "FlextExceptions",
    "FlextHandlers",
    "FlextLazy",
    "FlextLogger",
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

# Symbols removed from __all__ but kept resolvable via lazy __getattr__
_DROPPED_BUT_LAZY: tuple[str, ...] = (
    "FlextSettingsBase",
    "FlextModelsBase",
    "FlextUtilitiesText",
    "mc",
    "ube",
)


class TestsFlextCorePublicApiContract:
    """Lock the flext_core 1.0.0 public API surface against unintentional drift."""

    golden: ClassVar[dict[str, list[str]]] = json.loads(_GOLDEN_PATH.read_text())

    def test_all_equals_frozen_set(self) -> None:
        """``__all__`` must be exactly the 38-symbol frozen set."""
        actual = set(flext_core.__all__)
        assert actual == _FROZEN, (
            f"__all__ drift detected.\n"
            f"  extra  (in __all__ but not frozen): {sorted(actual - _FROZEN)}\n"
            f"  missing (in frozen but not __all__): {sorted(_FROZEN - actual)}"
        )

    def test_all_symbols_importable(self) -> None:
        """Every name in ``__all__`` must be reachable via ``getattr``."""
        missing = [
            name for name in flext_core.__all__
            if not hasattr(flext_core, name)
        ]
        assert not missing, f"Not importable from flext_core: {missing}"

    def test_alias_identity(self) -> None:
        """Single-letter aliases must be the exact same object as their facades."""
        for alias, facade_name in _ALIAS_MAP.items():
            alias_obj = getattr(flext_core, alias)
            facade_obj = getattr(flext_core, facade_name)
            assert alias_obj is facade_obj, (
                f"flext_core.{alias} is not flext_core.{facade_name}"
            )

    def test_settings_base_is_real_reference(self) -> None:
        """``FlextSettings.Base`` must be the actual ``FlextSettingsBase`` class."""
        assert flext_core.FlextSettings.Base is FlextSettingsBase

    def test_exceptions_mro_violation_is_real_reference(self) -> None:
        """``FlextExceptions.MroViolation`` must be the actual ``FlextMroViolation`` class."""
        assert flext_core.FlextExceptions.MroViolation is FlextMroViolation

    def test_facade_surfaces_match_golden(self) -> None:
        """Each facade's public attribute surface must match the golden snapshot exactly."""
        drift: list[str] = []
        for facade_name in _FACADES:
            cls = getattr(flext_core, facade_name)
            actual = sorted(a for a in dir(cls) if not a.startswith("_"))
            expected = self.golden.get(facade_name)
            if expected is None:
                drift.append(f"{facade_name}: missing from golden snapshot")
                continue
            extra = sorted(set(actual) - set(expected))
            missing = sorted(set(expected) - set(actual))
            if extra or missing:
                drift.append(
                    f"{facade_name}: extra={extra!r} missing={missing!r}"
                )
        assert not drift, "Facade surface drift:\n" + "\n".join(drift)

    def test_dropped_symbols_still_lazy_importable(self) -> None:
        """Sub-facades removed from ``__all__`` must still resolve via lazy ``__getattr__``."""
        fc = import_module("flext_core")
        not_resolvable = [
            sym for sym in _DROPPED_BUT_LAZY
            if not hasattr(fc, sym)
        ]
        assert not not_resolvable, (
            f"Lazy resolution broken for: {not_resolvable}"
        )
