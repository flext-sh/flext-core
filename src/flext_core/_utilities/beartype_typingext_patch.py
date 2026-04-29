"""Monkey-patch beartype to accept ``typing_extensions.TypeAliasType``.

Pydantic (and other libraries) create PEP-695-style type aliases via
``typing_extensions.TypeAliasType``.  beartype 0.23 only recognises the
stdlib ``typing.TypeAliasType`` (Python ≥ 3.12), so hints such as
``pydantic.JsonValue`` trigger an assertion failure during hint reduction.

This module patches ``beartype._cave._cavefast.HintPep695TypeAlias`` once
at import time.  The patch is idempotent: subsequent imports are a no-op.
"""

from __future__ import annotations

import importlib
import sys as _sys
import typing as _typing
from typing import ClassVar

import typing_extensions as _typing_extensions


class FlextUtilitiesBeartypeTypingExtPatch:
    """Idempotent monkey-patch adding ``typing_extensions.TypeAliasType`` support."""

    _applied: ClassVar[bool] = False

    @classmethod
    def apply(cls) -> None:
        """Extend ``HintPep695TypeAlias`` with ``typing_extensions.TypeAliasType``."""
        if cls._applied:
            return

        cf = importlib.import_module("beartype._cave._cavefast")

        # Already patched or not applicable on this interpreter
        if not hasattr(cf, "HintPep695TypeAlias"):
            cls._applied = True
            return

        current = cf.HintPep695TypeAlias
        new_value = current
        te = _typing_extensions.TypeAliasType
        std = _typing.TypeAliasType

        if isinstance(current, tuple):
            if te not in current:
                new_value = (*current, te)
        elif current is std:
            if len({std, te}) > 1:
                new_value = (std, te)
        elif current is not te:
            new_value = (current, te)

        if new_value is not current:
            # Use setattr to avoid pyrefly strict type-check on external attr.
            setattr(cf, "HintPep695TypeAlias", new_value)
            # Modules that did ``from _cavefast import HintPep695TypeAlias`` hold
            # a stale reference; patch only loaded beartype modules and avoid
            # module-level __getattr__ side effects during import-time patching.
            for mod_name, mod in tuple(_sys.modules.items()):
                if not mod_name.startswith("beartype"):
                    continue
                if mod.__dict__.get("HintPep695TypeAlias") is current:
                    setattr(mod, "HintPep695TypeAlias", new_value)
        cls._applied = True


# Apply immediately so that any subsequent beartype.claw usage sees the fix.
FlextUtilitiesBeartypeTypingExtPatch.apply()

__all__: list[str] = ["FlextUtilitiesBeartypeTypingExtPatch"]
