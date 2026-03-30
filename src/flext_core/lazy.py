"""Lazy import utilities for PEP 562 module-level __getattr__.

Provides reusable functions for lazy module loading and submodule
namespace cleanup across the FLEXT ecosystem.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Mapping, MutableMapping, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flext_core import FlextTypesServices

# INFRASTRUCTURE EXCEPTION: lazy_getattr and cleanup_submodule_namespace MUST remain
# module-level functions. They are imported directly by all auto-generated __init__.py
# files for PEP 562 lazy loading. Moving them into a class would break the lazy import
# mechanism across the entire package hierarchy.


def lazy_getattr(
    name: str,
    lazy_imports: Mapping[str, str | Sequence[str]],
    module_globals: MutableMapping[str, FlextTypesServices.ModuleExport],
    module_name: str,
) -> FlextTypesServices.ModuleExport:
    """Lazy-load a module attribute on first access (PEP 562).

    Args:
        name: The attribute name being accessed.
        lazy_imports: Mapping of export_name to either:
            - ``"module.path"`` — attr_name defaults to the key name
            - ``["module.path", "attr_name"]`` — explicit attr (or empty for submodule)
        module_globals: The calling module's globals() dict for caching.
        module_name: The calling module's __name__ for error messages.

    """
    if name in lazy_imports:
        entry = lazy_imports[name]
        if isinstance(entry, str):
            module_path, attr_name = entry, name
        else:
            module_path, attr_name = entry[0], entry[1]
        module = importlib.import_module(module_path)
        if not attr_name:
            module_globals[name] = module
            return module
        try:
            value = getattr(module, attr_name)
        except AttributeError:
            msg = f"module {module_path!r} has no attribute {attr_name!r}"
            raise AttributeError(msg) from None
        module_globals[name] = value
        return value
    msg = f"module {module_name!r} has no attribute {name!r}"
    raise AttributeError(msg)


def cleanup_submodule_namespace(
    module_name: str,
    lazy_imports: Mapping[str, str | Sequence[str]],
) -> None:
    """Remove submodules from namespace to force __getattr__ usage.

    When submodules are imported, Python adds them to the parent module's
    namespace. This removes them so attribute access goes through __getattr__.
    Supports unlimited hierarchy depth for nested submodules.

    Args:
        module_name: The __name__ of the module to clean up.
        lazy_imports: The _LAZY_IMPORTS dict to derive submodule names from.

    """
    current_module = sys.modules.get(module_name)
    if current_module is None:
        return
    submodule_names: set[str] = set()
    current_parts = module_name.split(".")
    for entry in lazy_imports.values():
        mod_path = entry if isinstance(entry, str) else entry[0]
        if mod_path:
            parts = mod_path.split(".")
            if (
                len(parts) > len(current_parts)
                and parts[: len(current_parts)] == current_parts
            ):
                submodule_names.add(parts[len(current_parts)])
    module_dict = vars(current_module)
    for sub_name in submodule_names:
        attr = module_dict.get(sub_name)
        if attr is not None and isinstance(attr, type(sys)):
            delattr(current_module, sub_name)


def install_lazy_exports(
    module_name: str,
    module_globals: dict[str, object],
    lazy_imports: Mapping[str, str | Sequence[str]],
    all_exports: Sequence[str],
) -> None:
    """Install PEP 562 lazy loading into a module's namespace.

    Sets ``__getattr__``, ``__dir__``, ``__all__`` and cleans up submodules
    in a single call. Replaces the ~30-line boilerplate that was previously
    inlined into every auto-generated ``__init__.py``.

    Args:
        module_name: The ``__name__`` of the calling module.
        module_globals: The calling module's ``globals()`` dict.
        lazy_imports: ``_LAZY_IMPORTS`` mapping of export → (module, attr).
        all_exports: List of public export names for ``__all__``.

    """
    resolved: MutableMapping[str, FlextTypesServices.ModuleExport] = {}
    typed_globals: MutableMapping[str, FlextTypesServices.ModuleExport] = module_globals  # type: ignore[assignment]

    def _getattr(name: str) -> FlextTypesServices.ModuleExport:
        if name in resolved:
            return resolved[name]
        value = lazy_getattr(name, lazy_imports, typed_globals, module_name)
        resolved[name] = value
        return value

    def _dir() -> list[str]:
        return sorted(all_exports)

    module_globals["__getattr__"] = _getattr
    module_globals["__dir__"] = _dir
    module_globals["__all__"] = list(all_exports)
    cleanup_submodule_namespace(module_name, lazy_imports)


__all__ = ["cleanup_submodule_namespace", "install_lazy_exports", "lazy_getattr"]
