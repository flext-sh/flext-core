"""Lazy import utilities for PEP 562 module-level __getattr__.

Provides reusable functions for lazy module loading and submodule
namespace cleanup across the FLEXT ecosystem.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import importlib
import sys
from typing import Any


def lazy_getattr(
    name: str,
    lazy_imports: dict[str, tuple[str, str]],
    module_globals: dict[str, Any],
    module_name: str,
) -> Any:  # noqa: ANN401
    """Lazy-load a module attribute on first access (PEP 562).

    Args:
        name: The attribute name being accessed.
        lazy_imports: Mapping of export_name -> (module_path, attr_name).
        module_globals: The calling module's globals() dict for caching.
        module_name: The calling module's __name__ for error messages.

    """
    if name in lazy_imports:
        module_path, attr_name = lazy_imports[name]
        module = importlib.import_module(module_path)
        value = getattr(module, attr_name)
        module_globals[name] = value
        return value
    msg = f"module {module_name!r} has no attribute {name!r}"
    raise AttributeError(msg)


def cleanup_submodule_namespace(
    module_name: str,
    lazy_imports: dict[str, tuple[str, str]],
) -> None:
    """Remove submodules from namespace to force __getattr__ usage.

    When submodules are imported, Python adds them to the parent module's
    namespace. This removes them so attribute access goes through __getattr__.

    Args:
        module_name: The __name__ of the module to clean up.
        lazy_imports: The _LAZY_IMPORTS dict to derive submodule names from.

    """
    current_module = sys.modules.get(module_name)
    if current_module is None:
        return

    submodule_names: set[str] = set()
    current_parts = module_name.split(".")
    for mod_path, _ in lazy_imports.values():
        if mod_path:
            parts = mod_path.split(".")
            if (
                len(parts) > len(current_parts)
                and parts[: len(current_parts)] == current_parts
            ):
                submodule_names.add(parts[len(current_parts)])

    for sub_name in submodule_names:
        attr = getattr(current_module, sub_name, None)
        if attr is not None and isinstance(attr, type(sys)):
            delattr(current_module, sub_name)
