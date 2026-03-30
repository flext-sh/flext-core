"""Lazy import utilities for PEP 562 module-level ``__getattr__``.

Provides reusable functions for lazy module loading and submodule
namespace cleanup across the FLEXT ecosystem.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from datetime import date, datetime, time
from enum import Enum
from types import ModuleType
from typing import Protocol

# INFRASTRUCTURE EXCEPTION: lazy_getattr and cleanup_submodule_namespace MUST remain
# module-level functions. They are imported directly by all auto-generated __init__.py
# files for PEP 562 lazy loading. Moving them into a class would break the lazy import
# mechanism across the entire package hierarchy.

type LazyScalar = str | int | float | bool | bytes | date | datetime | time
type LazyCollection = Mapping[str, LazyScalar] | Sequence[LazyScalar]
type LazyCallable = Callable[..., LazyScalar | LazyCollection | None]
type LazyGetattr = Callable[[str], LazyExport]
type LazyDir = Callable[[], list[str]]
type LazyExport = (
    type[BaseException | Enum] | LazyScalar | LazyCollection | ModuleType | LazyCallable
)


class LazyNamespace(Protocol):
    """Mutable namespace for generated module globals."""

    def __setitem__(
        self,
        key: str,
        value: LazyExport | Sequence[str] | LazyGetattr | LazyDir,
        /,
    ) -> None: ...


def _resolve_entry(
    name: str,
    entry: str | Sequence[str],
) -> tuple[str, str]:
    if isinstance(entry, str):
        return (entry, name)
    return (entry[0], entry[1])


def lazy_getattr(
    name: str,
    lazy_imports: Mapping[str, str | Sequence[str]],
    module_globals: LazyNamespace,
    module_name: str,
) -> LazyExport:
    """Lazy-load a module attribute on first access (PEP 562)."""
    if name not in lazy_imports:
        msg = f"module {module_name!r} has no attribute {name!r}"
        raise AttributeError(msg)

    module_path, attr_name = _resolve_entry(name, lazy_imports[name])
    module = importlib.import_module(module_path)
    if not attr_name:
        module_globals[name] = module
        return module

    try:
        value: LazyExport = getattr(module, attr_name)
    except AttributeError:
        msg = f"module {module_path!r} has no attribute {attr_name!r}"
        raise AttributeError(msg) from None
    module_globals[name] = value
    return value


def cleanup_submodule_namespace(
    module_name: str,
    lazy_imports: Mapping[str, str | Sequence[str]],
) -> None:
    """Remove submodules from namespace to force ``__getattr__`` usage."""
    current_module = sys.modules.get(module_name)
    if current_module is None:
        return
    submodule_names: set[str] = set()
    current_parts = module_name.split(".")
    for entry in lazy_imports.values():
        mod_path = entry if isinstance(entry, str) else entry[0]
        if not mod_path:
            continue
        parts = mod_path.split(".")
        if (
            len(parts) > len(current_parts)
            and parts[: len(current_parts)] == current_parts
        ):
            submodule_names.add(parts[len(current_parts)])
    module_dict = vars(current_module)
    for sub_name in submodule_names:
        attr = module_dict.get(sub_name)
        if attr is not None and isinstance(attr, ModuleType):
            delattr(current_module, sub_name)


def install_lazy_exports(
    module_name: str,
    module_globals: LazyNamespace,
    lazy_imports: Mapping[str, str | Sequence[str]],
    all_exports: Sequence[str],
) -> None:
    """Install PEP 562 lazy loading into a module's namespace."""
    resolved: MutableMapping[str, LazyExport] = {}

    def _getattr(name: str) -> LazyExport:
        if name in resolved:
            return resolved[name]
        if name not in lazy_imports:
            msg = f"module {module_name!r} has no attribute {name!r}"
            raise AttributeError(msg)
        value = lazy_getattr(name, lazy_imports, module_globals, module_name)
        resolved[name] = value
        return value

    def _dir() -> list[str]:
        return sorted(all_exports)

    module_globals["__getattr__"] = _getattr
    module_globals["__dir__"] = _dir
    module_globals["__all__"] = list(all_exports)
    cleanup_submodule_namespace(module_name, lazy_imports)


__all__ = ["cleanup_submodule_namespace", "install_lazy_exports", "lazy_getattr"]
