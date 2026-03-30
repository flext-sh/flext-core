"""Lazy import utilities for PEP 562 module-level ``__getattr__``.

Provides reusable functions for lazy module loading and submodule
namespace cleanup across the FLEXT ecosystem.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable, Mapping, Sequence
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
type LazyImportEntry = str | Sequence[str]
type LazyImportMap = Mapping[str, LazyImportEntry]
type LazyGetattr = Callable[[str], LazyExport]
type LazyDir = Callable[[], list[str]]
type LazyExport = (
    type[BaseException | Enum] | LazyScalar | LazyCollection | ModuleType | LazyCallable
)

_IMPORT_MODULE = importlib.import_module
_MODULE_CACHE: dict[str, ModuleType] = {}
_CHILD_LAZY_IMPORTS_CACHE: dict[str, LazyImportMap] = {}
_CHILD_MERGE_CACHE: dict[tuple[str, ...], dict[str, LazyImportEntry]] = {}


class LazyNamespace(Protocol):
    """Mutable namespace for generated module globals."""

    def __setitem__(
        self,
        key: str,
        value: LazyExport | Sequence[str] | LazyGetattr | LazyDir,
        /,
    ) -> None: ...


def _load_module(module_path: str) -> ModuleType:
    """Load a module using a small fast-path cache before importlib."""
    cached_module = _MODULE_CACHE.get(module_path)
    if cached_module is not None:
        return cached_module

    existing_module = sys.modules.get(module_path)
    if existing_module is not None:
        _MODULE_CACHE[module_path] = existing_module
        return existing_module

    loaded_module = _IMPORT_MODULE(module_path)
    _MODULE_CACHE[module_path] = loaded_module
    return loaded_module


def _derive_export_names(
    lazy_imports: LazyImportMap,
    all_exports: Sequence[str] | None,
) -> tuple[str, ...]:
    """Build export names with a zero-sort fast path for generated modules."""
    lazy_names = tuple(lazy_imports)
    if all_exports is None:
        return lazy_names

    provided_names = tuple(all_exports)
    if not provided_names:
        return lazy_names

    provided_set = set(provided_names)
    if len(provided_names) >= len(lazy_names) and all(
        name in provided_set for name in lazy_names
    ):
        return tuple(dict.fromkeys(provided_names))

    extra_names: list[str] = []
    seen_extras: set[str] = set()
    for name in provided_names:
        if name in lazy_imports or name in seen_extras:
            continue
        seen_extras.add(name)
        extra_names.append(name)
    return lazy_names + tuple(extra_names)


def lazy_getattr(
    name: str,
    lazy_imports: LazyImportMap,
    module_globals: LazyNamespace,
    module_name: str,
) -> LazyExport:
    """Lazy-load a module attribute on first access (PEP 562)."""
    entry = lazy_imports.get(name)
    if entry is None:
        msg = f"module {module_name!r} has no attribute {name!r}"
        raise AttributeError(msg)

    if isinstance(entry, str):
        module_path = entry
        attr_name = name
    else:
        module_path = entry[0]
        attr_name = entry[1]

    module = _load_module(module_path)
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
    lazy_imports: LazyImportMap,
) -> None:
    """Remove submodules from namespace to force ``__getattr__`` usage."""
    current_module = sys.modules.get(module_name)
    if current_module is None:
        return
    module_dict = vars(current_module)
    seen_submodules: set[str] = set()
    module_prefix = f"{module_name}."
    prefix_length = len(module_prefix)

    for entry in lazy_imports.values():
        mod_path = entry if isinstance(entry, str) else entry[0]
        if not mod_path.startswith(module_prefix):
            continue

        sub_name = mod_path[prefix_length:].partition(".")[0]
        if not sub_name or sub_name in seen_submodules:
            continue

        seen_submodules.add(sub_name)
        attr = module_dict.get(sub_name)
        if attr is not None and isinstance(attr, ModuleType):
            delattr(current_module, sub_name)


def _load_child_lazy_imports(module_path: str) -> LazyImportMap:
    """Load a child package lazy map once and cache it by module path."""
    cached_lazy_imports = _CHILD_LAZY_IMPORTS_CACHE.get(module_path)
    if cached_lazy_imports is not None:
        return cached_lazy_imports

    child_module = _load_module(module_path)
    child_lazy_imports = getattr(child_module, "_LAZY_IMPORTS", None)
    if child_lazy_imports is None or not isinstance(child_lazy_imports, Mapping):
        msg = f"module {module_path!r} has no valid _LAZY_IMPORTS mapping"
        raise AttributeError(msg)

    _CHILD_LAZY_IMPORTS_CACHE[module_path] = child_lazy_imports
    return child_lazy_imports


def merge_lazy_imports(
    child_module_paths: Sequence[str],
    local_lazy_imports: LazyImportMap,
) -> dict[str, LazyImportEntry]:
    """Merge child package lazy maps with local entries using cached children."""
    child_paths_key = tuple(child_module_paths)
    cached_children = _CHILD_MERGE_CACHE.get(child_paths_key)
    if cached_children is None:
        cached_children = {}
        for child_module_path in child_paths_key:
            cached_children.update(_load_child_lazy_imports(child_module_path))
        _CHILD_MERGE_CACHE[child_paths_key] = cached_children

    merged_lazy_imports = dict(cached_children)
    merged_lazy_imports.update(local_lazy_imports)
    return merged_lazy_imports


def install_lazy_exports(
    module_name: str,
    module_globals: LazyNamespace,
    lazy_imports: LazyImportMap,
    all_exports: Sequence[str] | None = None,
) -> None:
    """Install PEP 562 lazy loading into a module's namespace.

    When ``all_exports`` is omitted, ``__all__`` and ``__dir__`` are derived
    directly from ``lazy_imports``. Callers that need extra eager names may
    pass only those extras. Legacy callers that still pass the complete export
    list remain compatible.
    """
    export_names = _derive_export_names(lazy_imports, all_exports)

    def _getattr(name: str) -> LazyExport:
        return lazy_getattr(name, lazy_imports, module_globals, module_name)

    def _dir() -> list[str]:
        return list(export_names)

    module_globals["__getattr__"] = _getattr
    module_globals["__dir__"] = _dir
    module_globals["__all__"] = export_names
    cleanup_submodule_namespace(module_name, lazy_imports)


__all__ = (
    "cleanup_submodule_namespace",
    "install_lazy_exports",
    "lazy_getattr",
    "merge_lazy_imports",
)
