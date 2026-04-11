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
from datetime import date, time
from enum import Enum
from types import ModuleType
from typing import Protocol, runtime_checkable

from pydantic import TypeAdapter, ValidationError

# INFRASTRUCTURE EXCEPTION: lazy_getattr and cleanup_submodule_namespace MUST remain
# module-level functions. They are imported directly by all auto-generated __init__.py
# files for PEP 562 lazy loading. Moving them into a class would break the lazy import
# mechanism across the entire package hierarchy.

type LazyImportEntry = str | Sequence[str]
type LazyImportMap = Mapping[str, LazyImportEntry]
type LazyImportModuleGroups = Mapping[str, Sequence[str]]
type LazyImportAliasGroups = Mapping[str, Sequence[tuple[str, str]]]
type LazyScalar = str | int | float | bool | bytes | date | time
type LazyCollection = Mapping[str, LazyScalar] | Sequence[LazyScalar]
type LazyModuleExportValue = LazyScalar
type LazyModuleExport = (
    LazyModuleExportValue
    | LazyCollection
    | ModuleType
    | type[BaseException | Enum]
    | Callable[..., LazyModuleExportValue | LazyCollection | None]
)
type LazyGetattr = Callable[[str], LazyModuleExport]
type LazyDir = Callable[[], list[str]]
type LazyNamespaceValue = LazyModuleExport | Sequence[str] | LazyGetattr | LazyDir

_IMPORT_MODULE = importlib.import_module
_MODULE_CACHE: dict[str, ModuleType] = {}
_CHILD_LAZY_IMPORTS_CACHE: dict[str, LazyImportMap] = {}
_CHILD_MERGE_CACHE: dict[tuple[str, ...], dict[str, LazyImportEntry]] = {}
_LAZY_IMPORT_MAP_ADAPTER = TypeAdapter(dict[str, LazyImportEntry])
_ERR_LAZY_RELATIVE_PATH_REQUIRES_MODULE = (
    "relative lazy-import paths require a parent module name"
)


@runtime_checkable
class LazyNamespace(Protocol):
    """Mutable namespace for generated module globals."""

    def __setitem__(
        self,
        key: str,
        value: LazyNamespaceValue,
        /,
    ) -> None: ...


def build_lazy_import_map(
    module_groups: LazyImportModuleGroups | None = None,
    *,
    alias_groups: LazyImportAliasGroups | None = None,
    sort_keys: bool = True,
) -> dict[str, LazyImportEntry]:
    """Build a flat lazy-import mapping from compact grouped declarations.

    Args:
        module_groups: Mapping of module paths to export names. Each export is mapped
            to the module path directly (same-name import).
        alias_groups: Mapping of module paths to (export_name, attr_name) pairs. Each
            export is mapped to a (module_path, attr_name) tuple for renamed exports.
        sort_keys: Whether to sort the resulting mapping keys for deterministic export
            ordering.

    Returns:
        A dict suitable for ``install_lazy_exports``.

    """
    out: dict[str, LazyImportEntry] = {}
    for module_path, export_names in (module_groups or {}).items():
        for export_name in export_names:
            out[export_name] = module_path
    for module_path, pairs in (alias_groups or {}).items():
        for export_name, attr_name in pairs:
            out[export_name] = (module_path, attr_name)
    if not sort_keys:
        return out
    return {name: out[name] for name in sorted(out)}


def _validate_lazy_import_map(
    raw_lazy_imports: LazyImportMap | Mapping[str, LazyImportEntry] | None,
    module_path: str,
) -> dict[str, LazyImportEntry]:
    """Validate and normalize a child module _LAZY_IMPORTS mapping."""
    try:
        return _LAZY_IMPORT_MAP_ADAPTER.validate_python(raw_lazy_imports)
    except ValidationError as exc:
        msg = f"module {module_path!r} has no valid _LAZY_IMPORTS mapping"
        raise TypeError(msg) from exc


def _normalize_lazy_import_entry(
    module_path: str,
    entry: LazyImportEntry,
) -> LazyImportEntry:
    """Resolve relative lazy-import targets against the current module."""
    if isinstance(entry, str):
        return f"{module_path}{entry}" if entry.startswith(".") else entry
    target_module, attr_name = entry
    resolved_module = (
        f"{module_path}{target_module}"
        if target_module.startswith(".")
        else target_module
    )
    return (resolved_module, attr_name)


def _normalize_lazy_import_map(
    module_path: str,
    raw_lazy_imports: LazyImportMap | Mapping[str, LazyImportEntry] | None,
) -> dict[str, LazyImportEntry]:
    """Validate a lazy-import mapping and resolve relative module targets."""
    validated_lazy_imports = _validate_lazy_import_map(raw_lazy_imports, module_path)
    return {
        name: _normalize_lazy_import_entry(module_path, entry)
        for name, entry in validated_lazy_imports.items()
    }


def _normalize_child_module_path(
    module_name: str | None,
    child_module_path: str,
) -> str:
    """Resolve a relative child package path against the parent module."""
    if child_module_path.startswith("."):
        if not module_name:
            raise ValueError(_ERR_LAZY_RELATIVE_PATH_REQUIRES_MODULE)
        return f"{module_name}{child_module_path}"
    return child_module_path


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
) -> LazyModuleExport:
    """Lazy-load a module attribute on first access (PEP 562)."""
    entry = lazy_imports.get(name)
    if entry is None:
        msg = f"module {module_name!r} has no attribute {name!r}"
        raise AttributeError(msg)

    if isinstance(entry, str):
        direct_child_module_path = f"{module_name}.{name}"
        if direct_child_module_path != entry:
            try:
                direct_child_module = _load_module(direct_child_module_path)
            except ModuleNotFoundError:
                direct_child_module = None
            if direct_child_module is not None:
                module_globals[name] = direct_child_module
                return direct_child_module
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
        value: LazyModuleExport = getattr(module, attr_name)
    except AttributeError:
        module_basename = module_path.rsplit(".", maxsplit=1)[-1]
        if isinstance(entry, str) and module_basename == name:
            module_globals[name] = module
            return module
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
    module_dict = vars(child_module)
    child_lazy_imports = module_dict.get("_LAZY_IMPORTS")
    validated_lazy_imports = _normalize_lazy_import_map(
        child_module.__name__,
        child_lazy_imports,
    )

    _CHILD_LAZY_IMPORTS_CACHE[module_path] = validated_lazy_imports
    return validated_lazy_imports


def merge_lazy_imports(
    child_module_paths: Sequence[str],
    local_lazy_imports: LazyImportMap,
    *,
    exclude_names: Sequence[str] = (),
    module_name: str | None = None,
) -> dict[str, LazyImportEntry]:
    """Merge child package lazy maps with local entries using cached children.

    Lowercase/module-style names keep first-child precedence to avoid submodule
    shadowing. CamelCase/public export names keep last-child precedence so later
    packages can intentionally override earlier class exports. Local module
    entries still override merged child entries explicitly.
    """
    child_paths_key = tuple(
        _normalize_child_module_path(module_name, child_module_path)
        for child_module_path in child_module_paths
    )
    cached_children: dict[str, LazyImportEntry] | None = _CHILD_MERGE_CACHE.get(
        child_paths_key,
    )
    if cached_children is None:
        cached_children = {}
        for child_module_path in child_paths_key:
            for name, entry in _load_child_lazy_imports(child_module_path).items():
                if name in cached_children and name.lower() == name:
                    continue
                cached_children[name] = entry
        _CHILD_MERGE_CACHE[child_paths_key] = cached_children

    merged_lazy_imports = dict(cached_children)
    merged_lazy_imports.update(local_lazy_imports)
    for name in exclude_names:
        merged_lazy_imports.pop(name, None)
    return merged_lazy_imports


def install_lazy_exports(
    module_name: str,
    module_globals: LazyNamespace,
    lazy_imports: LazyImportMap,
    all_exports: Sequence[str] | None = None,
    *,
    publish_all: bool = True,
) -> None:
    """Install PEP 562 lazy loading into a module's namespace.

    When ``all_exports`` is omitted, ``__dir__`` is derived directly from
    ``lazy_imports``. By default ``__all__`` mirrors the same export set, but
    callers may disable that publication for non-root package ``__init__.py``
    files with ``publish_all=False``.

    beartype.claw activation is intentionally not installed here. Recent
    validation shows synthetic packages with Pydantic models and
    runtime-checkable Protocols now work, but flext-core still breaks on
    project-specific PEP 695 aliases in ``flext_core._typings.services`` and
    lazy-layer Protocol hints such as ``LazyNamespace``.
    """
    lazy_imports = _normalize_lazy_import_map(module_name, lazy_imports)
    export_names = _derive_export_names(lazy_imports, all_exports)

    def _getattr(name: str) -> LazyModuleExport:
        return lazy_getattr(name, lazy_imports, module_globals, module_name)

    def _dir() -> list[str]:
        return list(export_names)

    module_globals["__getattr__"] = _getattr
    module_globals["__dir__"] = _dir
    if publish_all:
        module_globals["__all__"] = export_names
    cleanup_submodule_namespace(module_name, lazy_imports)


__all__ = (
    "build_lazy_import_map",
    "cleanup_submodule_namespace",
    "install_lazy_exports",
    "lazy_getattr",
    "merge_lazy_imports",
)
