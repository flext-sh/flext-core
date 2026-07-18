"""PEP 562 lazy export helpers."""

from __future__ import annotations

import sys
from types import ModuleType

# NOTE (multi-agent, mro-0ftd.3.3.1): consume the finite structural contract;
# the former self-recursive module-global alias has no runtime owner.

from .flextlazy_part_01 import (
    FlextLazy as FlextLazyPart01,
    LazyImportDict,
    LazyImportMap,
    MutableLazyImportMap,
)

from typing import TYPE_CHECKING

# mro-qtjb (flext-core): p imported at runtime — the lazy __getattr__ wrapper
# runs at import time and beartype resolves its p.Namespace parameter hint in
# this module namespace, so p must be runtime-importable (forward-safe: this
# is u-layer infra importing the p facade).
from flext_core._protocols.lazy import FlextProtocolsLazy as p

if TYPE_CHECKING:
    from collections.abc import Sequence


class FlextLazyAttribute:
    """Descriptor that resolves a class attribute through ``FlextLazy``."""

    __slots__ = ("_lazy", "_lazy_imports", "_module_globals", "_module_name", "_name")

    def __init__(
        self,
        lazy: FlextLazy,
        name: str,
        lazy_imports: LazyImportMap,
        module_globals: p.Namespace,
        module_name: str,
    ) -> None:
        self._lazy = lazy
        self._name = name
        self._lazy_imports = lazy_imports
        self._module_globals = module_globals
        self._module_name = module_name

    def __get__(
        self, instance: p.ResolvedExport | None, owner: type | None = None
    ) -> p.ResolvedExport:
        """Resolve and cache the target symbol through the owning lazy container."""
        _ = instance, owner
        return self._lazy.get(
            self._name, self._lazy_imports, self._module_globals, self._module_name
        )


class FlextLazy(FlextLazyPart01):
    @staticmethod
    def _module_is_initializing(module: ModuleType) -> bool:
        """Return whether Python is still executing a module body."""
        return bool(getattr(getattr(module, "__spec__", None), "_initializing", False))

    def attribute(
        self,
        name: str,
        lazy_imports: LazyImportMap,
        module_globals: p.Namespace,
        module_name: str,
    ) -> FlextLazyAttribute:
        """Return a descriptor for class-namespace lazy attributes."""
        return FlextLazyAttribute(self, name, lazy_imports, module_globals, module_name)

    def get(
        self,
        name: str,
        lazy_imports: LazyImportMap,
        module_globals: p.Namespace,
        module_name: str,
    ) -> p.ResolvedExport:
        """Resolve one lazy symbol and cache it."""
        lazy_imports = self._norm_map(module_name, lazy_imports)
        entry = lazy_imports.get(name)
        if entry is None:
            msg = f"module {module_name!r} has no attribute {name!r}"
            raise AttributeError(msg)

        module_path, attr = (
            (entry, name)
            if isinstance(entry, str)
            else self._alias_adapter.validate_python(entry)
        )

        mod = self._load(module_path)
        if not attr:
            if not self._module_is_initializing(mod):
                module_globals[name] = mod
            return mod

        try:
            value: p.ResolvedExport = getattr(mod, attr)
        except AttributeError:
            if isinstance(entry, str) and module_path.rsplit(".", 1)[-1] == name:
                if not self._module_is_initializing(mod):
                    module_globals[name] = mod
                return mod
            msg = f"module {module_path!r} has no attribute {attr!r}"
            raise AttributeError(msg) from None

        if not self._module_is_initializing(mod):
            module_globals[name] = value
        return value

    def cleanup(self, module_name: str, lazy_imports: LazyImportMap) -> None:
        """Remove eager child module attrs."""
        current = sys.modules.get(module_name)
        if current is None:
            return
        mod_dict, seen, prefix = vars(current), set[str](), f"{module_name}."
        for entry in lazy_imports.values():
            path = entry if isinstance(entry, str) else entry[0]
            if not path.startswith(prefix):
                continue
            sub = path[len(prefix) :].partition(".")[0]
            if sub and sub not in seen and isinstance(mod_dict.get(sub), ModuleType):
                seen.add(sub)
                mod_dict.pop(sub, None)

    def merge(
        self,
        child_module_paths: Sequence[str],
        local_lazy_imports: LazyImportMap,
        *,
        exclude_names: Sequence[str] = (),
        module_name: str | None = None,
    ) -> MutableLazyImportMap:
        """Merge child lazy maps with local entries."""
        key = tuple(self._child_path(path, module_name) for path in child_module_paths)
        children: LazyImportDict | None = self.child_merge_cache.get(key)
        if children is None:
            children = {}
            for path in key:
                for name, entry in self._child_map(path).items():
                    if name not in children or name.lower() != name:
                        children[name] = entry
            self.child_merge_cache[key] = children

        merged = dict(children)
        merged.update(local_lazy_imports)
        for name in exclude_names:
            merged.pop(name, None)
        return merged

    def install(
        self,
        module_name: str,
        module_globals: p.Namespace,
        lazy_imports: LazyImportMap,
        all_exports: Sequence[str] | None = None,
        *,
        publish_all: bool = True,
        public_exports: Sequence[str] | None = None,
    ) -> None:
        """Install __getattr__/__dir__/__all__."""
        pre_signature: tuple[int, int, int, int, bool] = (
            id(module_globals),
            id(lazy_imports),
            0 if all_exports is None else id(all_exports),
            0 if public_exports is None else id(public_exports),
            publish_all,
        )
        if self.install_cache.get(module_name) == pre_signature:
            return

        normalized = self._norm_map(module_name, lazy_imports)
        for name in normalized:
            if name in module_globals:
                del module_globals[name]
        if public_exports is not None:
            names = tuple(dict.fromkeys(public_exports))
        elif all_exports is None:
            names = tuple(normalized)
        else:
            names = tuple(dict.fromkeys((*normalized, *all_exports)))

        module_globals["__getattr__"] = lambda name: self.get(
            name, normalized, module_globals, module_name
        )
        module_globals["__dir__"] = lambda: list(names)
        if publish_all:
            module_globals["__all__"] = names

        self.cleanup(module_name, normalized)
        self.install_cache[module_name] = pre_signature


__all__: list[str] = ["FlextLazy", "FlextLazyAttribute"]
