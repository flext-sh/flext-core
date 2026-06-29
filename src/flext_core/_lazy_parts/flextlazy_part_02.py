"""PEP 562 lazy export helpers."""

from __future__ import annotations

import sys
from collections.abc import (
    Sequence,
)
from types import ModuleType

from flext_core._typings.lazy import FlextTypesLazy

from .flextlazy_part_01 import (
    FlextLazy as FlextLazyPart01,
    LazyImportDict,
    LazyImportMap,
    MutableLazyImportMap,
)

type ModuleGlobalValue = FlextTypesLazy.ModuleGlobalValue
type ModuleGlobals = FlextTypesLazy.ModuleGlobals


class FlextLazy(FlextLazyPart01):
    def get(
        self,
        name: str,
        lazy_imports: LazyImportMap,
        module_globals: ModuleGlobals,
        module_name: str,
    ) -> ModuleGlobalValue:
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
            module_globals[name] = mod
            return mod

        try:
            value: ModuleGlobalValue = getattr(mod, attr)
        except AttributeError:
            if isinstance(entry, str) and module_path.rsplit(".", 1)[-1] == name:
                module_globals[name] = mod
                return mod
            msg = f"module {module_path!r} has no attribute {attr!r}"
            raise AttributeError(msg) from None

        module_globals[name] = value
        return value

    def cleanup(
        self,
        module_name: str,
        lazy_imports: LazyImportMap,
    ) -> None:
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
        module_globals: ModuleGlobals,
        lazy_imports: LazyImportMap,
        all_exports: Sequence[str] | None = None,
        *,
        publish_all: bool = True,
    ) -> None:
        """Install __getattr__/__dir__/__all__."""
        pre_signature: tuple[int, int, int, bool] = (
            id(module_globals),
            id(lazy_imports),
            0 if all_exports is None else id(all_exports),
            publish_all,
        )
        if self.install_cache.get(module_name) == pre_signature:
            return

        normalized = self._norm_map(module_name, lazy_imports)
        names = (
            tuple(normalized)
            if all_exports is None
            else tuple(dict.fromkeys((*normalized, *all_exports)))
        )

        module_globals["__getattr__"] = lambda name: self.get(
            name,
            normalized,
            module_globals,
            module_name,
        )
        module_globals["__dir__"] = lambda: list(names)
        if publish_all:
            module_globals["__all__"] = names

        self.cleanup(module_name, normalized)
        self.install_cache[module_name] = pre_signature


__all__: list[str] = ["FlextLazy"]
