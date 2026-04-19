"""PEP 562 lazy export helpers."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from types import ModuleType

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    TypeAdapter,
    ValidationError,
    computed_field,
)


class FlextLazy(BaseModel):
    """Canonical lazy API as a container with runtime reuse caches."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    err_relative_path_requires_module: str = (
        "relative lazy-import paths require a parent module name"
    )

    module_cache: dict[str, ModuleType] = Field(default_factory=dict)
    child_lazy_cache: dict[str, dict[str, str | tuple[str, str]]] = Field(
        default_factory=dict
    )
    child_merge_cache: dict[tuple[str, ...], dict[str, str | tuple[str, str]]] = Field(
        default_factory=dict,
    )
    normalized_map_cache: dict[tuple[str, int], dict[str, str | tuple[str, str]]] = (
        Field(
            default_factory=dict,
        )
    )
    install_cache: dict[str, tuple[int, int, int, bool]] = Field(default_factory=dict)

    _import_module: Callable[[str], ModuleType] = PrivateAttr(
        default_factory=lambda: importlib.import_module,
    )
    _map_adapter: TypeAdapter[dict[str, str | tuple[str, str]]] = PrivateAttr(
        default_factory=lambda: TypeAdapter(dict[str, str | tuple[str, str]]),
    )
    _alias_adapter: TypeAdapter[tuple[str, str]] = PrivateAttr(
        default_factory=lambda: TypeAdapter(tuple[str, str]),
    )

    @computed_field(return_type=dict[str, int])
    @property
    def cache_stats(self) -> dict[str, int]:
        """Expose cache sizes for diagnostics/observability."""
        return {
            "module_cache": len(self.module_cache),
            "child_lazy_cache": len(self.child_lazy_cache),
            "child_merge_cache": len(self.child_merge_cache),
            "normalized_map_cache": len(self.normalized_map_cache),
            "install_cache": len(self.install_cache),
        }

    def _norm_cache_key(
        self,
        module_path: str,
        raw: Mapping[str, str | tuple[str, str]] | None,
    ) -> tuple[str, int]:
        return (module_path, id(raw))

    def _norm_map(
        self,
        module_path: str,
        raw: Mapping[str, str | tuple[str, str]] | None,
    ) -> dict[str, str | tuple[str, str]]:
        cache_key = self._norm_cache_key(module_path, raw)
        cached = self.normalized_map_cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            validated = self._map_adapter.validate_python(raw)
        except ValidationError as exc:
            msg = f"module {module_path!r} has no valid _LAZY_IMPORTS mapping"
            raise TypeError(msg) from exc

        out: dict[str, str | tuple[str, str]] = {}
        for name, entry in validated.items():
            if isinstance(entry, str):
                out[name] = f"{module_path}{entry}" if entry.startswith(".") else entry
                continue
            target, attr = self._alias_adapter.validate_python(entry)
            resolved = f"{module_path}{target}" if target.startswith(".") else target
            out[name] = (resolved, attr)

        self.normalized_map_cache[cache_key] = out
        return out

    def _load(self, module_path: str) -> ModuleType:
        cached = self.module_cache.get(module_path)
        if cached is not None:
            return cached
        mod = sys.modules.get(module_path) or self._import_module(module_path)
        self.module_cache[module_path] = mod
        return mod

    def _child_map(self, module_path: str) -> dict[str, str | tuple[str, str]]:
        cached = self.child_lazy_cache.get(module_path)
        if cached is not None:
            return cached
        child = self._load(module_path)
        normalized = self._norm_map(child.__name__, vars(child).get("_LAZY_IMPORTS"))
        self.child_lazy_cache[module_path] = normalized
        return normalized

    def _child_path(self, path: str, module_name: str | None) -> str:
        if not path.startswith("."):
            return path
        if module_name:
            return f"{module_name}{path}"
        raise ValueError(self.err_relative_path_requires_module)

    def reset(self) -> None:
        """Reset all lazy caches to a clean state."""
        self.module_cache.clear()
        self.child_lazy_cache.clear()
        self.child_merge_cache.clear()
        self.normalized_map_cache.clear()
        self.install_cache.clear()

    def build_map(
        self,
        module_groups: Mapping[str, Sequence[str]] | None = None,
        *,
        alias_groups: Mapping[str, Sequence[tuple[str, str]]] | None = None,
        sort_keys: bool = True,
    ) -> dict[str, str | tuple[str, str]]:
        """Build one flat lazy-import map."""
        out: dict[str, str | tuple[str, str]] = {
            name: module
            for module, names in (module_groups or {}).items()
            for name in names
        }
        for module, pairs in (alias_groups or {}).items():
            for name, attr in pairs:
                out[name] = (module, attr)
        return {name: out[name] for name in sorted(out)} if sort_keys else out

    def get(
        self,
        name: str,
        lazy_imports: Mapping[str, str | tuple[str, str]],
        module_globals: MutableMapping[str, object],
        module_name: str,
    ) -> object:
        """Resolve one lazy symbol and cache it."""
        entry = lazy_imports.get(name)
        if entry is None:
            msg = f"module {module_name!r} has no attribute {name!r}"
            raise AttributeError(msg)

        module_path, attr = (
            (entry, name)
            if isinstance(entry, str)
            else self._alias_adapter.validate_python(entry)
        )

        if isinstance(entry, str):
            child_path = f"{module_name}.{name}"
            if child_path != entry:
                try:
                    child = self._load(child_path)
                except ModuleNotFoundError:
                    child = None
                if child is not None:
                    module_globals[name] = child
                    return child

        mod = self._load(module_path)
        if not attr:
            module_globals[name] = mod
            return mod

        try:
            value = getattr(mod, attr)
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
        lazy_imports: Mapping[str, str | tuple[str, str]],
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
        local_lazy_imports: Mapping[str, str | tuple[str, str]],
        *,
        exclude_names: Sequence[str] = (),
        module_name: str | None = None,
    ) -> dict[str, str | tuple[str, str]]:
        """Merge child lazy maps with local entries."""
        key = tuple(self._child_path(path, module_name) for path in child_module_paths)
        children = self.child_merge_cache.get(key)
        if children is None:
            children = dict[str, str | tuple[str, str]]()
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
        module_globals: MutableMapping[str, object],
        lazy_imports: Mapping[str, str | tuple[str, str]],
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


lazy = FlextLazy()
"""Shared ``FlextLazy`` singleton used by package-level lazy exports."""
build_lazy_import_map = lazy.build_map
"""Convenience alias for building flat lazy import maps."""
lazy_getattr = lazy.get
cleanup_submodule_namespace = lazy.cleanup
merge_lazy_imports = lazy.merge
install_lazy_exports = lazy.install

__all__ = (
    "FlextLazy",
    "build_lazy_import_map",
    "cleanup_submodule_namespace",
    "install_lazy_exports",
    "lazy",
    "lazy_getattr",
    "merge_lazy_imports",
)
