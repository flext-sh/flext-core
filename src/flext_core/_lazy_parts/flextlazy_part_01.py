"""PEP 562 lazy export helpers."""

from __future__ import annotations

import importlib
import sys
from collections.abc import (
    Callable,
    Mapping,
    Sequence,
)
from typing import TYPE_CHECKING

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    TypeAdapter,
    ValidationError,
    computed_field,
)

if TYPE_CHECKING:
    from types import ModuleType

type StrPair = tuple[str, str]
type LazyImportEntry = str | StrPair
type LazyImportMap = Mapping[str, LazyImportEntry]
type LazyImportDict = dict[str, LazyImportEntry]
type MutableLazyImportMap = dict[str, LazyImportEntry]
type LazyImportAliasGroups = Mapping[str, Sequence[StrPair]]


class FlextLazy(BaseModel):
    """Canonical lazy API as a container with runtime reuse caches."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    err_relative_path_requires_module: str = (
        "relative lazy-import paths require a parent module name"
    )

    module_cache: dict[str, ModuleType] = Field(default_factory=dict)

    child_lazy_cache: dict[str, LazyImportDict] = Field(default_factory=dict)

    child_merge_cache: dict[tuple[str, ...], LazyImportDict] = Field(
        default_factory=dict,
    )

    normalized_map_cache: dict[tuple[str, int], LazyImportDict] = Field(
        default_factory=dict,
    )

    install_cache: dict[str, tuple[int, int, int, int, bool]] = Field(
        default_factory=dict,
    )

    _import_module: Callable[[str], ModuleType] = PrivateAttr(
        default_factory=lambda: importlib.import_module,
    )

    _map_adapter: TypeAdapter[LazyImportDict] = PrivateAttr(
        default_factory=lambda: TypeAdapter(LazyImportDict),
    )

    _alias_adapter: TypeAdapter[StrPair] = PrivateAttr(
        default_factory=lambda: TypeAdapter(StrPair),
    )

    _activate_core_beartype: Callable[[], None] = PrivateAttr(
        default_factory=lambda: (
            importlib.import_module(
                "flext_core._beartype_bootstrap",
            ).FlextCoreBeartypeBootstrap.activate_package_beartype
        ),
    )

    _activating_core_beartype: bool = PrivateAttr(default=False)

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
        raw: LazyImportMap | None,
    ) -> tuple[str, int]:
        return (module_path, id(raw))

    def _norm_map(
        self,
        module_path: str,
        raw: LazyImportMap | None,
    ) -> LazyImportDict:
        cache_key = self._norm_cache_key(module_path, raw)
        cached = self.normalized_map_cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            validated = self._map_adapter.validate_python(raw)
        except ValidationError as exc:
            msg = f"module {module_path!r} has no valid _LAZY_IMPORTS mapping"
            raise TypeError(msg) from exc

        out: LazyImportDict = {}
        for name, entry in validated.items():
            if isinstance(entry, str):
                out[name] = f"{module_path}{entry}" if entry.startswith(".") else entry
                continue
            target, attr = self._alias_adapter.validate_python(entry)
            resolved = f"{module_path}{target}" if target.startswith(".") else target
            out[name] = (resolved, attr)

        self.normalized_map_cache[cache_key] = out
        return out

    def _must_activate_core_beartype(self, module_path: str) -> bool:
        """Return whether importing a module should activate flext_core beartype."""
        root_module = sys.modules.get("flext_core")
        root_ready = root_module is not None and "t" in vars(root_module)
        return (
            root_ready
            and module_path.startswith("flext_core.")
            and not module_path.startswith("flext_core._constants")
        )

    def normalize_map(
        self,
        module_path: str,
        raw: LazyImportMap | None,
    ) -> LazyImportDict:
        """Return normalized lazy-import entries for runtime metadata readers."""
        return self._norm_map(module_path, raw)

    def _load(self, module_path: str) -> ModuleType:
        cached = self.module_cache.get(module_path)
        if cached is not None:
            return cached
        if (
            self._must_activate_core_beartype(module_path)
            and not self._activating_core_beartype
        ):
            self._activating_core_beartype = True
            try:
                self._activate_core_beartype()
            finally:
                self._activating_core_beartype = False
        mod = sys.modules.get(module_path) or self._import_module(module_path)
        self.module_cache[module_path] = mod
        return mod

    def _child_map(self, module_path: str) -> LazyImportDict:
        cached = self.child_lazy_cache.get(module_path)
        if cached is not None:
            return cached
        child = self._load(module_path)
        raw_lazy_imports = vars(child).get("_LAZY_IMPORTS")
        if raw_lazy_imports is None:
            normalized: LazyImportDict = {}
            self.child_lazy_cache[module_path] = normalized
            return normalized
        normalized = self._norm_map(child.__name__, raw_lazy_imports)
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
        alias_groups: LazyImportAliasGroups | None = None,
        sort_keys: bool = True,
    ) -> LazyImportDict:
        """Build one flat lazy-import map."""
        out: LazyImportDict = {
            name: module
            for module, names in (module_groups or {}).items()
            for name in names
        }
        for module, pairs in (alias_groups or {}).items():
            for name, attr in pairs:
                out[name] = (module, attr)
        return {name: out[name] for name in sorted(out)} if sort_keys else out


__all__: list[str] = [
    "FlextLazy",
    "LazyImportAliasGroups",
    "LazyImportDict",
    "LazyImportEntry",
    "LazyImportMap",
    "MutableLazyImportMap",
    "StrPair",
]
