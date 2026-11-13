"""PEP 562 lazy export helpers.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from types import ModuleType
from typing import TYPE_CHECKING

from flext_core.flextlazy_bootstrap import FlextLazyBootstrap
from flext_core.flextlazy_member import FlextLazyMember, FlextLazyStandIn

if TYPE_CHECKING:
    from flext_core import t


class FlextLazyExports(FlextLazyBootstrap):
    """Install and resolve exports without generated-package bootstrap imports."""

    def member(self, module: str, owner: str, member: str) -> FlextLazyMember:
        """Return a descriptor deferring ``owner.member`` from ``module``.

        The descriptor loads ``module`` through this owner's ``load``, the
        single dynamic-import path of the lazy machinery.

        Returns:
            A descriptor deferring ``owner.member`` from ``module``.

        """
        return FlextLazyMember(module, owner, member, self.load)

    def deferred(self, module: str, owner: str, members: Sequence[str]) -> type:
        """Return a stand-in for ``module.owner`` whose members load on first use.

        Every member is a ``FlextLazyMember`` descriptor, so composing the
        stand-in by MRO imports nothing. The first member access loads
        ``module`` and caches the real member on the stand-in.

        Returns:
            A class named ``owner`` that defers each of ``members``.

        """
        namespace: dict[
            str, str | FlextLazyMember | tuple[str, str, Callable[[str], ModuleType]]
        ] = {
            "__module__": module,
            "__qualname__": owner,
            "__flext_lazy_target__": (module, owner, self.load),
        }
        namespace.update({name: self.member(module, owner, name) for name in members})
        return FlextLazyStandIn(owner, (), namespace)

    def _deferred_export(
        self,
        name: str,
        lazy_imports: Mapping[str, str | tuple[str, str]],
        members: Sequence[str],
        module_globals: t.ModuleGlobals,
    ) -> type:
        """Publish the stand-in of one deferred export into its package globals.

        Returns:
            The stand-in class, cached so every importer shares its identity.

        """
        entry = lazy_imports[name]
        module_path, attr = (entry, name) if isinstance(entry, str) else entry
        standin = self.deferred(module_path, attr, members)
        module_globals[name] = standin
        return standin

    @staticmethod
    def _require_deferred(
        module_name: str,
        lazy_imports: Mapping[str, str | tuple[str, str]],
        deferred_members: Mapping[str, Sequence[str]],
    ) -> None:
        """Reject a deferred export that names no class export of the package.

        Raises:
            ValueError: If a deferred name is not a class export in the map.

        """
        for name, members in deferred_members.items():
            entry = lazy_imports.get(name)
            if entry is None or (not isinstance(entry, str) and not entry[1]):
                msg = f"module {module_name!r} defers {name!r} without a class export"
                raise ValueError(msg)
            if not members:
                msg = f"module {module_name!r} defers {name!r} with no members"
                raise ValueError(msg)

    @staticmethod
    def resolve_members(target: type) -> tuple[str, ...]:
        """Resolve every lazy member along ``target``'s MRO; return their names.

        Check gates and tests call this so a broken deferred target fails
        exactly as an eager import would.

        Returns:
            The names of the members that were still deferred, in MRO order.

        """
        pending = tuple(
            (name, value)
            for klass in target.__mro__
            for name, value in tuple(vars(klass).items())
            if isinstance(value, FlextLazyMember)
        )
        for _, member in pending:
            member.resolve()
        return tuple(name for name, _ in pending)

    def get(
        self,
        name: str,
        lazy_imports: Mapping[str, str | tuple[str, str]],
        module_globals: t.ModuleGlobals,
        module_name: str,
    ) -> t.ModuleGlobalValue:
        """Resolve one lazy symbol and cache it.

        Returns:
            The resulting ``t.ModuleGlobalValue``.

        Raises:
            AttributeError: If module.
            ImportError: If lazy import of.

        """
        lazy_imports = self._norm_map(module_name, lazy_imports)
        entry = lazy_imports.get(name)
        if entry is None:
            msg = f"module {module_name!r} has no attribute {name!r}"
            raise AttributeError(msg)

        module_path, attr = (entry, name) if isinstance(entry, str) else entry

        try:
            mod = self.load(module_path)
        except AttributeError as exc:
            # CPython's from-import swallows an AttributeError escaping a module
            # __getattr__ together with its cause; a target module that fails
            # to execute is a defect, never a missing name, so it fails loud.
            msg = (
                f"lazy import of {module_path!r} for {name!r} in {module_name!r} failed"
            )
            raise ImportError(msg, name=module_path) from exc
        self.cleanup(module_name, lazy_imports)
        if not attr:
            if not self._module_is_initializing(mod):
                module_globals[name] = mod
            return mod

        try:
            value: t.ModuleGlobalValue = getattr(mod, attr)
        except AttributeError as exc:
            if isinstance(entry, str) and module_path.rsplit(".", 1)[-1] == name:
                if not self._module_is_initializing(mod):
                    module_globals[name] = mod
                return mod
            reason = (
                "is still initializing (circular lazy import) and lacks"
                if self._module_is_initializing(mod)
                else "has no attribute"
            )
            msg = f"module {module_path!r} {reason} {attr!r}"
            raise AttributeError(msg) from exc

        if not self._module_is_initializing(mod):
            module_globals[name] = value
        return value

    @staticmethod
    def cleanup(
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
            if path.startswith("."):
                path = f"{module_name}{path}"
            if not path.startswith(prefix):
                continue
            sub = path[len(prefix) :].partition(".")[0]
            if sub and sub not in seen and isinstance(mod_dict.get(sub), ModuleType):
                declared = lazy_imports.get(sub)
                declared_path = (
                    (declared if isinstance(declared, str) else declared[0])
                    if declared is not None
                    else None
                )
                if declared_path is not None and declared_path.startswith("."):
                    declared_path = f"{module_name}{declared_path}"
                if declared_path != f"{prefix}{sub}":
                    seen.add(sub)
                    mod_dict.pop(sub, None)

    def merge(
        self,
        child_module_paths: Sequence[str],
        local_lazy_imports: Mapping[str, str | tuple[str, str]],
        *,
        exclude_names: Sequence[str] = (),
        module_name: str | None = None,
    ) -> MutableMapping[str, str | tuple[str, str]]:
        """Merge child lazy maps with local entries.

        Returns:
            The resulting ``MutableLazyImportMap``.

        """
        key = tuple(self._child_path(path, module_name) for path in child_module_paths)
        children = self.child_merge_cache.get(key)
        if children is None:
            new_children: MutableMapping[str, str | tuple[str, str]] = {}
            for path in key:
                for name, entry in self._child_map(path).items():
                    if name not in new_children or name.lower() != name:
                        new_children[name] = entry
            self.child_merge_cache[key] = new_children
            children = new_children

        merged = dict(children)
        merged.update(local_lazy_imports)
        for name in exclude_names:
            merged.pop(name, None)
        return merged

    def install(
        self,
        module_name: str,
        module_globals: t.ModuleGlobals,
        lazy_imports: Mapping[str, str | tuple[str, str]],
        *,
        public_exports: Sequence[str] | None = None,
        deferred_members: Mapping[str, Sequence[str]] | None = None,
    ) -> t.LazyExportHooks:
        """Install __getattr__/__dir__/__all__ and publish _LAZY_IMPORTS.

        The normalized map is published into the module globals as
        ``_LAZY_IMPORTS`` because it is the runtime metadata contract:
        ``_child_map`` (parent merge) and the beartype helpers
        (``lazy_alias_suffixes``/``runtime_alias_names``) read it from
        ``vars(module)``. Publishing here makes every install shape —
        module-level literal or inline call — satisfy that contract from
        the single owner.

        ``deferred_members`` maps a class export to its member names. That
        export resolves to its ``deferred`` stand-in without loading its
        module, so a facade composing it loads no heavy capability.

        Returns:
            The installed PEP 562 hooks, reused for identical installation inputs.

        """
        pre_signature: tuple[int, int, int, int] = (
            id(module_globals),
            id(lazy_imports),
            0 if public_exports is None else id(public_exports),
            0 if deferred_members is None else id(deferred_members),
        )
        cached = self.install_cache.get(module_name)
        if cached is not None and cached[0] == pre_signature:
            return cached[1]

        normalized = self._norm_map(module_name, lazy_imports)
        deferred = {} if deferred_members is None else deferred_members
        self._require_deferred(module_name, normalized, deferred)
        for name in normalized:
            module_globals.pop(name, None)
        if public_exports is not None:
            names = tuple(dict.fromkeys(public_exports))
        else:
            names = tuple(normalized)

        module_globals["_LAZY_IMPORTS"] = normalized

        def _module_getattr(name: str) -> t.ModuleGlobalValue:
            members = deferred.get(name)
            if members is not None:
                return self._deferred_export(name, normalized, members, module_globals)
            return self.get(name, normalized, module_globals, module_name)

        def _module_dir() -> list[str]:
            return list(names)

        module_globals["__getattr__"] = _module_getattr
        # Synthetic consumer namespaces (plain dicts that never enter
        # sys.modules) install the dict-level ``__getattr__`` above; the
        # module-object publication only applies to registered modules.
        target = sys.modules.get(module_name)
        if target is not None:
            vars(target)["__getattr__"] = _module_getattr
        module_globals["__dir__"] = _module_dir
        module_globals["__all__"] = names

        self.cleanup(module_name, normalized)
        hooks = (_module_getattr, _module_dir)
        self.install_cache[module_name] = (pre_signature, hooks)
        return hooks


__all__: list[str] = ["FlextLazyExports"]
