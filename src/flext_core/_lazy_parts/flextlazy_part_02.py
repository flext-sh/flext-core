"""PEP 562 lazy export helpers.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import TYPE_CHECKING

from .._typings.lazy import FlextTypesLazy
from .flextlazy_part_01 import (
    FlextLazyPart01,
    LazyImportDict,
    LazyImportMap,
    MutableLazyImportMap,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

type ModuleGlobalValue = FlextTypesLazy.ModuleGlobalValue
type ModuleGlobals = FlextTypesLazy.ModuleGlobals


class FlextLazyMember:
    """Class-namespace descriptor that defers one mixin member to first access.

    A generated deferred base declares one descriptor per member of a capability
    mixin. The first access imports the mixin module, takes the member's raw
    value from the mixin's MRO (so a staticmethod or classmethod keeps its
    binding semantics) and replaces the descriptor on its host, making every
    later lookup a plain class attribute. A failure raises with its cause and
    caches nothing.
    """

    __slots__ = ("_host", "_member", "_module", "_owner")

    def __init__(self, module: str, owner: str, member: str) -> None:
        self._module = module
        self._owner = owner
        self._member = member
        self._host: type | None = None

    def __set_name__(self, owner: type, name: str) -> None:
        """Bind the host class; the attribute name must equal the member.

        Raises:
            TypeError: When the attribute name differs from the member name.

        """
        if name != self._member:
            msg = f"lazy member {self._member!r} is bound to attribute {name!r}"
            raise TypeError(msg)
        self._host = owner

    def __get__(
        self,
        instance: ModuleGlobalValue | None,
        owner: type | None = None,
    ) -> ModuleGlobalValue:
        """Resolve, cache on the host, and bind like the original member.

        Returns:
            The member, bound to ``instance``/``owner`` when it is a descriptor.

        """
        raw = self.resolve()
        binder = getattr(type(raw), "__get__", None)
        if binder is None:
            return raw
        bound: ModuleGlobalValue = binder(raw, instance, owner or self._host)
        return bound

    def resolve(self) -> ModuleGlobalValue:
        """Return the member's raw value and replace this descriptor with it.

        Returns:
            The raw value the mixin's MRO declares for the member.

        Raises:
            TypeError: When the descriptor was never bound to a class.
            ImportError: When the mixin cannot load or lacks the member.

        """
        host = self._host
        if host is None:
            msg = f"lazy member {self._member!r} is not bound to a class"
            raise TypeError(msg)
        target = f"{self._module}.{self._owner}"
        try:
            source = getattr(importlib.import_module(self._module), self._owner)
        except AttributeError as exc:
            msg = f"lazy member {self._member!r} cannot load {target!r}"
            raise ImportError(msg, name=self._module) from exc
        for klass in source.__mro__:
            namespace = vars(klass)
            if self._member in namespace:
                raw: ModuleGlobalValue = namespace[self._member]
                type.__setattr__(host, self._member, raw)
                return raw
        msg = f"{target!r} declares no member {self._member!r}"
        raise ImportError(msg, name=self._module)


class FlextLazy(FlextLazyPart01):
    @staticmethod
    def member(module: str, owner: str, member: str) -> FlextLazyMember:
        """Return a descriptor deferring ``owner.member`` from ``module``.

        Returns:
            A descriptor deferring ``owner.member`` from ``module``.

        """
        return FlextLazyMember(module, owner, member)

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
        lazy_imports: LazyImportMap,
        module_globals: ModuleGlobals,
        module_name: str,
    ) -> ModuleGlobalValue:
        """Resolve one lazy symbol and cache it.

        Returns:
            The resulting ``ModuleGlobalValue``.

        Raises:
            AttributeError: If module.
            ImportError: If lazy import of.

        """
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

        try:
            mod = self._load(module_path)
        except AttributeError as exc:
            # CPython's from-import swallows an AttributeError escaping a module
            # __getattr__ together with its cause; a target module that fails
            # to execute is a defect, never a missing name, so it fails loud.
            msg = (
                f"lazy import of {module_path!r} for {name!r} in {module_name!r} failed"
            )
            raise ImportError(msg, name=module_path) from exc
        if not attr:
            if not self._module_is_initializing(mod):
                module_globals[name] = mod
            return mod

        try:
            value: ModuleGlobalValue = getattr(mod, attr)
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
        """Merge child lazy maps with local entries.

        Returns:
            The resulting ``MutableLazyImportMap``.

        """
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
        public_exports: Sequence[str] | None = None,
    ) -> None:
        """Install __getattr__/__dir__/__all__ and publish _LAZY_IMPORTS.

        The normalized map is published into the module globals as
        ``_LAZY_IMPORTS`` because it is the runtime metadata contract:
        ``_child_map`` (parent merge) and the beartype helpers
        (``lazy_alias_suffixes``/``runtime_alias_names``) read it from
        ``vars(module)``. Publishing here makes every install shape —
        module-level literal or inline call — satisfy that contract from
        the single owner.

        Raises:
            RuntimeError: If module.

        """
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
            module_globals.pop(name, None)
        if public_exports is not None:
            names = tuple(dict.fromkeys(public_exports))
        elif all_exports is None:
            names = tuple(normalized)
        else:
            names = tuple(dict.fromkeys((*normalized, *all_exports)))

        module_globals["_LAZY_IMPORTS"] = normalized

        def _module_getattr(name: str) -> ModuleGlobalValue:
            return self.get(name, normalized, module_globals, module_name)

        target = sys.modules.get(module_name)
        if target is None:
            msg = f"module {module_name!r} is not registered in sys.modules"
            raise RuntimeError(msg)
        module_globals["__getattr__"] = _module_getattr
        target.__getattr__ = _module_getattr
        module_globals["__dir__"] = lambda: list(names)
        if publish_all:
            module_globals["__all__"] = names

        self.cleanup(module_name, normalized)
        self.install_cache[module_name] = pre_signature


__all__: list[str] = ["FlextLazy", "FlextLazyMember"]
