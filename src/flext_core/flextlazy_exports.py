"""PEP 562 lazy export helpers.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import builtins
import sys
import types
from collections.abc import (
    Callable,
    Collection,
    Iterator,
    Mapping,
    MutableMapping,
    Sequence,
)
from importlib.machinery import PathFinder
from types import ModuleType
from typing import TYPE_CHECKING, cast

from flext_core.flextlazy_bootstrap import FlextLazyBootstrap
from flext_core.flextlazy_instance import FlextLazyInstance
from flext_core.flextlazy_member import FlextLazyMember

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

    class _PartBuiltins(Mapping[str, "t.ModuleGlobalValue"]):
        """Builtins a PEP 695 class statement runs with, resolved on demand.

        CPython reads a missing global, ``__build_class__`` and ``__import__``
        from a function's builtins, which may be any mapping: while the
        statement runs, a name resolves as a stand-in base does; afterwards
        (a lazily evaluated bound, constraint or default) it is read from the
        loaded part module. Anything else is a real builtin.
        """

        __slots__ = ("resolve",)
        __import__ = staticmethod(builtins.__import__)

        def __init__(self, resolve: Callable[[str], t.ModuleGlobalValue]) -> None:
            """Bind the resolver of the statement's names."""
            self.resolve = resolve

        def __getitem__(self, key: str) -> t.ModuleGlobalValue:
            """Resolve a name the statement or an evaluator reads.

            Returns:
                The resolved value.

            """
            return self.resolve(key)

        def __iter__(self) -> Iterator[str]:
            """Iterate the real builtins.

            Returns:
                An iterator over the builtin names.

            """
            return iter(vars(builtins))

        def __len__(self) -> int:
            """Count the real builtins.

            Returns:
                The number of builtin names.

            """
            return len(vars(builtins))

    def deferred(
        self,
        module: str,
        owner: str,
        members: Sequence[str],
        bases: tuple[t.ModuleGlobalValue, ...] = (),
        type_params: tuple[t.ModuleGlobalValue, ...] = (),
    ) -> type:
        """Return a stand-in for ``module.owner`` whose members load on first use.

        The stand-in is a class of metaclass ``type`` with one
        ``FlextLazyMember`` descriptor per member and no fallback, so it
        composes with any metaclass and composing it by MRO imports nothing;
        a name the part does not bind is a plain ``AttributeError``. The first
        member access loads ``module`` and caches the real member on the
        stand-in. ``bases`` are the real class's bases, as stand-ins or real
        exports, so C3 reproduces the real MRO; a PEP 695 part passes its
        ``Generic`` base and its ``type_params``, so the stand-in is generic
        exactly like the real class.

        Returns:
            A class named ``owner`` that defers each of ``members``.

        """
        namespace: dict[
            str, t.ModuleGlobalValue | FlextLazyMember | tuple[t.ModuleGlobalValue, ...]
        ] = {"__module__": module, "__qualname__": owner}
        if type_params:
            namespace["__type_params__"] = type_params
        namespace.update({name: self.member(module, owner, name) for name in members})
        return types.new_class(
            owner.rpartition(".")[2], bases, {}, lambda body: body.update(namespace)
        )

    def standin(self, module_path: str, qualname: str) -> type:
        """Return the stand-in of class ``qualname`` of ``module_path``.

        Built once per process and kept in memory: its members are the names
        its class body stores and its bases are resolved from its class
        statement and its module's import bindings, all read from the code
        without loading it. A base imported from a package resolves through
        that package's export (its stand-in or its real class); a base defined
        in the same module or imported by module path becomes its own
        stand-in. C3 reproduces the real MRO; nothing is flattened. A PEP 695
        statement runs in its own scope with ``__build_class__`` captured:
        the class body never runs, and the type parameters and bases are the
        ones the real statement creates.

        Returns:
            The stand-in class, shared by every route that reaches it.

        Raises:
            ValueError: If the class binds no member and names no base.
            TypeError: If a base is not a class with metaclass ``type``.

        """
        key = (module_path, qualname)
        cached = self.standins.get(key)
        if cached is not None:
            return cached
        members = self._class_members(module_path, qualname)
        _, scope = self._class_statement(module_path, qualname)
        if scope.co_name.startswith(self.generic_scope_prefix):
            bases, type_params = self._generic_statement(module_path, qualname, scope)
        else:
            bases = tuple(
                self._base_value(module_path, qualname, base)
                for base in self._class_bases(module_path, qualname)
            )
            type_params = ()
        if not members and not bases:
            msg = f"stand-in {qualname!r} has neither members nor bases"
            raise ValueError(msg)
        for base in types.resolve_bases(bases):
            if type(base) is not type:
                msg = (
                    f"class {module_path}.{qualname} has the base "
                    f"{getattr(base, '__qualname__', base)!r}, "
                    "which is not a class with metaclass 'type'"
                )
                raise TypeError(msg)
        standin = self.deferred(module_path, qualname, members, bases, type_params)
        self.standins[key] = standin
        return standin

    def _generic_statement(
        self, module_path: str, qualname: str, scope: types.CodeType
    ) -> tuple[tuple[t.ModuleGlobalValue, ...], tuple[t.ModuleGlobalValue, ...]]:
        """Run a PEP 695 class statement without its body; return its bases.

        The statement's scope runs with its own globals: a base name resolves
        as a stand-in base, and ``__build_class__`` only captures its
        arguments, so the class body never runs. The type parameters it
        creates are the real ones, with their bounds, constraints and
        defaults still evaluated lazily, from the loaded part, on first use.

        Returns:
            The bases (the ``Generic`` alias included) and the type parameters.

        Raises:
            TypeError: If the statement passes keywords (a metaclass), which no
                stand-in represents, or its scope closes over another name.

        """
        loaded = False

        def capture(
            body: t.ModuleGlobalValue,
            _name: str,
            *bases: t.ModuleGlobalValue,
            **keywords: t.ModuleGlobalValue,
        ) -> tuple[t.ModuleGlobalValue, tuple[t.ModuleGlobalValue, ...]]:
            if keywords:
                msg = (
                    f"class {module_path}.{qualname} passes {sorted(keywords)}; "
                    "no stand-in represents a class keyword"
                )
                raise TypeError(msg)
            return body, bases

        def resolve(name: str) -> t.ModuleGlobalValue:
            if name == "__build_class__" and not loaded:
                return cast("t.ModuleGlobalValue", capture)
            if loaded:
                found = vars(self.load(module_path)).get(name)
                if found is not None:
                    return cast("t.ModuleGlobalValue", found)
            else:
                try:
                    return self._base_value(module_path, qualname, (name,))
                except NameError:
                    pass
            if name in vars(builtins):
                return cast("t.ModuleGlobalValue", vars(builtins)[name])
            raise KeyError(name)

        if scope.co_freevars:
            msg = (
                f"class {module_path}.{qualname} is a PEP 695 class closing over "
                f"{scope.co_freevars}, which only its enclosing scope binds"
            )
            raise TypeError(msg)
        namespace = {
            "__name__": module_path,
            "__builtins__": self._PartBuiltins(resolve),
        }
        body, bases = cast(
            "tuple[types.FunctionType, tuple[t.ModuleGlobalValue, ...]]",
            types.FunctionType(scope, namespace)(),
        )
        loaded = True
        cells = dict(
            zip(body.__code__.co_freevars, body.__closure__ or (), strict=True)
        )
        type_params = cast(
            "tuple[t.ModuleGlobalValue, ...]", cells[".type_params"].cell_contents
        )
        return bases, type_params

    def _base_value(
        self, module_path: str, qualname: str, base: tuple[str, ...]
    ) -> t.ModuleGlobalValue:
        """Resolve one base expression of a class statement to its value.

        Returns:
            The base: a stand-in, a package's export, or a builtin.

        Raises:
            NameError: If the module binds the base name by no class
                statement, no ``from`` import and no builtin.

        """
        head, *path = base
        scope = qualname.rpartition(".")[0]
        classes = self._module_classes(module_path)
        local = next(
            (
                name
                for name in (f"{scope}.{head}" if scope else "", head)
                if name in classes
            ),
            None,
        )
        value: t.ModuleGlobalValue
        route = None if local is not None else self._import_route(module_path, head)
        if local is not None:
            value = self.standin(module_path, ".".join((local, *path)))
            path.clear()
        elif route is None:
            value = getattr(builtins, head, None)
            if value is None:
                msg = f"class {module_path}.{qualname} names an unbound base {head}"
                raise NameError(msg)
        elif self._is_package(route[0]):
            value = getattr(self.load(route[0]), route[1])
        else:
            value = self.standin(route[0], ".".join((route[1], *path)))
            path.clear()
        for part in path:
            value = getattr(value, part)
        return value

    def _is_package(self, module_path: str) -> bool:
        """Return whether ``module_path`` names a package, without loading it.

        Returns:
            ``True`` for a package (its spec has a search path).

        """
        cached = sys.modules.get(module_path)
        if cached is not None:
            return hasattr(cached, "__path__")
        parent = module_path.rpartition(".")[0]
        spec = PathFinder.find_spec(
            module_path, self.load(parent).__path__ if parent else None
        )
        return spec is not None and spec.submodule_search_locations is not None

    def defer_instances(
        self,
        module_name: str,
        module_globals: t.ModuleGlobals,
        lazy_imports: Mapping[str, str | tuple[str, str]],
        names: Sequence[str],
    ) -> None:
        """Publish each named configuration root as a ``FlextLazyInstance``.

        Runs after ``install`` on the same globals and map, because ``install``
        clears every exported name. A published stand-in is a plain global, so
        the lazy ``__getattr__`` never loads its module; the first attribute
        read publishes the root through its module. Every root is validated
        before any global changes.

        Raises:
            ValueError: If a root is not an export of the lazy map.

        """
        normalized = self._norm_map(module_name, lazy_imports)
        missing = [name for name in names if name not in normalized]
        if missing:
            msg = (
                f"module {module_name!r} defers instances it does not export: {missing}"
            )
            raise ValueError(msg)
        for name in names:
            entry = normalized[name]
            module_path, attr = (entry, name) if isinstance(entry, str) else entry
            module_globals[name] = FlextLazyInstance(module_path, attr, self.load)

    @staticmethod
    def defer_bases(
        module_name: str,
        bases: Mapping[str, Sequence[tuple[str, str]]],
    ) -> None:
        """Accept an initializer generated before bases were resolved at run time.

        Stand-in bases now come from the part's class statement (ADR-018
        stand-in law rule 4), so the declared bases carry nothing to forward.
        Kept only until every generated initializer is regenerated and deleted
        at zero uses; the public ``defer_lazy_bases`` alias carries the
        deprecation marker.
        """
        del module_name, bases

    @staticmethod
    def _require_deferred(
        module_name: str,
        lazy_imports: Mapping[str, str | tuple[str, str]],
        deferred_members: Collection[str],
    ) -> None:
        """Reject a deferred export that names no class export of the package.

        Raises:
            ValueError: If a deferred name is not a class export in the map.

        """
        for name in deferred_members:
            entry = lazy_imports.get(name)
            if entry is None or (not isinstance(entry, str) and not entry[1]):
                msg = f"module {module_name!r} defers {name!r} without a class export"
                raise ValueError(msg)

    @staticmethod
    def resolve_members(target: type) -> tuple[str, ...]:
        """Resolve every lazy member along ``target``'s MRO; return their names.

        Check gates and tests call this so a broken deferred target fails
        exactly as an eager import would.

        Returns:
            The names of the members that were still deferred, in MRO order.

        Raises:
            AttributeError: If a member its own class binds fails to resolve.

        """
        pending = tuple(
            (klass, name, value)
            for klass in target.__mro__
            for name, value in tuple(vars(klass).items())
            if isinstance(value, FlextLazyMember)
        )
        resolved: list[str] = []
        for klass, name, member in pending:
            try:
                member.resolve()
            except AttributeError:
                # A name only a never-run branch stores leaves the stand-in.
                if name in vars(klass):
                    raise
                continue
            resolved.append(name)
        return tuple(resolved)

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
        deferred_members: Collection[str] | None = None,
    ) -> t.LazyExportHooks:
        """Install __getattr__/__dir__/__all__ and publish _LAZY_IMPORTS.

        The normalized map is published into the module globals as
        ``_LAZY_IMPORTS`` because it is the runtime metadata contract:
        ``_child_map`` (parent merge) and the beartype helpers
        (``lazy_alias_suffixes``/``runtime_alias_names``) read it from
        ``vars(module)``. Publishing here makes every install shape —
        module-level literal or inline call — satisfy that contract from
        the single owner.

        ``deferred_members`` names the class exports published as stand-ins.
        Each resolves to its ``deferred`` stand-in without loading its module,
        so a facade composing it loads no heavy capability; its members are
        resolved at run time from the part's own code (``standin``).

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
        deferred = frozenset(() if deferred_members is None else deferred_members)
        self._require_deferred(module_name, normalized, deferred)
        for name in normalized:
            module_globals.pop(name, None)
        if public_exports is not None:
            names = tuple(dict.fromkeys(public_exports))
        else:
            names = tuple(normalized)

        module_globals["_LAZY_IMPORTS"] = normalized

        def _module_getattr(name: str) -> t.ModuleGlobalValue:
            if name in deferred:
                entry = normalized[name]
                module_path, attr = (entry, name) if isinstance(entry, str) else entry
                standin = self.standin(module_path, attr)
                module_globals[name] = standin
                return standin
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
