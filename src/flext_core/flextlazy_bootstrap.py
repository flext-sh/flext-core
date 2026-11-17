"""PEP 562 bootstrap helpers independent of generated child initializers.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import dis
import importlib
import inspect
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from importlib.abc import InspectLoader
from importlib.machinery import PathFinder
from importlib.util import resolve_name
from types import CodeType, ModuleType
from typing import TYPE_CHECKING

from pydantic import RootModel

if TYPE_CHECKING:
    from types import FrameType

    from flext_core import t


class FlextLazyBootstrap:
    """Canonical lazy API as a container with runtime reuse caches.

    This bootstrap loader sits below the generated typing and model packages it
    installs, so it cannot import their facade: it is a behavior container, not
    a validated data contract, and stays a plain class exactly like its
    ``FlextLazyMember``/``FlextLazyAttribute`` siblings.
    """

    __slots__ = (
        "_import_module",
        "child_lazy_cache",
        "child_merge_cache",
        "class_cache",
        "code_cache",
        "import_cache",
        "install_cache",
        "module_cache",
        "normalized_map_cache",
        "standins",
    )

    generic_scope_prefix: str = "<generic parameters of "
    """Name prefix of the code that holds a PEP 695 class statement."""

    err_relative_path_requires_module: str = (
        "relative lazy-import paths require a parent module name"
    )

    def __init__(self) -> None:
        """Initialize the empty reuse caches and the default module importer."""
        self.module_cache: MutableMapping[str, ModuleType] = {}
        self.child_lazy_cache: MutableMapping[
            str, MutableMapping[str, str | tuple[str, str]]
        ] = {}
        self.child_merge_cache: MutableMapping[
            tuple[str, ...],
            MutableMapping[str, str | tuple[str, str]],
        ] = {}
        # The entry keeps its source map alive: an ``id()`` key alone is reused
        # once a transient map is freed, and would serve another map's
        # normalization.
        self.normalized_map_cache: MutableMapping[
            tuple[str, int],
            tuple[
                Mapping[str, str | tuple[str, str]] | None,
                MutableMapping[str, str | tuple[str, str]],
            ],
        ] = {}
        self.install_cache: MutableMapping[
            str, tuple[tuple[int, int, int, int], t.LazyExportHooks]
        ] = {}
        # Compiled code, classes and imports read for stand-ins, and every
        # stand-in built by ``(module, qualname)``: run time only, in memory,
        # reused and grown by each discovery, never discarded.
        self.code_cache: MutableMapping[str, CodeType] = {}
        self.class_cache: MutableMapping[
            str, Mapping[str, tuple[CodeType, CodeType]]
        ] = {}
        self.import_cache: MutableMapping[str, Mapping[str, tuple[str, str]]] = {}
        self.standins: MutableMapping[tuple[str, str], type] = {}
        self._import_module: Callable[[str], ModuleType] = importlib.import_module

    @property
    def cache_stats(self) -> Mapping[str, int]:
        """Expose cache sizes for diagnostics/observability."""
        return {
            "module_cache": len(self.module_cache),
            "child_lazy_cache": len(self.child_lazy_cache),
            "child_merge_cache": len(self.child_merge_cache),
            "normalized_map_cache": len(self.normalized_map_cache),
            "install_cache": len(self.install_cache),
        }

    @staticmethod
    def _norm_cache_key(
        module_path: str,
        raw: Mapping[str, str | tuple[str, str]] | None,
    ) -> tuple[str, int]:
        return (module_path, id(raw))

    def _norm_map(
        self,
        module_path: str,
        raw: Mapping[str, str | tuple[str, str]] | None,
    ) -> MutableMapping[str, str | tuple[str, str]]:
        cache_key = self._norm_cache_key(module_path, raw)
        cached = self.normalized_map_cache.get(cache_key)
        if cached is not None and cached[0] is raw:
            return cached[1]

        # The entry shape is validated with pydantic's typed containers. This
        # bootstrap precedes the models facade it installs, so it binds only
        # ``RootModel`` here (never a model base that would cross the facade
        # boundary) and lets the native validator parse once; its
        # ValidationError escapes unchanged.
        validated = RootModel[dict[str, str | tuple[str, str]]].model_validate(raw).root

        out: MutableMapping[str, str | tuple[str, str]] = {}
        for name, entry in validated.items():
            if isinstance(entry, str):
                out[name] = self._child_path(entry, module_path)
                continue
            target, attr = entry
            resolved = self._child_path(target, module_path)
            out[name] = (resolved, attr)

        self.normalized_map_cache[cache_key] = (raw, out)
        return out

    def normalize_map(
        self,
        module_path: str,
        raw: Mapping[str, str | tuple[str, str]] | None,
    ) -> MutableMapping[str, str | tuple[str, str]]:
        """Return normalized lazy-import entries for runtime metadata readers.

        Returns:
            Normalized lazy-import entries for runtime metadata readers.

        """
        return self._norm_map(module_path, raw)

    @staticmethod
    def _module_is_initializing(module: ModuleType) -> bool:
        """Return whether this thread is still executing ``module``'s body.

        ``importlib.import_module`` waits on the per-module import lock while
        another thread initializes a module, so the only partial module it can
        hand back is one this thread is importing (a circular import). That
        state is public runtime data: a ``<module>`` code frame on the current
        stack whose globals are the module namespace.

        Returns:
            Whether this thread is still executing ``module``'s body.

        Raises:
            RuntimeError: If flext_core lazy exports require interpreter stack frames
                (inspect.currentframe() returned None; CPython is required).

        """
        frame: FrameType | None = inspect.currentframe()
        if frame is None:
            msg = (
                "flext_core lazy exports require interpreter stack frames "
                "(inspect.currentframe() returned None; CPython is required)"
            )
            raise RuntimeError(msg)
        namespace = vars(module)
        try:
            while frame is not None:
                if frame.f_code.co_name == "<module>" and frame.f_globals is namespace:
                    return True
                frame = frame.f_back
            return False
        finally:
            del frame

    def load(self, module_path: str) -> ModuleType:
        """Import ``module_path`` once and reuse it from the module cache.

        This is the lazy machinery's single dynamic-import path: lazy exports,
        lazy members and data-driven discovery (module names read from
        published metadata) all load through it. A module still initializing
        in this thread (a circular import) is returned but never cached.

        Returns:
            The imported module.

        """
        cached = self.module_cache.get(module_path)
        if cached is not None:
            return cached
        mod = self._import_module(module_path)
        if not self._module_is_initializing(mod):
            self.module_cache[module_path] = mod
        return mod

    def _module_code(self, module_path: str) -> CodeType:
        """Return the compiled code of ``module_path`` without importing it.

        The code is read once per process by the standard path-based finder's
        own loader, so an import hook (assertion rewriting, a type-checking
        claw) neither hides nor alters it, and is kept in memory for every
        later stand-in of the same module.

        Returns:
            The module's code object.

        Raises:
            ModuleNotFoundError: If ``module_path`` does not exist.
            ImportError: If its loader exposes no code to read.

        """
        cached = self.code_cache.get(module_path)
        if cached is not None:
            return cached
        parent = module_path.rpartition(".")[0]
        spec = PathFinder.find_spec(
            module_path, self.load(parent).__path__ if parent else None
        )
        if spec is None:
            msg = f"no module named {module_path!r}"
            raise ModuleNotFoundError(msg, name=module_path)
        loader = spec.loader
        code = (
            loader.get_code(module_path) if isinstance(loader, InspectLoader) else None
        )
        if code is None:
            msg = f"module {module_path!r} exposes no code to read"
            raise ImportError(msg, name=module_path)
        self.code_cache[module_path] = code
        return code

    def _module_classes(
        self, module_path: str
    ) -> Mapping[str, tuple[CodeType, CodeType]]:
        """Return every class of ``module_path``: its body and its statement scope.

        Class statements live in a module, a class body or the type-parameter
        scope of a PEP 695 class, never in a function, so only those code
        objects are walked and nothing is decoded. The result is kept in
        memory for every later stand-in of the module.

        Returns:
            ``{qualname: (body, scope holding its statement)}``.

        """
        cached = self.class_cache.get(module_path)
        if cached is not None:
            return cached
        classes: dict[str, tuple[CodeType, CodeType]] = {}
        pending = [self._module_code(module_path)]
        while pending:
            scope = pending.pop()
            for const in scope.co_consts:
                if not isinstance(const, CodeType):
                    continue
                if not const.co_flags & inspect.CO_OPTIMIZED:
                    classes[const.co_qualname] = (const, scope)
                    pending.append(const)
                elif const.co_name.startswith(self.generic_scope_prefix):
                    pending.append(const)
        self.class_cache[module_path] = classes
        return classes

    def _class_statement(
        self, module_path: str, qualname: str
    ) -> tuple[CodeType, CodeType]:
        """Return the class body of ``qualname`` and the scope of its statement.

        Returns:
            ``(body, scope)``.

        Raises:
            AttributeError: If the code declares no class ``qualname``.

        """
        found = self._module_classes(module_path).get(qualname)
        if found is None:
            msg = f"module {module_path!r} declares no class {qualname!r}"
            raise AttributeError(msg, name=qualname)
        return found

    def _class_bases(
        self, module_path: str, qualname: str
    ) -> tuple[tuple[str, ...], ...]:
        """Return each base of a plain class statement as a dotted path.

        Only the statement is decoded: the units that follow the body
        constant, up to the call that builds the class.

        Returns:
            One tuple of names per base, in base order (``Outer.Inner`` is
            ``("Outer", "Inner")``).

        Raises:
            TypeError: If a base is not a dotted name or the statement passes
                keywords (a metaclass), which no stand-in represents.

        """
        body, scope = self._class_statement(module_path, qualname)
        op = dis.opmap
        units = self._units(scope)
        index = next(
            position
            for position, (opcode, argument) in enumerate(units)
            if opcode == op["LOAD_CONST"] and scope.co_consts[argument] is body
        )
        packing = {op["BUILD_LIST"], op["LIST_APPEND"], op["CALL_INTRINSIC_1"]}
        bases: list[tuple[str, ...]] = []
        # MAKE_FUNCTION, then the class name, then one load chain per base;
        # a long base list is packed into a list and called with ``*args``.
        for opcode, argument in units[index + 3 :]:
            if opcode in {op["CALL"], op["CALL_FUNCTION_EX"]}:
                return tuple(bases)
            if opcode in packing:
                continue
            if opcode == op["LOAD_NAME"]:
                bases.append((scope.co_names[argument],))
            elif opcode == op["LOAD_ATTR"] and bases:
                # ``LOAD_ATTR`` keeps a method-load flag in its lowest bit.
                bases[-1] = (*bases[-1], scope.co_names[argument >> 1])
            else:
                break
        msg = (
            f"class {module_path}.{qualname} has a base or keyword that is not "
            "a dotted name; no stand-in represents it"
        )
        raise TypeError(msg)

    def _class_members(self, module_path: str, qualname: str) -> tuple[str, ...]:
        """Return the names the class body stores, mangled as the class stores them.

        Returns:
            The non-dunder member names in binding order.

        """
        body, _ = self._class_statement(module_path, qualname)
        store, extend = dis.opmap["STORE_NAME"], dis.opmap["EXTENDED_ARG"]
        raw, names, stored, prefix = body.co_code, body.co_names, dict[str, None](), 0
        # Inline ``CACHE`` units never carry ``STORE_NAME``; ``EXTENDED_ARG``
        # prefixes fold into the argument.
        for index in range(0, len(raw), 2):
            opcode, argument = raw[index], prefix | raw[index + 1]
            prefix = argument << 8 if opcode == extend else 0
            if opcode == store:
                stored[names[argument]] = None
        return tuple(
            name
            for name in stored
            if not (name.startswith("__") and name.endswith("__"))
        )

    def _import_route(self, module_path: str, name: str) -> tuple[str, str] | None:
        """Return the ``(module, attribute)`` a ``from`` import binds to ``name``.

        The module's ``from`` imports are decoded once and kept in memory.

        Returns:
            The absolute source module and imported attribute, or ``None``
            when the module binds ``name`` by no ``from`` import.

        """
        routes = self.import_cache.get(module_path)
        if routes is None:
            routes = {}
            code = self._module_code(module_path)
            units = self._units(code)
            package = module_path.rpartition(".")[0]
            op = dis.opmap
            for index, (opcode, argument) in enumerate(units):
                if opcode != op["IMPORT_NAME"]:
                    continue
                level = code.co_consts[units[index - 2][1]]
                source = code.co_names[argument]
                absolute = (
                    resolve_name("." * level + source, package)
                    if isinstance(level, int) and level
                    else source
                )
                for (imported, origin), (stored, alias) in zip(
                    units[index + 1 :: 2], units[index + 2 :: 2], strict=False
                ):
                    if imported != op["IMPORT_FROM"] or stored != op["STORE_NAME"]:
                        break
                    routes[code.co_names[alias]] = (absolute, code.co_names[origin])
            self.import_cache[module_path] = routes
        return routes.get(name)

    @staticmethod
    def _units(code: CodeType) -> Sequence[tuple[int, int]]:
        """Return ``code``'s instructions as ``(opcode, argument)`` pairs.

        The wordcode is read directly, two bytes per unit: ``EXTENDED_ARG``
        prefixes fold into the argument and inline ``CACHE`` units are
        skipped. ``dis.get_instructions`` builds an object per instruction
        and measured 65 times slower over the 660 parts of flext-infra and
        flext-cli (377 ms against 5.7 ms). Only statement scopes and module
        code are decoded, never class bodies.

        Returns:
            The decoded instructions in order.

        """
        extend, cache = dis.opmap["EXTENDED_ARG"], dis.opmap["CACHE"]
        raw, units, prefix = code.co_code, list[tuple[int, int]](), 0
        for index in range(0, len(raw), 2):
            opcode, argument = raw[index], prefix | raw[index + 1]
            prefix = argument << 8 if opcode == extend else 0
            if opcode not in {extend, cache}:
                units.append((opcode, argument))
        return units

    def _child_map(
        self, module_path: str
    ) -> MutableMapping[str, str | tuple[str, str]]:
        cached = self.child_lazy_cache.get(module_path)
        if cached is not None:
            return cached
        child = self.load(module_path)
        raw_lazy_imports = vars(child).get("_LAZY_IMPORTS")
        if raw_lazy_imports is None:
            normalized: MutableMapping[str, str | tuple[str, str]] = {}
            self.child_lazy_cache[module_path] = normalized
            return normalized
        normalized = self._norm_map(child.__name__, raw_lazy_imports)
        self.child_lazy_cache[module_path] = normalized
        return normalized

    def _child_path(self, path: str, module_name: str | None) -> str:
        if not path.startswith("."):
            return path
        if module_name:
            return resolve_name(path, module_name)
        raise ValueError(self.err_relative_path_requires_module)

    def reset(self) -> None:
        """Reset all lazy caches to a clean state."""
        self.module_cache.clear()
        self.child_lazy_cache.clear()
        self.child_merge_cache.clear()
        self.normalized_map_cache.clear()
        self.install_cache.clear()

    @staticmethod
    def build_map(
        module_groups: Mapping[str, Sequence[str]] | None = None,
        *,
        alias_groups: Mapping[str, Sequence[tuple[str, str]]] | None = None,
        sort_keys: bool = True,
    ) -> MutableMapping[str, str | tuple[str, str]]:
        """Build one flat lazy-import map.

        Returns:
            The resulting ``MutableLazyImportMap``.

        """
        out: MutableMapping[str, str | tuple[str, str]] = {
            name: module
            for module, names in (module_groups or {}).items()
            for name in names
        }
        for module, pairs in (alias_groups or {}).items():
            for name, attr in pairs:
                out[name] = (module, attr)
        return {name: out[name] for name in sorted(out)} if sort_keys else out


__all__: list[str] = ["FlextLazyBootstrap"]
