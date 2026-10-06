"""PEP 562 lazy attribute descriptor.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from flext_core._lazy_parts.flextlazy_part_01 import LazyImportMap

if TYPE_CHECKING:
    from flext_core.typings import ModuleGlobals, ModuleGlobalValue
    from flext_core._lazy_parts.flextlazy_part_02 import FlextLazy


class FlextLazyAttribute[T]:
    """Descriptor that resolves a class attribute through ``FlextLazy``.

    Generic in the resolved symbol so a class-namespace lazy attribute keeps its
    declared static type instead of widening to the untyped module-global union.
    """

    __slots__ = ("_lazy", "_lazy_imports", "_module_globals", "_module_name", "_name")

    def __init__(
        self,
        lazy: FlextLazy,
        name: str,
        lazy_imports: LazyImportMap,
        module_globals: ModuleGlobals,
        module_name: str,
    ) -> None:
        self._lazy = lazy
        self._name = name
        self._lazy_imports = lazy_imports
        self._module_globals = module_globals
        self._module_name = module_name

    def __get__(
        self,
        instance: ModuleGlobalValue | None,
        owner: type | None = None,
    ) -> T:
        """Resolve and cache the target symbol through the owning lazy container.

        Returns:
            The resulting ``T``.

        """
        _ = instance, owner
        resolved: T = cast(
            "T",
            self._lazy.get(
                self._name,
                self._lazy_imports,
                self._module_globals,
                self._module_name,
            ),
        )
        return resolved


__all__: list[str] = ["FlextLazyAttribute"]
