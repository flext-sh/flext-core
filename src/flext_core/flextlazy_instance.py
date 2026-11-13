"""Stand-in for a configuration root whose validation is deferred to first use.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    from flext_core import t


class FlextLazyInstance:
    """Forward attribute reads to a root instance published on first use.

    Importing the stand-in builds nothing. The first attribute read loads the
    declaring module through the lazy loader, which publishes the root, and
    keeps that root: never one of its members, so an in-place
    ``reset_for_testing`` of the root stays visible to every importer. A
    failure raises with its cause and keeps nothing.
    """

    __slots__ = ("_attr", "_load", "_module", "_root")

    def __init__(
        self,
        module: str,
        attr: str,
        load: Callable[[str], ModuleType],
    ) -> None:
        """Record the declaring module, the root's name and the loader.

        Args:
            module: Dotted path of the module that publishes the root.
            attr: Name the root is bound under in ``module``.
            load: The ``FlextLazy`` module loader, the single dynamic-import
                owner, used on first access.

        """
        self._module = module
        self._attr = attr
        self._load = load
        self._root: t.ModuleGlobalValue | None = None

    def __getattr__(self, name: str) -> t.ModuleGlobalValue:
        """Read ``name`` from the published root.

        Returns:
            The root's attribute.

        Raises:
            AttributeError: When the declaring module cannot publish the root.

        """
        root = self._root
        if root is None:
            target = f"{self._module}.{self._attr}"
            try:
                root = getattr(self._load(self._module), self._attr)
            except AttributeError as exc:
                msg = f"lazy instance cannot load {target!r}"
                raise AttributeError(msg) from exc
            self._root = root
        value: t.ModuleGlobalValue = getattr(root, name)
        return value


__all__: list[str] = ["FlextLazyInstance"]
