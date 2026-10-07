"""PEP 562 lazy class-member descriptor.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flext_core.typings import ModuleGlobalValue


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

        Args:
            owner: The host class the descriptor is bound in.
            name: The attribute name the descriptor is bound under.

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

        Args:
            instance: The instance the descriptor is read through, if any.
            owner: The owner class the descriptor is read through.

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
                setattr(host, self._member, raw)
                return raw
        msg = f"{target!r} declares no member {self._member!r}"
        raise ImportError(msg, name=self._module)


__all__: list[str] = ["FlextLazyMember"]
