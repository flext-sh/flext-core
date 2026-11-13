"""PEP 562 lazy class-member descriptor.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    from flext_core import t


class FlextLazyMember:
    """Class-namespace descriptor that defers one mixin member to first access.

    A generated deferred base declares one descriptor per member of a capability
    mixin. The first access imports the mixin module, takes the member's raw
    value from the mixin's MRO (so a staticmethod or classmethod keeps its
    binding semantics) and replaces the descriptor on its host, making every
    later lookup a plain class attribute. A failure raises with its cause and
    caches nothing.
    """

    __slots__ = ("_host", "_load", "_member", "_module", "_owner")

    def __init__(
        self,
        module: str,
        owner: str,
        member: str,
        load: Callable[[str], ModuleType],
    ) -> None:
        """Record the mixin member and the lazy owner's module loader.

        Args:
            module: Dotted path of the module declaring the mixin.
            owner: Name of the mixin class inside ``module``.
            member: Name of the member to resolve from the mixin's MRO.
            load: The ``FlextLazy`` module loader, the single dynamic-import
                owner, used on first access.

        """
        self._module = module
        self._owner = owner
        self._member = member
        self._load = load
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
        instance: t.ModuleGlobalValue | None,
        owner: type | None = None,
    ) -> t.ModuleGlobalValue:
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
        bound: t.ModuleGlobalValue = binder(raw, instance, owner or self._host)
        return bound

    def resolve(self) -> t.ModuleGlobalValue:
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
            source = cast(
                "type",
                getattr(self._load(self._module), self._owner),
            )
        except AttributeError as exc:
            msg = f"lazy member {self._member!r} cannot load {target!r}"
            raise ImportError(msg, name=self._module) from exc
        for klass in source.__mro__:
            namespace = cast("t.ModuleGlobals", vars(klass))
            if self._member in namespace:
                raw: t.ModuleGlobalValue = cast(
                    "t.ModuleGlobalValue",
                    namespace[self._member],
                )
                setattr(host, self._member, raw)
                return raw
        msg = f"{target!r} declares no member {self._member!r}"
        raise ImportError(msg, name=self._module)


class FlextLazyStandIn(type):
    """Metaclass of a deferred stand-in: a name it does not list reads the class.

    The listed members are ``FlextLazyMember`` descriptors so a facade can
    compose the stand-in by MRO without importing anything. A name that no
    class in the MRO binds, such as a member added to the source after the
    initializer was generated, resolves from the real class of the first
    stand-in that declares it. This holds on the stand-in itself and on every
    class composed from it, so a stale initializer never hides a member of the
    class it stands for, and a generator never needs its own regenerated
    initializer to run.
    """

    def __getattr__(cls, name: str) -> t.ModuleGlobalValue:
        """Resolve an unlisted name from the classes the MRO's stand-ins stand for.

        Returns:
            The real class's member, cached on its stand-in and bound to the
            class it is read through.

        Raises:
            AttributeError: For a dunder, or when no stand-in's real class
                declares the attribute.

        """
        if not (name.startswith("__") and name.endswith("__")):
            for standin in cls.__mro__:
                target = standin.__dict__.get("__flext_lazy_target__")
                if target is None:
                    continue
                module, owner, load = cast(
                    "tuple[str, str, Callable[[str], ModuleType]]", target
                )
                real = cast("type", getattr(load(module), owner))
                for klass in real.__mro__:
                    namespace = cast("t.ModuleGlobals", vars(klass))
                    if name in namespace:
                        # Cache the raw member on the stand-in, as a listed
                        # member is, so the next read is a plain MRO lookup.
                        setattr(standin, name, namespace[name])
                        return cast(
                            "t.ModuleGlobalValue", type.__getattribute__(cls, name)
                        )
        msg = f"type object {cls.__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)


__all__: list[str] = ["FlextLazyMember", "FlextLazyStandIn"]
