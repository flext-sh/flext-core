"""Stand-in members are resolved at run time from the part's own code.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import importlib
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar, cast

from flext_core import FlextLazyMember, install_lazy_exports
from tests import tm, u

if TYPE_CHECKING:
    from tests import p


class TestsFlextCoreLazyDynamicMembers:
    """Operator decision 2026-11-15: no member list, only the code decides."""

    _BOUND: tuple[str, ...] = ("VALUE", "ADDED", "LEFT", "RIGHT", "who", "twice")
    _OTHER_BOUND: tuple[str, ...] = ("OTHER",)
    _DECLARED: str = "spawn"
    _GENERIC_BOUND: tuple[str, ...] = ("SIZE", "first")
    _SUBSCRIPT: str = "__class_getitem__"
    _LOCAL_MRO: tuple[str, ...] = ("Local", "Helper")
    _LOCAL_BOUND: tuple[str, ...] = ("HELPER", "LOCAL")
    _EXPORTS: tuple[str, ...] = (
        "Part",
        "Other",
        "Empty",
        "Generic",
        "Missing",
        "Local",
        "Declared",
        "Provider",
    )

    @classmethod
    def _publish(
        cls,
        fixture_tree: p.Tests.FixtureTree,
        tmp_path: Path,
        consumer: str,
    ) -> tuple[Callable[[str], object], str]:
        """Install the scenario's exports, every one a stand-in.

        ``Missing`` names a class no module declares.

        Returns:
            The initializer's resolver and the scenario package name.

        """
        package = f"dynamic_{tmp_path.name.replace('-', '_')}"
        fixture_tree("lazy-dynamic-members", destination=tmp_path / package)
        resolve, _ = install_lazy_exports(
            consumer,
            {},
            {
                name: (f"{package}.{module}", name)
                for name, module in zip(
                    cls._EXPORTS,
                    (
                        "part",
                        "other",
                        "empty",
                        "generic",
                        "part",
                        "local",
                        "declared",
                        "declared",
                    ),
                    strict=True,
                )
            },
            deferred_members=cls._EXPORTS,
        )
        return resolve, package

    @staticmethod
    def _deferred(standin: type) -> tuple[str, ...]:
        """Return the member names a stand-in still defers, in binding order."""
        return tuple(
            name
            for name, value in vars(standin).items()
            if isinstance(value, FlextLazyMember)
        )

    def test_members_come_from_the_code_and_a_member_loads_only_its_part(
        self,
        fixture_tree: p.Tests.FixtureTree,
        tmp_path: Path,
    ) -> None:
        """Every bound name resolves, an absent one fails; the sibling never loads."""
        with tm.scope(python_paths=[str(tmp_path)]):
            resolve, package = self._publish(fixture_tree, tmp_path, "lazy_dyn_names")
            part, other = f"{package}.part", f"{package}.other"
            standin = cast("type", resolve("Part"))
            sibling = cast("type", resolve("Other"))
            tm.ok(resolve("Part") is standin)
            tm.that(self._deferred(standin), eq=self._BOUND)
            tm.that(self._deferred(sibling), eq=self._OTHER_BOUND)
            tm.fail(part in sys.modules)
            namespace = type("Namespace", (standin, sibling), {})
            absent = "absent"
            with u.Tests.raises(AttributeError, match=absent):
                getattr(namespace, absent)
            tm.fail(part in sys.modules)
            added, left = self._BOUND[1], self._BOUND[2]
            tm.that(getattr(namespace, added), eq="added")
            tm.that(getattr(namespace, left), eq=1)
            tm.ok(part in sys.modules)
            tm.fail(other in sys.modules)
            with u.Tests.raises(AttributeError, match=absent):
                getattr(namespace, absent)

    def test_generic_part_is_a_generic_stand_in_equal_to_its_class(
        self,
        fixture_tree: p.Tests.FixtureTree,
        tmp_path: Path,
    ) -> None:
        """A PEP 695 part's stand-in is generic like the class and loads nothing."""
        with tm.scope(python_paths=[str(tmp_path)]):
            resolve, package = self._publish(fixture_tree, tmp_path, "lazy_dyn_generic")
            module = f"{package}.generic"
            standin = cast("type", resolve("Generic"))
            subscript = cast(
                "Callable[[object], type]", getattr(standin, self._SUBSCRIPT)
            )
            subscribed = subscript(int)

            U = TypeVar("U")
            facade = types.new_class("Facade", (subscript(U),))

            class Concrete(subscribed):
                pass

            tm.ok(type(standin) is type)
            tm.that(self._deferred(standin), eq=self._GENERIC_BOUND)
            tm.fail(module in sys.modules)
            real = cast("type", vars(importlib.import_module(module))["Generic"])
            tm.that(
                [klass.__qualname__ for klass in standin.__mro__],
                eq=[klass.__qualname__ for klass in real.__mro__],
            )
            tm.that(
                [param.__name__ for param in standin.__parameters__],
                eq=[param.__name__ for param in real.__parameters__],
            )
            real_subscript = cast(
                "Callable[[object], type]", getattr(real, self._SUBSCRIPT)
            )
            tm.that(subscribed.__args__, eq=real_subscript(int).__args__)
            tm.that(facade.__mro__[1:], eq=standin.__mro__)
            tm.that(facade.__parameters__, eq=(U,))
            tm.that(getattr(Concrete, self._GENERIC_BOUND[0]), eq=3)

    def test_same_module_base_is_its_own_stand_in(
        self,
        fixture_tree: p.Tests.FixtureTree,
        tmp_path: Path,
    ) -> None:
        """A base the module itself defines becomes a stand-in; nothing flattens."""
        with tm.scope(python_paths=[str(tmp_path)]):
            resolve, package = self._publish(fixture_tree, tmp_path, "lazy_dyn_local")
            standin = cast("type", resolve("Local"))
            tm.that(
                [klass.__qualname__ for klass in standin.__mro__],
                eq=[*self._LOCAL_MRO, "object"],
            )
            tm.that(self._deferred(standin), eq=(self._LOCAL_BOUND[1],))
            tm.fail(f"{package}.local" in sys.modules)
            tm.that(getattr(standin, self._LOCAL_BOUND[0]), eq=1)
            tm.ok(f"{package}.local" in sys.modules)

    def test_class_the_code_does_not_declare_is_an_attribute_error(
        self,
        fixture_tree: p.Tests.FixtureTree,
        tmp_path: Path,
    ) -> None:
        """What does not exist in the code is an error, and nothing loads."""
        with tm.scope(python_paths=[str(tmp_path)]):
            resolve, package = self._publish(fixture_tree, tmp_path, "lazy_dyn_missing")
            with u.Tests.raises(AttributeError, match="declares no class 'Missing'"):
                resolve("Missing")
            tm.fail(f"{package}.part" in sys.modules)

    def test_part_binding_nothing_without_bases_fails_when_built(
        self,
        fixture_tree: p.Tests.FixtureTree,
        tmp_path: Path,
    ) -> None:
        """A stand-in with neither members nor bases stands for nothing."""
        with tm.scope(python_paths=[str(tmp_path)]):
            resolve, package = self._publish(fixture_tree, tmp_path, "lazy_dyn_empty")
            with u.Tests.raises(ValueError, match="neither members nor bases"):
                resolve("Empty")
            tm.fail(f"{package}.empty" in sys.modules)

    def test_name_stored_only_for_type_checkers_continues_along_the_mro(
        self,
        fixture_tree: p.Tests.FixtureTree,
        tmp_path: Path,
    ) -> None:
        """A name the class never binds leaves the stand-in; the next base binds it."""
        with tm.scope(python_paths=[str(tmp_path)]):
            resolve, _ = self._publish(fixture_tree, tmp_path, "lazy_dyn_declared")
            declared = cast("type", resolve("Declared"))
            namespace = type(
                "Namespace", (declared, cast("type", resolve("Provider"))), {}
            )
            spawn = cast("Callable[[], str]", getattr(namespace, self._DECLARED))
            tm.that(spawn(), eq="provider")
            tm.fail(self._DECLARED in vars(declared))


__all__: list[str] = ["TestsFlextCoreLazyDynamicMembers"]
