"""Deferred class members load on first access, bind like the original, fail loud.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

from flext_core import (
    FlextLazyMember,
    defer_lazy_instances,
    install_lazy_exports,
    lazy_member,
    resolve_lazy_members,
)
from tests import tm, u

if TYPE_CHECKING:
    from pathlib import Path

    from tests import p, t


class TestsFlextCoreLazyMembers:
    """Drive the public lazy-member descriptor against real modules."""

    _MIXIN_SOURCE = (
        "class Base:\n"
        "    @staticmethod\n"
        "    def twice(value: int) -> int:\n"
        "        return 2 * value\n"
        "\n"
        "class Mixin(Base):\n"
        "    VALUE = 7\n"
        "\n"
        "    class Nested:\n"
        "        pass\n"
        "\n"
        "    @classmethod\n"
        "    def who(cls) -> str:\n"
        "        return cls.__name__\n"
    )

    @staticmethod
    def _package(tmp_path: Path, source: str) -> str:
        package = f"lazy_members_{tmp_path.name.replace('-', '_')}"
        root = tmp_path / package
        root.mkdir()
        (root / "__init__.py").write_text("", encoding="utf-8")
        (root / "heavy.py").write_text(source, encoding="utf-8")
        return f"{package}.heavy"

    def test_members_defer_the_module_and_bind_like_the_original(
        self,
        tmp_path: Path,
    ) -> None:
        """No import before first access; classmethods bind to the subclass."""
        module = self._package(tmp_path, self._MIXIN_SOURCE)
        with tm.scope(python_paths=[str(tmp_path)]):

            class Deferred:
                VALUE = lazy_member(module, "Mixin", "VALUE")
                Nested = lazy_member(module, "Mixin", "Nested")
                who = lazy_member(module, "Mixin", "who")
                twice = lazy_member(module, "Mixin", "twice")

            class Namespace(Deferred):
                pass

            tm.fail(module in sys.modules)
            tm.that(Namespace.VALUE, eq=7)
            tm.ok(module in sys.modules)
            who = cast("Callable[[], str]", Namespace.who)
            twice = cast("Callable[[int], int]", Namespace.twice)
            nested = cast("type", Namespace.Nested)
            tm.that(callable(who), truth=True)
            tm.that(callable(twice), truth=True)
            tm.that(nested, is_=type)
            tm.that(who(), eq="Namespace")
            tm.that(twice(4), eq=8)
            tm.that(nested.__qualname__, eq="Mixin.Nested")
            tm.fail(isinstance(vars(Deferred)["VALUE"], FlextLazyMember))
            tm.ok(isinstance(vars(Deferred)["who"], classmethod))

    def test_resolve_members_forces_every_deferred_member(
        self,
        tmp_path: Path,
    ) -> None:
        """The check entry point resolves the whole MRO and names each member."""
        module = self._package(tmp_path, self._MIXIN_SOURCE)
        with tm.scope(python_paths=[str(tmp_path)]):

            class Deferred:
                VALUE = lazy_member(module, "Mixin", "VALUE")
                twice = lazy_member(module, "Mixin", "twice")

            class Namespace(Deferred):
                pass

            tm.that(resolve_lazy_members(Namespace), eq=("VALUE", "twice"))
            tm.that(resolve_lazy_members(Namespace), eq=())

    def test_absent_member_fails_loud_and_caches_nothing(
        self,
        tmp_path: Path,
    ) -> None:
        """A member the mixin does not declare is an import defect."""
        module = self._package(tmp_path, self._MIXIN_SOURCE)
        with tm.scope(python_paths=[str(tmp_path)]):

            class Deferred:
                missing = lazy_member(module, "Mixin", "missing")

            with u.Tests.raises(ImportError, match="declares no member"):
                _ = Deferred.missing
            tm.ok(isinstance(vars(Deferred)["missing"], FlextLazyMember))

    def test_broken_module_raises_with_its_cause(self, tmp_path: Path) -> None:
        """A mixin module whose body fails propagates, never a missing name."""
        module = self._package(
            tmp_path,
            'msg = "stale body"\nraise AttributeError(msg)\n',
        )
        with tm.scope(python_paths=[str(tmp_path)]):

            class Deferred:
                VALUE = lazy_member(module, "Mixin", "VALUE")

            with u.Tests.raises(ImportError, match="cannot load") as caught:
                _ = Deferred.VALUE
            tm.that(isinstance(caught.value.__cause__, AttributeError), truth=True)

    @staticmethod
    def test_member_must_bind_to_its_own_name() -> None:
        """A descriptor assigned under another name is a generator defect."""

        def _install_renamed() -> None:
            class Deferred:
                renamed = lazy_member("json", "JSONDecoder", "decode")

            _ = Deferred

        with u.Tests.raises(TypeError, match="is bound to attribute"):
            _install_renamed()

    @staticmethod
    def _deferred_capability(
        fixture_tree: p.Tests.FixtureTree,
        tmp_path: Path,
    ) -> str:
        """Materialize the capability scenario as a package no test imported yet.

        Returns:
            The dotted path of the materialized capability module.

        """
        package = f"lazy_deferred_{tmp_path.name.replace('-', '_')}"
        fixture_tree("lazy-deferred-capability", destination=tmp_path / package)
        return f"{package}.capability"

    def test_deferred_export_composes_without_loading_its_module(
        self,
        fixture_tree: p.Tests.FixtureTree,
        tmp_path: Path,
    ) -> None:
        """A facade composes the stand-in; resolving its members loads the module."""
        module = self._deferred_capability(fixture_tree, tmp_path)
        members = ("VALUE", "Nested", "twice", "who")
        with tm.scope(python_paths=[str(tmp_path)]):
            package_globals: t.ModuleGlobals = {}
            resolve, _ = install_lazy_exports(
                "lazy_deferred_consumer",
                package_globals,
                {"Capability": (module, "Capability")},
                deferred_members={"Capability": members},
            )
            standin = cast("type", resolve("Capability"))
            tm.ok(package_globals["Capability"] is standin)
            namespace = type("Namespace", (standin,), {})
            tm.fail(module in sys.modules)
            tm.that(resolve_lazy_members(namespace), eq=members)
            tm.ok(module in sys.modules)
            tm.that(vars(standin)["VALUE"], eq=7)
            tm.ok(isinstance(vars(standin)["who"], classmethod))

    _ROOT_SOURCE = (
        "class Root:\n"
        "    def __init__(self) -> None:\n"
        "        self.Infra = 'first'\n"
        "\n"
        "root = Root()\n"
    )

    def test_deferred_instance_publishes_its_root_on_first_read(
        self,
        tmp_path: Path,
    ) -> None:
        """Importing the root builds nothing; reads forward to the live root."""
        module = self._package(tmp_path, self._ROOT_SOURCE)
        with tm.scope(python_paths=[str(tmp_path)]):
            package_globals: t.ModuleGlobals = {}
            install_lazy_exports(
                "lazy_instance_consumer",
                package_globals,
                {"root": (module, "root")},
            )
            defer_lazy_instances("lazy_instance_consumer", package_globals, ("root",))
            standin = package_globals["root"]
            tm.fail(module in sys.modules)
            tm.that(standin, attr_eq={"Infra": "first"})
            tm.ok(module in sys.modules)
            # An in-place reload of the root stays visible to the importer.
            vars(sys.modules[module])["root"].Infra = "reloaded"
            tm.that(standin, attr_eq={"Infra": "reloaded"})

    def test_deferred_instance_failure_keeps_nothing(self, tmp_path: Path) -> None:
        """A root module whose body fails propagates with its cause."""
        module = self._package(
            tmp_path,
            'msg = "stale body"\nraise AttributeError(msg)\n',
        )
        with tm.scope(python_paths=[str(tmp_path)]):
            package_globals: t.ModuleGlobals = {}
            install_lazy_exports(
                "lazy_instance_broken",
                package_globals,
                {"root": (module, "root")},
            )
            defer_lazy_instances("lazy_instance_broken", package_globals, ("root",))
            with u.Tests.raises(ImportError, match="cannot load") as caught:
                tm.that(package_globals["root"], attr_eq={"Infra": "first"})
            tm.that(isinstance(caught.value.__cause__, AttributeError), truth=True)

    @staticmethod
    def test_deferred_instance_without_export_fails() -> None:
        """Deferring a root the lazy map does not export is a generator defect."""
        package_globals: t.ModuleGlobals = {}
        install_lazy_exports("lazy_instance_invalid", package_globals, {})
        with u.Tests.raises(ValueError, match="does not export"):
            defer_lazy_instances("lazy_instance_invalid", package_globals, ("root",))

    def test_stale_stand_in_reads_an_unlisted_member_from_its_class(
        self,
        fixture_tree: p.Tests.FixtureTree,
        tmp_path: Path,
    ) -> None:
        """A member the generated list omits still resolves on the stand-in itself.

        A stale initializer lists fewer members than the source declares; a
        direct read of the missing member loads the class, while a class
        composed from the stand-in keeps the ordinary lookup.
        """
        module = self._deferred_capability(fixture_tree, tmp_path)
        with tm.scope(python_paths=[str(tmp_path)]):
            package_globals: t.ModuleGlobals = {}
            resolve, _ = install_lazy_exports(
                "lazy_deferred_stale_consumer",
                package_globals,
                {"Capability": (module, "Capability")},
                deferred_members={"Capability": ("VALUE",)},
            )
            standin = cast("type", resolve("Capability"))
            namespace = type("Namespace", (standin,), {})
            # The behavior under test is resolution by name, as a stale
            # initializer's consumer reads it, so every read names its member.
            unlisted = ("twice", "who", "absent")
            with u.Tests.raises(AttributeError, match=unlisted[0]):
                getattr(namespace, unlisted[0])
            tm.fail(module in sys.modules)
            tm.that(getattr(standin, unlisted[0])(3), eq=6)
            tm.that(getattr(standin, unlisted[1])(), eq="Capability")
            tm.ok(module in sys.modules)
            tm.ok(isinstance(vars(standin)[unlisted[1]], classmethod))
            tm.that(getattr(namespace, unlisted[0])(4), eq=8)
            with u.Tests.raises(
                AttributeError, match="'Capability' has no attribute 'absent'"
            ):
                getattr(standin, unlisted[2])

    @staticmethod
    def test_deferred_name_without_class_export_fails_at_install() -> None:
        """Deferring a name the lazy map does not export is a generator defect."""
        with u.Tests.raises(ValueError, match="without a class export"):
            install_lazy_exports(
                "lazy_deferred_invalid",
                {},
                {},
                deferred_members={"Capability": ("VALUE",)},
            )
