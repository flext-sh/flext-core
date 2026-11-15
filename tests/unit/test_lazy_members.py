"""Deferred class members load on first access, bind like the original, fail loud.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Protocol, cast, runtime_checkable

from flext_core import (
    FlextLazyMember,
    defer_lazy_bases,
    defer_lazy_instances,
    install_lazy_exports,
    lazy_member,
    resolve_lazy_members,
)
from tests import c, m, tm, u

if TYPE_CHECKING:
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
            lazy_imports = {"root": (module, "root")}
            install_lazy_exports(
                "lazy_instance_consumer", package_globals, lazy_imports
            )
            defer_lazy_instances(
                "lazy_instance_consumer", package_globals, lazy_imports, ("root",)
            )
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
            lazy_imports = {"root": (module, "root")}
            install_lazy_exports("lazy_instance_broken", package_globals, lazy_imports)
            defer_lazy_instances(
                "lazy_instance_broken", package_globals, lazy_imports, ("root",)
            )
            with u.Tests.raises(ImportError, match="cannot load") as caught:
                tm.that(package_globals["root"], attr_eq={"Infra": "first"})
            tm.that(isinstance(caught.value.__cause__, AttributeError), truth=True)

    @staticmethod
    def test_deferred_instance_without_export_fails() -> None:
        """Deferring a root the lazy map does not export is a generator defect."""
        package_globals: t.ModuleGlobals = {}
        install_lazy_exports("lazy_instance_invalid", package_globals, {})
        with u.Tests.raises(ValueError, match="does not export"):
            defer_lazy_instances(
                "lazy_instance_invalid", package_globals, {}, ("root",)
            )
        tm.fail("root" in package_globals)

    def _standin(
        self,
        fixture_tree: p.Tests.FixtureTree,
        tmp_path: Path,
        consumer: str,
    ) -> tuple[type, str]:
        """Publish the capability stand-in through a fresh initializer.

        Returns:
            The stand-in and the dotted path of the module it defers.

        """
        module = self._deferred_capability(fixture_tree, tmp_path)
        resolve, _ = install_lazy_exports(
            consumer,
            {},
            {"Capability": (module, "Capability")},
            deferred_members={"Capability": ("VALUE", "who")},
        )
        return cast("type", resolve("Capability")), module

    @staticmethod
    def _compose_model_first(standin: type) -> type:
        class ModelFirst(m.BaseModel, standin):
            name: str = "model"

        return ModelFirst

    @staticmethod
    def _compose_model_last(standin: type) -> type:
        class ModelLast(standin, m.BaseModel):
            name: str = "model"

        return ModelLast

    @staticmethod
    def _compose_protocol(standin: type) -> type:
        @runtime_checkable
        class Valued(Protocol):
            VALUE: int

        class Implementing(standin, Valued):
            pass

        return Implementing

    @staticmethod
    def _compose_abc(standin: type) -> type:
        class Abstract(standin, ABC):
            pass

        return Abstract

    @staticmethod
    def _compose_metaclass(standin: type) -> type:
        class Meta(type):
            pass

        class Authored(standin, metaclass=Meta):
            pass

        return Authored

    @staticmethod
    def _compose_generic(standin: type) -> type:
        class Generic[T](standin):
            item: T | None = None

        return Generic

    @staticmethod
    def _compose_enum(standin: type) -> type:
        class Kind(standin, Enum):
            FIRST = 1

        return Kind

    @staticmethod
    def _compose_dataclass(standin: type) -> type:
        @dataclass
        class Record(standin):
            size: int = 0

        return Record

    @c.Tests.MARK.parametrize(
        "kind",
        [
            "model_first",
            "model_last",
            "protocol",
            "abc",
            "metaclass",
            "generic",
            "enum",
            "dataclass",
        ],
    )
    def test_stand_in_composes_with_every_metaclass_without_loading(
        self,
        fixture_tree: p.Tests.FixtureTree,
        tmp_path: Path,
        kind: str,
    ) -> None:
        """A plain-type stand-in composes with any metaclass and loads nothing."""
        with tm.scope(python_paths=[str(tmp_path)]):
            standin, module = self._standin(
                fixture_tree, tmp_path, f"lazy_matrix_{kind}"
            )
            tm.ok(type(standin) is type)
            compose = cast("Callable[[type], type]", getattr(self, f"_compose_{kind}"))
            derived = compose(standin)
            tm.ok(issubclass(derived, standin))
            tm.fail(module in sys.modules)
            tm.that(vars(standin)["VALUE"], is_=FlextLazyMember)
            tm.that(derived.VALUE, eq=7)
            tm.ok(module in sys.modules)

    @staticmethod
    def _loading_part(
        fixture_tree: p.Tests.FixtureTree,
        tmp_path: Path,
        consumer: str,
        members: tuple[str, ...],
    ) -> tuple[type, str]:
        """Publish the stand-in of a part a deriving metaclass reads.

        Returns:
            The stand-in and the dotted path of the module it defers.

        """
        package = f"lazy_loads_{tmp_path.name.replace('-', '_')}"
        fixture_tree("lazy-standin-loads", destination=tmp_path / package)
        module = f"{package}.part"
        resolve, _ = install_lazy_exports(
            consumer, {}, {"Part": (module, "Part")}, deferred_members={"Part": members}
        )
        return cast("type", resolve("Part")), module

    @staticmethod
    def test_abc_resolving_an_abstract_member_loads_the_part(
        fixture_tree: p.Tests.FixtureTree,
        tmp_path: Path,
    ) -> None:
        """ABCMeta reads the member that implements an abstract method."""
        with tm.scope(python_paths=[str(tmp_path)]):
            standin, module = TestsFlextCoreLazyMembers._loading_part(
                fixture_tree, tmp_path, "lazy_loads_abc", ("who",)
            )

            class Contract(ABC):
                @abstractmethod
                def who(self) -> str:
                    """Name the concrete class."""

            tm.fail(module in sys.modules)

            concrete = cast(
                "Callable[[], Contract]", type("Concrete", (standin, Contract), {})
            )
            tm.ok(module in sys.modules)
            tm.that(concrete().who(), eq="Concrete")

    @staticmethod
    def test_model_reading_model_config_loads_the_part(
        fixture_tree: p.Tests.FixtureTree,
        tmp_path: Path,
    ) -> None:
        """Pydantic reads ``model_config`` from every base."""
        with tm.scope(python_paths=[str(tmp_path)]):
            standin, module = TestsFlextCoreLazyMembers._loading_part(
                fixture_tree, tmp_path, "lazy_loads_config", ("model_config",)
            )
            tm.fail(module in sys.modules)

            class Configured(m.BaseModel, standin):
                name: str = "model"

            tm.ok(module in sys.modules)
            tm.that(Configured.model_config.get("frozen"), eq=True)

    @staticmethod
    def test_model_field_shadowing_a_member_loads_the_part(
        fixture_tree: p.Tests.FixtureTree,
        tmp_path: Path,
    ) -> None:
        """Pydantic reads a base member that a field of the same name shadows."""
        with tm.scope(python_paths=[str(tmp_path)]):
            standin, module = TestsFlextCoreLazyMembers._loading_part(
                fixture_tree, tmp_path, "lazy_loads_field", ("VALUE",)
            )
            tm.fail(module in sys.modules)
            with u.Tests.warns(UserWarning, match="shadows an attribute"):

                class Shadowing(m.BaseModel, standin):
                    VALUE: int = 3

            tm.ok(module in sys.modules)
            tm.that(Shadowing().VALUE, eq=3)

    def test_unlisted_member_is_an_attribute_error_and_loads_nothing(
        self,
        fixture_tree: p.Tests.FixtureTree,
        tmp_path: Path,
    ) -> None:
        """A stale member index never falls back to the real class."""
        with tm.scope(python_paths=[str(tmp_path)]):
            standin, module = self._standin(fixture_tree, tmp_path, "lazy_unlisted")
            namespace = type("Namespace", (standin,), {})
            unlisted = "twice"
            with u.Tests.raises(AttributeError, match=unlisted):
                getattr(standin, unlisted)
            with u.Tests.raises(AttributeError, match=unlisted):
                getattr(namespace, unlisted)
            tm.fail(module in sys.modules)

    @staticmethod
    def test_stand_in_without_members_or_bases_fails_when_built() -> None:
        """A stand-in with neither members nor bases stands for nothing."""
        resolve, _ = install_lazy_exports(
            "lazy_deferred_empty",
            {},
            {"Capability": ("lazy_deferred_empty.part", "Capability")},
            deferred_members={"Capability": ()},
        )
        with u.Tests.raises(ValueError, match="neither members nor bases"):
            resolve("Capability")

    def test_base_with_another_metaclass_fails_by_name(
        self,
        fixture_tree: p.Tests.FixtureTree,
        tmp_path: Path,
    ) -> None:
        """A shared base whose metaclass is not ``type`` fails when resolved."""
        module = self._deferred_capability(fixture_tree, tmp_path)
        with tm.scope(python_paths=[str(tmp_path)]):
            resolve, _ = install_lazy_exports(
                "lazy_bad_base",
                {},
                {"Capability": (module, "Capability"), "Abstract": ("abc", "ABC")},
                deferred_members={"Capability": ("VALUE",)},
            )
            defer_lazy_bases("lazy_bad_base", {"Capability": ((".", "Abstract"),)})
            with u.Tests.raises(TypeError, match="'Abstract'.*metaclass 'type'"):
                resolve("Capability")

    _PROBE = (
        Path(__file__).resolve().parents[1] / "fixtures" / "lazy-eager" / "probe.py"
    )

    @c.Tests.MARK.parametrize(
        ("mode", "stdout", "succeeds"),
        [
            ("never", "False FlextLazyMember\n", True),
            ("early", "True int\n", True),
            ("late", "False FlextLazyMember\n", False),
        ],
    )
    def test_eager_flag_publishes_real_classes_only_when_set_first(
        self,
        fixture_tree: p.Tests.FixtureTree,
        tmp_path: Path,
        mode: str,
        stdout: str,
        *,
        succeeds: bool,
    ) -> None:
        """Set first, the flag publishes real classes; set late, it fails."""
        module = self._deferred_capability(fixture_tree, tmp_path)
        result = u.Cli.run_raw(
            [sys.executable, str(self._PROBE), mode, str(tmp_path), module],
            cwd=tmp_path,
        )
        tm.that(result.success, truth=True, msg=result.error)
        tm.that(result.value.stdout, eq=stdout)
        tm.that(
            result.value.outcome.raw_return_code == 0,
            eq=succeeds,
            msg=result.value.stderr,
        )
        if not succeeds:
            tm.that(result.value.stderr, has="after a stand-in was published")

    # Behaviour the lazy-deferred-bases fixture declares, read by name.
    _DERIVED_VALUES: ClassVar[Mapping[str, int]] = {"VALUE": 7}
    _BASE_CALLS: ClassVar[Mapping[str, tuple[int, int]]] = {"twice": (2, 4)}

    def test_stand_in_shares_its_base_members_through_the_real_mro(
        self,
        fixture_tree: p.Tests.FixtureTree,
        tmp_path: Path,
    ) -> None:
        """A stand-in lists its own members; its base stand-in supplies the rest."""
        package = f"lazy_bases_{tmp_path.name.replace('-', '_')}"
        fixture_tree("lazy-deferred-bases", destination=tmp_path / package)
        derived, base = f"{package}.derived", f"{package}.bases.base"
        with tm.scope(python_paths=[str(tmp_path)]):
            resolve, _ = install_lazy_exports(
                "lazy_bases_consumer",
                {},
                {"Mixin": (derived, "Mixin")},
                deferred_members={"Mixin": ("VALUE", "who")},
            )
            defer_lazy_bases(
                "lazy_bases_consumer",
                {"Mixin": ((f"{package}.bases", "Base"),)},
            )
            namespace = type("Namespace", (cast("type", resolve("Mixin")),), {})
            tm.that(
                [klass.__name__ for klass in namespace.__mro__],
                eq=["Namespace", "Mixin", "Base", "object"],
            )
            tm.fail(derived in sys.modules)
            tm.fail(base in sys.modules)
            # The override declared by the deriving part wins, as on the real
            # class; its module imports the base module like any import does.
            for name, value in self._DERIVED_VALUES.items():
                tm.that(getattr(namespace, name), eq=value)
            tm.ok(derived in sys.modules)
            # A member only the base declares resolves through the base stand-in.
            for name, (argument, result) in self._BASE_CALLS.items():
                tm.that(getattr(namespace, name)(argument), eq=result)
            tm.ok(base in sys.modules)

    @staticmethod
    def test_bases_for_an_export_that_is_not_deferred_fail_at_install() -> None:
        """Declaring bases for a name the initializer does not defer is a defect."""
        install_lazy_exports(
            "lazy_bases_undeclared",
            {},
            {"Mixin": ("lazy_bases_undeclared.derived", "Mixin")},
        )
        with u.Tests.raises(ValueError, match="bases exports it does not defer"):
            defer_lazy_bases("lazy_bases_undeclared", {"Mixin": ((".", "Base"),)})

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
