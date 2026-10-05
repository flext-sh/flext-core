"""Deferred class members load on first access, bind like the original, fail loud.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import pytest
from flext_tests import tm

from flext_core.lazy import FlextLazyMember, lazy_member, resolve_lazy_members

if TYPE_CHECKING:
    from pathlib import Path


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

            tm.that(module in sys.modules, eq=False)
            tm.that(Namespace.VALUE, eq=7)
            tm.that(module in sys.modules, eq=True)
            tm.that(Namespace.who(), eq="Namespace")
            tm.that(Namespace.twice(4), eq=8)
            tm.that(Namespace.Nested.__qualname__, eq="Mixin.Nested")
            tm.that(isinstance(vars(Deferred)["VALUE"], FlextLazyMember), eq=False)
            tm.that(isinstance(vars(Deferred)["who"], classmethod), eq=True)

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

            with pytest.raises(ImportError, match="declares no member"):
                _ = Deferred.missing
            tm.that(isinstance(vars(Deferred)["missing"], FlextLazyMember), eq=True)

    def test_broken_module_raises_with_its_cause(self, tmp_path: Path) -> None:
        """A mixin module whose body fails propagates, never a missing name."""
        module = self._package(
            tmp_path,
            'msg = "stale body"\nraise AttributeError(msg)\n',
        )
        with tm.scope(python_paths=[str(tmp_path)]):

            class Deferred:
                VALUE = lazy_member(module, "Mixin", "VALUE")

            with pytest.raises(ImportError, match="cannot load") as caught:
                _ = Deferred.VALUE
            assert isinstance(caught.value.__cause__, AttributeError)

    @staticmethod
    def test_member_must_bind_to_its_own_name() -> None:
        """A descriptor assigned under another name is a generator defect."""

        def _install_renamed() -> None:
            class Deferred:
                renamed = lazy_member("json", "JSONDecoder", "decode")

            _ = Deferred

        with pytest.raises(TypeError, match="is bound to attribute"):
            _install_renamed()
