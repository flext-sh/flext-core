"""A composition part is a stand-in whose members come from its shared bases.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

from flext_core import install_lazy_exports
from tests import tm

if TYPE_CHECKING:
    from tests import p


class TestsFlextCoreLazyCompositionParts:
    """ADR-018 rule 5: a class binding only bases is deferred through them."""

    @staticmethod
    def test_member_free_part_with_bases_is_a_stand_in(
        fixture_tree: p.Tests.FixtureTree,
        tmp_path: Path,
    ) -> None:
        """A part binding no member of its own shares its bases and loads nothing."""
        package = f"lazy_composition_{tmp_path.name.replace('-', '_')}"
        fixture_tree("lazy-deferred-bases", destination=tmp_path / package)
        composed, base = f"{package}.composed", f"{package}.bases.base"
        with tm.scope(python_paths=[str(tmp_path)]):
            resolve, _ = install_lazy_exports(
                "lazy_composition_consumer",
                {},
                {"Composed": (composed, "Composed")},
                deferred_members=("Composed",),
            )
            standin = cast("type", resolve("Composed"))
            tm.that(
                [klass.__name__ for klass in standin.__mro__],
                eq=["Composed", "Base", "object"],
            )
            tm.fail(composed in sys.modules)
            tm.fail(base in sys.modules)

    @staticmethod
    def test_base_imported_from_a_package_is_that_package_export(
        fixture_tree: p.Tests.FixtureTree,
        tmp_path: Path,
    ) -> None:
        """A base read through its package is the stand-in that package publishes."""
        package = f"lazy_routed_{tmp_path.name.replace('-', '_')}"
        fixture_tree("lazy-deferred-bases", destination=tmp_path / package)
        routed, base = f"{package}.routed", f"{package}.bases.base"
        with tm.scope(python_paths=[str(tmp_path)]):
            resolve, _ = install_lazy_exports(
                "lazy_routed_consumer",
                {},
                {"Routed": (routed, "Routed")},
                deferred_members=("Routed",),
            )
            standin = cast("type", resolve("Routed"))
            exported = vars(importlib.import_module(f"{package}.bases"))["Base"]
            tm.ok(standin.__mro__[1] is exported)
            tm.fail(routed in sys.modules)
            tm.fail(base in sys.modules)


__all__: list[str] = ["TestsFlextCoreLazyCompositionParts"]
