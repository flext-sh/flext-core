"""A composition part is a stand-in whose members come from its shared bases.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

from flext_core import defer_lazy_bases, install_lazy_exports
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
        derived, base = f"{package}.derived", f"{package}.bases.base"
        with tm.scope(python_paths=[str(tmp_path)]):
            resolve, _ = install_lazy_exports(
                "lazy_composition_consumer",
                {},
                {"Mixin": (derived, "Mixin")},
                deferred_members={"Mixin": ()},
            )
            defer_lazy_bases(
                "lazy_composition_consumer",
                {"Mixin": ((f"{package}.bases", "Base"),)},
            )
            standin = cast("type", resolve("Mixin"))
            tm.that(
                [klass.__name__ for klass in standin.__mro__],
                eq=["Mixin", "Base", "object"],
            )
            tm.fail(derived in sys.modules)
            tm.fail(base in sys.modules)


__all__: list[str] = ["TestsFlextCoreLazyCompositionParts"]
