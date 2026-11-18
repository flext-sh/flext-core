"""A stand-in base read through an attribute of another package's export.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

from tests import install_lazy_exports, tm

if TYPE_CHECKING:
    from tests import p


class TestsFlextCoreLazyDottedBases:
    """ADR-018 rule 4: an ancestor base is shared exactly as the source names it."""

    @staticmethod
    def test_dotted_base_resolves_through_the_export(
        fixture_tree: p.Tests.FixtureTree,
        tmp_path: Path,
    ) -> None:
        """A base read as ``Outer.Inner`` resolves along the export's attributes."""
        package = f"lazy_dotted_{tmp_path.name.replace('-', '_')}"
        fixture_tree("lazy-dotted-base", destination=tmp_path / package)
        part = f"{package}.part"
        with tm.scope(python_paths=[str(tmp_path)]):
            resolve, _ = install_lazy_exports(
                "lazy_dotted_consumer",
                {},
                {"Part": (part, "Part")},
                deferred_members=("Part",),
            )
            standin = cast("type", resolve("Part"))
            outer = importlib.import_module(f"{package}.outer")
            tm.that(standin.__mro__[1], eq=outer.Outer.Inner)
            tm.fail(part in sys.modules)


__all__: list[str] = ["TestsFlextCoreLazyDottedBases"]
