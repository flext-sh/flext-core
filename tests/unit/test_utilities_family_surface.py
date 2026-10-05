"""Family-surface derivation tests.

Covers the discovery seed of ``FlextUtilitiesFamilySurface``: a distribution
that requires a family member is a discovery candidate exactly like a
family-prefixed one, while an unrelated distribution never joins the surface.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT.
"""

from __future__ import annotations

from flext_tests import tm

from flext_core.utilities import FlextUtilitiesFamilySurface


class TestsFlextCoreUtilitiesFamilySurface:
    """Tests for ``FlextCoreUtilitiesFamilySurface`` consumer discovery."""

    @staticmethod
    def test_distribution_with_family_prefix_is_discovered() -> None:
        """A family-prefixed name is a candidate with no requirement edge."""
        candidate = FlextUtilitiesFamilySurface.distribution_in_family_discovery(
            "flext_cli",
            (),
        )
        tm.that(candidate, eq=True)

    @staticmethod
    def test_consumer_with_family_requirement_is_discovered() -> None:
        """A requirement on a family member makes the consumer a candidate."""
        candidate = FlextUtilitiesFamilySurface.distribution_in_family_discovery(
            "cosmos_main",
            ("pydantic", "flext_cli"),
        )
        tm.that(candidate, eq=True)

    @staticmethod
    def test_unrelated_distribution_is_not_discovered() -> None:
        """No family prefix and no family requirement keeps a dist outside."""
        candidate = FlextUtilitiesFamilySurface.distribution_in_family_discovery(
            "probe_consumer_surface",
            ("pydantic",),
        )
        tm.that(candidate, eq=False)

    @staticmethod
    def test_consumer_without_any_requirement_is_not_discovered() -> None:
        """A dependency-free dist joins only through its own family prefix."""
        candidate = FlextUtilitiesFamilySurface.distribution_in_family_discovery(
            "cosmos_main",
            (),
        )
        tm.that(candidate, eq=False)
