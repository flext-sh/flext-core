"""Behavioral tests for the tests typing facade (``t``) public surface.

Exercises the observable contract callers depend on:
- public/flat type aliases are reachable through the ``t.*`` namespace,
- ``flext_core`` keeps shared ``TypeVar`` / ``ParamSpec`` helpers out of its
  public surface,
- the runtime type-check tuples (``PRIMITIVES_TYPES`` etc.) expose the exact
  membership documented for them and actually classify values via
  ``isinstance``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from flext_tests import tm

import flext_core as core
from tests import t
from tests.unit._typings_support import FLAT_ALIAS_NAMES, PUBLIC_ALIAS_NAMES

LEGACY_GENERIC_NAMES: tuple[str, ...] = (
    "EnumT",
    "MessageT_contra",
    "P",
    "R",
    "ResultT",
    "RootValueT",
    "T",
    "T_DomainResult",
    "T_Model",
    "T_Namespace",
    "T_Settings",
    "T_co",
    "T_contra",
    "TRuntime",
    "U",
    "TV",
    "TV_co",
)


class TestsFlextCoreTypingsAliases:
    """Public contract of the ``t`` typing facade and its type-check tuples."""

    @pytest.mark.parametrize("alias_name", PUBLIC_ALIAS_NAMES)
    def test_public_alias_reachable_through_facade(self, alias_name: str) -> None:
        """Every declared public alias resolves to a real object on ``t``."""
        # Arrange / Act
        resolved = getattr(t, alias_name)
        # Assert - attribute exists and is a concrete (non-None) alias object.
        tm.that(resolved, ne=None)

    @pytest.mark.parametrize("alias_name", FLAT_ALIAS_NAMES)
    def test_flat_mapping_alias_reachable_through_facade(self, alias_name: str) -> None:
        """Every flat mapping alias resolves to a real object on ``t``."""
        resolved = getattr(t, alias_name)
        tm.that(resolved, ne=None)

    @pytest.mark.parametrize("legacy_name", LEGACY_GENERIC_NAMES)
    def test_flext_core_hides_shared_generic_helpers(self, legacy_name: str) -> None:
        """flext_core must not expose shared TypeVar/ParamSpec helpers publicly."""
        tm.that(hasattr(core, legacy_name), eq=False)

    def test_primitives_types_membership(self) -> None:
        """PRIMITIVES_TYPES is exactly (str, int, float, bool)."""
        tm.that(t.PRIMITIVES_TYPES, eq=(str, int, float, bool))

    def test_numeric_types_membership(self) -> None:
        """NUMERIC_TYPES is exactly (int, float)."""
        tm.that(t.NUMERIC_TYPES, eq=(int, float))

    def test_scalar_types_membership(self) -> None:
        """SCALAR_TYPES is exactly (str, int, float, bool, datetime)."""
        tm.that(t.SCALAR_TYPES, eq=(str, int, float, bool, datetime))

    def test_container_types_membership(self) -> None:
        """CONTAINER_TYPES extends the scalar set with Path."""
        tm.that(t.CONTAINER_TYPES, eq=(str, int, float, bool, datetime, Path))

    def test_container_and_collection_types_include_collections(self) -> None:
        """CONTAINER_AND_COLLECTION_TYPES adds list/dict/tuple to CONTAINER_TYPES."""
        tm.that(
            t.CONTAINER_AND_COLLECTION_TYPES,
            eq=(str, int, float, bool, datetime, Path, list, dict, tuple),
        )

    @pytest.mark.parametrize(
        ("value", "is_primitive", "is_numeric", "is_scalar"),
        [
            ("text", True, False, True),
            (7, True, True, True),
            (1.5, True, True, True),
            (True, True, True, True),
            (datetime(2026, 1, 1, tzinfo=UTC), False, False, True),
            (Path("/tmp"), False, False, False),
            ([1, 2], False, False, False),
        ],
    )
    def test_type_check_tuples_classify_values(
        self, value: object, is_primitive: bool, is_numeric: bool, is_scalar: bool
    ) -> None:
        """The type-check tuples classify values correctly via isinstance."""
        tm.that(isinstance(value, t.PRIMITIVES_TYPES), eq=is_primitive)
        tm.that(isinstance(value, t.NUMERIC_TYPES), eq=is_numeric)
        tm.that(isinstance(value, t.SCALAR_TYPES), eq=is_scalar)

    @pytest.mark.parametrize(
        "value",
        [
            "text",
            3,
            2.0,
            True,
            datetime(2026, 1, 1, tzinfo=UTC),
            Path("/tmp"),
            [1],
            {"a": 1},
            (1,),
        ],
    )
    def test_container_and_collection_tuple_accepts_every_container_value(
        self, value: object
    ) -> None:
        """Every documented container/collection value is recognised by the tuple."""
        tm.that(isinstance(value, t.CONTAINER_AND_COLLECTION_TYPES), eq=True)
