"""Typing facade alias tests."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest
from flext_tests import tm

import flext_core as core
from tests import t
from tests.unit._typings_support import FLAT_ALIAS_NAMES, PUBLIC_ALIAS_NAMES


class TestsFlextTypesAliases:
    @pytest.mark.parametrize("alias_name", PUBLIC_ALIAS_NAMES)
    def test_public_alias_accessible(self, alias_name: str) -> None:
        """Public type aliases are accessible through t.* namespace."""
        tm.that(hasattr(t, alias_name), eq=True)

    def test_public_generic_exports_stay_absent(self) -> None:
        """flext_core must not expose shared TypeVar or ParamSpec helpers."""
        for legacy_name in (
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
        ):
            tm.that(hasattr(core, legacy_name), eq=False)

    def test_primitives_types_tuple(self) -> None:
        """PRIMITIVES_TYPES contains str, int, float, bool."""
        tm.that(t.PRIMITIVES_TYPES, eq=t.PRIMITIVES_TYPES)

    def test_numeric_types_tuple(self) -> None:
        """NUMERIC_TYPES contains int, float."""
        tm.that(t.NUMERIC_TYPES, eq=(int, float))

    def test_scalar_types_tuple(self) -> None:
        """SCALAR_TYPES contains str, int, float, bool, datetime."""
        tm.that(t.SCALAR_TYPES, eq=(str, int, float, bool, datetime))

    def test_container_types_tuple(self) -> None:
        """CONTAINER_TYPES contains scalar types + Path."""
        tm.that(t.CONTAINER_TYPES, eq=(str, int, float, bool, datetime, Path))

    def test_container_and_collection_types_tuple(self) -> None:
        """CONTAINER_AND_COLLECTION_TYPES includes list, dict, tuple."""
        tm.that(list in t.CONTAINER_AND_COLLECTION_TYPES, eq=True)
        tm.that(dict in t.CONTAINER_AND_COLLECTION_TYPES, eq=True)
        tm.that(tuple in t.CONTAINER_AND_COLLECTION_TYPES, eq=True)

    @pytest.mark.parametrize("alias_name", FLAT_ALIAS_NAMES)
    def test_flat_alias_accessible(self, alias_name: str) -> None:
        """Flat mapping type aliases are accessible through t.*."""
        tm.that(hasattr(t, alias_name), eq=True)
