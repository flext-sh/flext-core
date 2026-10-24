"""Behavioral tests for FlextUtilitiesBeartypeEngine annotation inspection.

These assert the OBSERVABLE CONTRACT of the public static methods only:
their return values for representative type hints. No private attribute or
method of the engine is accessed; the engine is exercised exclusively through
its public surface (``contains_any``, ``has_forbidden_collection_origin``,
``count_union_members``, ``matches_str_none_union``, ``alias_contains_any``).

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, TypeAliasType

import pytest

import tests.utilities
from tests import c, m, t, tm, u
from tests.unit._enforcement_support import invalid_annotations

if TYPE_CHECKING:
    from pathlib import PurePath as LazyResolvableType


class TestsFlextBeartypeEngineAnnotations(
    tests.utilities.TestsFlextUtilities.TestsFlextBeartypeEngine,
):
    """Contract of the beartype annotation-inspection engine."""

    @staticmethod
    @pytest.mark.parametrize(
        ("hint", "expected"),
        [
            (str, False),
            (int, False),
            (str | None, False),
            (t.IntMapping, False),
            (t.StrSequence, False),
            (t.JsonPayload, False),
            (t.ContextHookMap, False),
            (t.MappingKV[str, t.JsonPayload], False),
            (Mapping[str, t.SequenceOf[int]], False),
            (None, False),
        ],
    )
    def test_contains_any_reports_presence_of_any(
        hint: t.TypeHintSpecifier | None,
        *,
        expected: bool,
    ) -> None:
        """contains_any is True iff typing.Any appears at any nesting depth."""
        tm.that(u.contains_any(hint) is expected, truth=True)

    @staticmethod
    @pytest.mark.parametrize("field_name", ["hooks", "scope_vars"])
    def test_loaded_context_fields_retain_their_strict_payload_contract(
        field_name: str,
    ) -> None:
        """The actual Pydantic field must retain the declaration's no-Any contract."""
        annotation = m.ContextRuntimeState.model_fields[field_name].annotation
        tm.that(u.contains_any(annotation), eq=False, msg=repr(annotation))

    @staticmethod
    @pytest.mark.parametrize(
        "alias_name",
        [
            "BareAnyAlias",
            "NestedAnyAlias",
            "NestedAnySequenceAlias",
            "AnyAlias",
            "OptionalAnyAlias",
        ],
    )
    def test_contains_any_rejects_invalid_source_hints(
        tmp_path: Path, alias_name: str
    ) -> None:
        """Invalid annotations are source inputs, never owned type declarations."""
        with invalid_annotations(tmp_path) as source:
            alias = getattr(source.InvalidTypes, alias_name)
            if not isinstance(alias, TypeAliasType):
                pytest.fail("Invalid source fixture is not a native alias")
            tm.ok(u.contains_any(alias.__value__))

    @pytest.mark.parametrize(
        ("hint", "expected"),
        [
            (dict[str, int], (True, "dict")),
            (list[str], (True, c.Action.LIST)),
            (set[int], (True, "set")),
            (t.IntMapping, (False, "")),
            (t.StrSequence, (False, "")),
            (str, (False, "")),
            (None, (False, "")),
        ],
    )
    def test_has_forbidden_collection_origin_names_offending_origin(
        self,
        hint: t.TypeHintSpecifier | None,
        expected: tuple[bool, str],
    ) -> None:
        """Bare mutable collection origins are flagged with their name."""
        tm.that(u.has_forbidden_collection_origin(hint, self.FORBIDDEN), eq=expected)

    @staticmethod
    @pytest.mark.parametrize(
        ("hint", "expected"),
        [
            (str | int, 2),
            (str | int | float, 3),
            (str | None, 1),
            (str | int | float | None, 3),
            (str, 0),
            (None, 0),
        ],
    )
    def test_count_union_members_excludes_none(
        hint: t.TypeHintSpecifier | None,
        *,
        expected: int,
    ) -> None:
        """count_union_members counts non-None members; 0 for non-unions."""
        tm.that(u.count_union_members(hint), eq=expected)

    @staticmethod
    @pytest.mark.parametrize(
        ("hint", "expected"),
        [
            (str | None, True),
            (str | int | None, True),
            (str, False),
            (int | None, False),
            (str | int, False),
            (None, False),
        ],
    )
    def test_matches_str_none_union_requires_str_and_none(
        hint: t.TypeHintSpecifier | None,
        *,
        expected: bool,
    ) -> None:
        """Union matches iff both str and None are members."""
        tm.that(u.matches_str_none_union(hint) is expected, truth=True)

    @staticmethod
    @pytest.mark.parametrize(
        ("alias_name", "expected"),
        [
            ("AnyAlias", True),
            ("NestedAnyAlias", True),
            ("CleanAlias", False),
            (None, False),
        ],
    )
    def test_alias_contains_any_unwraps_alias_values(
        tmp_path: Path,
        alias_name: str | None,
        *,
        expected: bool,
    ) -> None:
        """alias_contains_any detects Any inside a resolved type-alias value."""
        with invalid_annotations(tmp_path) as source:
            if alias_name is None:
                tm.that(u.alias_contains_any(None), eq=expected)
            else:
                alias = getattr(source.InvalidTypes, alias_name)
                if not isinstance(alias, TypeAliasType):
                    pytest.fail("Source fixture did not declare a native type alias")
                tm.that(u.alias_contains_any(alias.__value__), eq=expected)

    @staticmethod
    def test_alias_inspection_uses_explicit_owner_for_static_only_names() -> None:
        """Explicit owner context proves a local static-only alias deferral."""

        class Host:
            type LazyAlias = LazyResolvableType | str

        with pytest.raises(NameError) as native:
            _ = Host.LazyAlias.__value__
        tm.that(native.value.name, eq="LazyResolvableType")

        tm.that(u.resolve_type_alias_value(Host.LazyAlias, owner=Host), none=True)
        tm.that(u.alias_contains_any(Host.LazyAlias, owner=Host) is False, truth=True)
        deferred = u.deferred_alias(Host.LazyAlias, owner=Host)
        tm.that(deferred, none=False)
        deferred = tm.not_none(deferred)
        tm.that(deferred.module, eq=Host.__module__)
        tm.that(deferred.qualname, eq=f"{Host.__qualname__}.LazyAlias")
        tm.that(deferred.unavailable_imports, eq=("LazyResolvableType",))

        with pytest.raises(NameError) as still_lazy:
            _ = Host.LazyAlias.__value__
        tm.that(still_lazy.value.name, eq=native.value.name)

    @staticmethod
    def test_local_alias_rejects_a_shadowed_static_import_guard(tmp_path: Path) -> None:
        """A local flag with the same name is not typing's static-only guard."""
        with invalid_annotations(tmp_path) as source:
            owner = source.shadowed_guard_owner()
            if not isinstance(owner, type):
                pytest.fail("Source fixture did not declare a class owner")
            alias = vars(owner)["LazyAlias"]
            if not isinstance(alias, TypeAliasType):
                pytest.fail("Source fixture did not declare a native type alias")
            tm.that(u.deferred_alias(alias, owner=owner), none=True)
            with pytest.raises(NameError):
                _ = alias.__value__
            with pytest.raises(NameError) as resolved:
                u.resolve_type_alias_value(alias, owner=owner)
            tm.that(resolved.value.name, eq="GuardedPath")

    @staticmethod
    def test_local_alias_rejects_a_copied_owner_with_forged_qualname() -> None:
        """An alias binding and a copied qualname do not prove a declaration."""

        class Host:
            type LazyAlias = LazyResolvableType | str

        class Copied:
            LazyAlias = Host.LazyAlias

        Copied.__qualname__ = Host.__qualname__
        tm.that(u.deferred_alias(Host.LazyAlias, owner=Copied), none=True)
        with pytest.raises(NameError) as raised:
            u.resolve_type_alias_value(Host.LazyAlias, owner=Copied)
        tm.that(raised.value.name, eq="LazyResolvableType")

    @staticmethod
    def test_local_alias_requires_the_exact_owner_binding() -> None:
        """A homonymous native declaration cannot authenticate another alias."""

        class Host:
            type LazyAlias = LazyResolvableType | str

        class Other:
            type LazyAlias = LazyResolvableType | str

        Other.__qualname__ = Host.__qualname__
        tm.that(u.deferred_alias(Host.LazyAlias, owner=Other), none=True)
        with pytest.raises(NameError) as raised:
            u.resolve_type_alias_value(Host.LazyAlias, owner=Other)
        tm.that(raised.value.name, eq="LazyResolvableType")

    @staticmethod
    def test_local_alias_without_owner_keeps_its_native_name_error() -> None:
        """Source discovery never substitutes for a missing local class owner."""

        class Host:
            type LazyAlias = LazyResolvableType | str

        with pytest.raises(NameError) as raised:
            u.resolve_type_alias_value(Host.LazyAlias)
        tm.that(raised.value.name, eq="LazyResolvableType")

    @staticmethod
    def test_local_alias_payload_strings_are_not_static_import_dependencies() -> None:
        """Literal values and Annotated metadata remain data, not type lookups."""

        class Host:
            type Label = Literal["LazyResolvableType"]
            type Labeled = Annotated[str, "LazyResolvableType"]

        for alias in (Host.Label, Host.Labeled):
            tm.that(u.deferred_alias(alias, owner=Host), none=True)
            tm.that(
                u.resolve_type_alias_value(alias, owner=Host) is alias.__value__,
                truth=True,
            )
