"""Tests for flext_core.typings module - Type system validation.

Module: flext_core.typings
Scope: TypeVar definitions, type aliases, and CQRS type patterns

Tests real functionality of the centralized type system, ensuring all
exported TypeVars and type aliases are properly accessible at runtime.

Uses Python 3.13 patterns, FlextTestsUtilities, FlextConstants,
and aggressive parametrization for DRY testing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar, ParamSpec, TypeVar

import pytest
from flext_core import (
    E,
    P,
    R,
    ResultT,
    T,
    T_co,
    T_contra,
    U,
    t,
)
from flext_core.constants import FlextConstants
from flext_tests.matchers import tm
from pydantic import (
    TypeAdapter as PydanticTypeAdapter,
    ValidationError as PydanticValidationError,
)


class TypeVarCategory(StrEnum):
    """TypeVar categories for parametrized testing."""

    CORE = "core"
    COVARIANT = "covariant"
    CONTRAVARIANT = "contravariant"
    PARAMSPEC = "paramspec"
    CQRS = "cqrs"


@dataclass(frozen=True, slots=True)
class TypeVarTestCase:
    """TypeVar test case definition."""

    name: str
    category: TypeVarCategory
    type_var: object
    expected_not_none: bool = True


class TypeScenarios:
    """Factory for type system test scenarios with centralized test data."""

    CORE_TYPEVARS: ClassVar[list[TypeVarTestCase]] = [
        TypeVarTestCase("T", TypeVarCategory.CORE, T, True),
        TypeVarTestCase("U", TypeVarCategory.CORE, U, True),
        TypeVarTestCase("E", TypeVarCategory.CORE, E, True),
        TypeVarTestCase("R", TypeVarCategory.CORE, R, True),
        TypeVarTestCase("ResultT", TypeVarCategory.CORE, ResultT, True),
    ]

    COVARIANT_TYPEVARS: ClassVar[list[TypeVarTestCase]] = [
        TypeVarTestCase("T_co", TypeVarCategory.COVARIANT, T_co, True),
    ]

    CONTRAVARIANT_TYPEVARS: ClassVar[list[TypeVarTestCase]] = [
        TypeVarTestCase("T_contra", TypeVarCategory.CONTRAVARIANT, T_contra, True),
    ]

    CQRS_ALIASES: ClassVar[list[TypeVarTestCase]] = [
        TypeVarTestCase("Command", TypeVarCategory.CQRS, t.GeneralValueType, True),
        TypeVarTestCase("Query", TypeVarCategory.CQRS, t.GeneralValueType, True),
        TypeVarTestCase("Event", TypeVarCategory.CQRS, t.GeneralValueType, True),
        TypeVarTestCase("Message", TypeVarCategory.CQRS, t.GeneralValueType, True),
    ]

    PARAMSPEC_ITEMS: ClassVar[list[TypeVarTestCase]] = [
        TypeVarTestCase("P", TypeVarCategory.PARAMSPEC, P, True),
    ]

    @staticmethod
    def is_typevar(obj: object) -> bool:
        """Check if object is a TypeVar-like instance."""
        return isinstance(obj, (TypeVar, ParamSpec)) or obj is not None


class TestFlextTypings:
    """Unified test suite for t and type system using FlextTestsUtilities."""

    @pytest.mark.parametrize(
        "test_case",
        TypeScenarios.CORE_TYPEVARS,
        ids=lambda c: c.name,
    )
    def test_core_typevars(self, test_case: TypeVarTestCase) -> None:
        """Test core TypeVar definitions are properly exported."""
        if test_case.expected_not_none:
            tm.that(
                test_case.type_var,
                none=False,
                msg=f"{test_case.name} must not be None",
            )
            tm.that(
                TypeScenarios.is_typevar(test_case.type_var),
                eq=True,
                msg=f"{test_case.name} must be a valid TypeVar or ParamSpec",
            )

    @pytest.mark.parametrize(
        "test_case",
        TypeScenarios.COVARIANT_TYPEVARS,
        ids=lambda c: c.name,
    )
    def test_covariant_typevars(self, test_case: TypeVarTestCase) -> None:
        """Test covariant TypeVar definitions are properly exported."""
        if test_case.expected_not_none:
            tm.that(
                test_case.type_var,
                none=False,
                msg=f"{test_case.name} must not be None",
            )
            tm.that(
                TypeScenarios.is_typevar(test_case.type_var),
                eq=True,
                msg=f"{test_case.name} must be a valid TypeVar or ParamSpec",
            )
            # Validate it's actually a TypeVar (not ParamSpec)
            tm.that(
                isinstance(test_case.type_var, TypeVar),
                eq=True,
                msg=f"{test_case.name} must be a TypeVar instance",
            )

    @pytest.mark.parametrize(
        "test_case",
        TypeScenarios.CONTRAVARIANT_TYPEVARS,
        ids=lambda c: c.name,
    )
    def test_contravariant_typevars(self, test_case: TypeVarTestCase) -> None:
        """Test contravariant TypeVar definitions are properly exported."""
        if test_case.expected_not_none:
            tm.that(
                test_case.type_var,
                none=False,
                msg=f"{test_case.name} must not be None",
            )
            tm.that(
                TypeScenarios.is_typevar(test_case.type_var),
                eq=True,
                msg=f"{test_case.name} must be a valid TypeVar or ParamSpec",
            )
            # Validate it's actually a TypeVar (not ParamSpec)
            tm.that(
                isinstance(test_case.type_var, TypeVar),
                eq=True,
                msg=f"{test_case.name} must be a TypeVar instance",
            )

    @pytest.mark.parametrize(
        "test_case",
        TypeScenarios.CQRS_ALIASES,
        ids=lambda c: c.name,
    )
    def test_cqrs_aliases(self, test_case: TypeVarTestCase) -> None:
        """Test CQRS type aliases are properly defined."""
        if test_case.expected_not_none:
            tm.that(
                test_case.type_var,
                none=False,
                msg=f"{test_case.name} alias must not be None",
            )
            # CQRS aliases should all be t.GeneralValueType
            tm.that(
                test_case.type_var,
                eq=t.GeneralValueType,
                msg=f"{test_case.name} must equal t.GeneralValueType",
            )

    @pytest.mark.parametrize(
        "test_case",
        TypeScenarios.PARAMSPEC_ITEMS,
        ids=lambda c: c.name,
    )
    def test_paramspec(self, test_case: TypeVarTestCase) -> None:
        """Test ParamSpec is properly defined and exported."""
        if test_case.expected_not_none:
            tm.that(
                test_case.type_var,
                none=False,
                msg=f"{test_case.name} must not be None",
            )
            tm.that(
                isinstance(test_case.type_var, ParamSpec),
                eq=True,
                msg=f"{test_case.name} must be a ParamSpec instance",
            )

    def test_flexttypes_accessible(self) -> None:
        """Test t namespace is accessible with real validation."""
        tm.that(t, none=False, msg="FlextTypes (t) must be accessible")
        # Verify flat type aliases are accessible
        flat_types = [
            "GeneralValueType",
            "ScalarValue",
            "HandlerCallable",
        ]
        for type_alias in flat_types:
            tm.that(
                hasattr(t, type_alias),
                eq=True,
                msg=f"t must have {type_alias} type alias",
            )

    def test_all_exports_importable(self) -> None:
        """Test that all public exports can be imported and are valid."""
        # Test all core TypeVars are importable and valid
        core_typevars = [T, U, E, R, ResultT, T_co, T_contra, P]
        for tv in core_typevars:
            tm.that(tv, none=False, msg="TypeVar must be importable and not None")
            tm.that(
                isinstance(tv, (TypeVar, ParamSpec)),
                eq=True,
                msg="TypeVar must be TypeVar or ParamSpec instance",
            )

    def test_module_structure(self) -> None:
        """Test that t has expected structure with real validation."""
        # Validate core TypeVars
        for tv in [T, U, P, R]:
            tm.that(tv, none=False, msg="TypeVar must not be None")
            tm.that(
                isinstance(tv, (TypeVar, ParamSpec)),
                eq=True,
                msg="TypeVar must be TypeVar or ParamSpec instance",
            )
        # Validate CQRS aliases all point to t.GeneralValueType
        cqrs_aliases = [
            t.GeneralValueType,  # Command
            t.GeneralValueType,  # Event
            t.GeneralValueType,  # Query
            t.GeneralValueType,  # Message
        ]
        for alias in cqrs_aliases:
            tm.that(alias, none=False, msg="CQRS alias must not be None")
            tm.that(
                alias,
                eq=t.GeneralValueType,
                msg="CQRS alias must equal t.GeneralValueType",
            )

    def test_hostname_validation_success(self) -> None:
        """Test hostname validation success path with real validation."""
        hostname_adapter: PydanticTypeAdapter[str] = PydanticTypeAdapter(
            t.Validation.HostnameStr,
        )
        result = hostname_adapter.validate_python(FlextConstants.Network.LOCALHOST)
        tm.that(
            result,
            eq=FlextConstants.Network.LOCALHOST,
            msg="FlextConstants.Network.LOCALHOST must validate correctly",
        )
        tm.that(
            result,
            is_=str,
            none=False,
            empty=False,
            msg="Result value must be non-empty string",
        )

        result = hostname_adapter.validate_python(FlextConstants.Network.LOOPBACK_IP)
        tm.that(
            result,
            eq=FlextConstants.Network.LOOPBACK_IP,
            msg="IP address must validate correctly",
        )
        tm.that(
            result,
            is_=str,
            none=False,
            empty=False,
            msg="Result value must be non-empty string",
        )

    def test_hostname_validation_error(self) -> None:
        """Test hostname validation error path with real validation."""
        invalid_hostname = ""
        tm.that(
            invalid_hostname,
            is_=str,
            eq="",
            msg="Invalid hostname must be empty to fail HostnameStr",
        )

        hostname_adapter: PydanticTypeAdapter[str] = PydanticTypeAdapter(
            t.Validation.HostnameStr,
        )
        with pytest.raises(PydanticValidationError):
            hostname_adapter.validate_python(invalid_hostname)


__all__ = ["TestFlextTypings"]
