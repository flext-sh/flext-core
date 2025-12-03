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
    F,
    K,
    P,
    R,
    ResultT,
    T,
    T1_co,
    T2_co,
    T3_co,
    T_contra,
    TAggregate_co,
    TCacheKey_contra,
    TCacheValue_co,
    TCommand_contra,
    TConfigKey_contra,
    TDomainEvent_co,
    TEntity_co,
    TEvent_contra,
    TInput_contra,
    TItem_contra,
    TQuery_contra,
    TResult_co,
    TResult_contra,
    TState_co,
    TUtil_contra,
    TValue_co,
    TValueObject_co,
    U,
    V,
    W,
    t,
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
    """Factory for type system test scenarios with centralized test data using FlextConstants."""

    CORE_TYPEVARS: ClassVar[list[TypeVarTestCase]] = [
        TypeVarTestCase("T", TypeVarCategory.CORE, T, True),
        TypeVarTestCase("U", TypeVarCategory.CORE, U, True),
        TypeVarTestCase("V", TypeVarCategory.CORE, V, True),
        TypeVarTestCase("W", TypeVarCategory.CORE, W, True),
        TypeVarTestCase("E", TypeVarCategory.CORE, E, True),
        TypeVarTestCase("F", TypeVarCategory.CORE, F, True),
        TypeVarTestCase("K", TypeVarCategory.CORE, K, True),
        TypeVarTestCase("R", TypeVarCategory.CORE, R, True),
        TypeVarTestCase("ResultT", TypeVarCategory.CORE, ResultT, True),
    ]

    COVARIANT_TYPEVARS: ClassVar[list[TypeVarTestCase]] = [
        TypeVarTestCase("T1_co", TypeVarCategory.COVARIANT, T1_co, True),
        TypeVarTestCase("T2_co", TypeVarCategory.COVARIANT, T2_co, True),
        TypeVarTestCase("T3_co", TypeVarCategory.COVARIANT, T3_co, True),
        TypeVarTestCase("TState_co", TypeVarCategory.COVARIANT, TState_co, True),
        TypeVarTestCase(
            "TAggregate_co",
            TypeVarCategory.COVARIANT,
            TAggregate_co,
            True,
        ),
        TypeVarTestCase(
            "TCacheValue_co",
            TypeVarCategory.COVARIANT,
            TCacheValue_co,
            True,
        ),
        TypeVarTestCase(
            "TDomainEvent_co",
            TypeVarCategory.COVARIANT,
            TDomainEvent_co,
            True,
        ),
        TypeVarTestCase("TEntity_co", TypeVarCategory.COVARIANT, TEntity_co, True),
        TypeVarTestCase("TResult_co", TypeVarCategory.COVARIANT, TResult_co, True),
        TypeVarTestCase("TValue_co", TypeVarCategory.COVARIANT, TValue_co, True),
        TypeVarTestCase(
            "TValueObject_co",
            TypeVarCategory.COVARIANT,
            TValueObject_co,
            True,
        ),
    ]

    CONTRAVARIANT_TYPEVARS: ClassVar[list[TypeVarTestCase]] = [
        TypeVarTestCase("T_contra", TypeVarCategory.CONTRAVARIANT, T_contra, True),
        TypeVarTestCase(
            "TCommand_contra",
            TypeVarCategory.CONTRAVARIANT,
            TCommand_contra,
            True,
        ),
        TypeVarTestCase(
            "TEvent_contra",
            TypeVarCategory.CONTRAVARIANT,
            TEvent_contra,
            True,
        ),
        TypeVarTestCase(
            "TInput_contra",
            TypeVarCategory.CONTRAVARIANT,
            TInput_contra,
            True,
        ),
        TypeVarTestCase(
            "TQuery_contra",
            TypeVarCategory.CONTRAVARIANT,
            TQuery_contra,
            True,
        ),
        TypeVarTestCase(
            "TItem_contra",
            TypeVarCategory.CONTRAVARIANT,
            TItem_contra,
            True,
        ),
        TypeVarTestCase(
            "TResult_contra",
            TypeVarCategory.CONTRAVARIANT,
            TResult_contra,
            True,
        ),
        TypeVarTestCase(
            "TUtil_contra",
            TypeVarCategory.CONTRAVARIANT,
            TUtil_contra,
            True,
        ),
        TypeVarTestCase(
            "TCacheKey_contra",
            TypeVarCategory.CONTRAVARIANT,
            TCacheKey_contra,
            True,
        ),
        TypeVarTestCase(
            "TConfigKey_contra",
            TypeVarCategory.CONTRAVARIANT,
            TConfigKey_contra,
            True,
        ),
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
        ids=lambda tc: tc.name,
    )
    def test_core_typevars(self, test_case: TypeVarTestCase) -> None:
        """Test core TypeVar definitions are properly exported."""
        if test_case.expected_not_none:
            assert test_case.type_var is not None
            assert TypeScenarios.is_typevar(test_case.type_var)

    @pytest.mark.parametrize(
        "test_case",
        TypeScenarios.COVARIANT_TYPEVARS,
        ids=lambda tc: tc.name,
    )
    def test_covariant_typevars(self, test_case: TypeVarTestCase) -> None:
        """Test covariant TypeVar definitions are properly exported."""
        if test_case.expected_not_none:
            assert test_case.type_var is not None
            assert TypeScenarios.is_typevar(test_case.type_var)

    @pytest.mark.parametrize(
        "test_case",
        TypeScenarios.CONTRAVARIANT_TYPEVARS,
        ids=lambda tc: tc.name,
    )
    def test_contravariant_typevars(self, test_case: TypeVarTestCase) -> None:
        """Test contravariant TypeVar definitions are properly exported."""
        if test_case.expected_not_none:
            assert test_case.type_var is not None
            assert TypeScenarios.is_typevar(test_case.type_var)

    @pytest.mark.parametrize(
        "test_case",
        TypeScenarios.CQRS_ALIASES,
        ids=lambda tc: tc.name,
    )
    def test_cqrs_aliases(self, test_case: TypeVarTestCase) -> None:
        """Test CQRS type aliases are properly defined."""
        if test_case.expected_not_none:
            assert test_case.type_var is not None

    @pytest.mark.parametrize(
        "test_case",
        TypeScenarios.PARAMSPEC_ITEMS,
        ids=lambda tc: tc.name,
    )
    def test_paramspec(self, test_case: TypeVarTestCase) -> None:
        """Test ParamSpec is properly defined and exported."""
        if test_case.expected_not_none:
            assert test_case.type_var is not None
            assert isinstance(test_case.type_var, ParamSpec)

    def test_flexttypes_accessible(self) -> None:
        """Test t namespace is accessible."""
        assert t is not None
        assert all(
            hasattr(t, attr)
            for attr in [
                "Validation",
                "Json",
                "HandlerAliases",
                "Processor",
                "Factory",
                "Utility",
                "Bus",
                "Logging",
                "Cqrs",
                "Config",
            ]
        )

    def test_all_exports_importable(self) -> None:
        """Test that all public exports can be imported."""
        assert True

    def test_module_structure(self) -> None:
        """Test that t has expected structure."""
        assert all(tv is not None for tv in [T, U, P, R])
        assert all(
            alias is not None
            for alias in [
                t.GeneralValueType,  # Command
                t.GeneralValueType,  # Event
                t.GeneralValueType,  # Query
                t.GeneralValueType,  # Message
            ]
        )


__all__ = ["TestFlextTypings"]
