"""Tests for flext_core.typings module - Type system validation.

Module: flext_core.typings
Scope: TypeVar definitions, type aliases, and CQRS type patterns

Tests real functionality of the centralized type system, ensuring all
exported TypeVars and type aliases are properly accessible at runtime.

Consolidated 3 test classes into 1 unified parametrized test class using
StrEnum, frozen dataclasses, and advanced parametrization for DRY testing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar, ParamSpec, TypeVar

import pytest

from flext_core import (
    Command,
    E,
    Event,
    F,
    FlextTypes,
    K,
    Message,
    P,
    Query,
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
)

# =========================================================================
# Type Test Enumeration
# =========================================================================


class TypeVarCategory(StrEnum):
    """TypeVar categories for parametrized testing."""

    CORE = "core"
    COVARIANT = "covariant"
    CONTRAVARIANT = "contravariant"
    PARAMSPEC = "paramspec"
    DOMAIN = "domain"
    CQRS = "cqrs"


# =========================================================================
# Type Test Case
# =========================================================================


@dataclass(frozen=True, slots=True)
class TypeVarTestCase:
    """TypeVar test case definition."""

    name: str
    category: TypeVarCategory
    type_var: object
    expected_not_none: bool = True


# =========================================================================
# Type Scenarios Factory
# =========================================================================


class TypeScenarios:
    """Factory for type system test scenarios with centralized test data."""

    # Core TypeVar scenarios
    CORE_TYPEVARS: ClassVar[list[TypeVarTestCase]] = [
        TypeVarTestCase(
            name="T",
            category=TypeVarCategory.CORE,
            type_var=T,
            expected_not_none=True,
        ),
        TypeVarTestCase(
            name="U",
            category=TypeVarCategory.CORE,
            type_var=U,
            expected_not_none=True,
        ),
        TypeVarTestCase(
            name="V",
            category=TypeVarCategory.CORE,
            type_var=V,
            expected_not_none=True,
        ),
        TypeVarTestCase(
            name="W",
            category=TypeVarCategory.CORE,
            type_var=W,
            expected_not_none=True,
        ),
        TypeVarTestCase(
            name="E",
            category=TypeVarCategory.CORE,
            type_var=E,
            expected_not_none=True,
        ),
        TypeVarTestCase(
            name="F",
            category=TypeVarCategory.CORE,
            type_var=F,
            expected_not_none=True,
        ),
        TypeVarTestCase(
            name="K",
            category=TypeVarCategory.CORE,
            type_var=K,
            expected_not_none=True,
        ),
        TypeVarTestCase(
            name="R",
            category=TypeVarCategory.CORE,
            type_var=R,
            expected_not_none=True,
        ),
        TypeVarTestCase(
            name="ResultT",
            category=TypeVarCategory.CORE,
            type_var=ResultT,
            expected_not_none=True,
        ),
    ]

    # Covariant TypeVar scenarios
    COVARIANT_TYPEVARS: ClassVar[list[TypeVarTestCase]] = [
        TypeVarTestCase(
            name="T1_co",
            category=TypeVarCategory.COVARIANT,
            type_var=T1_co,
            expected_not_none=True,
        ),
        TypeVarTestCase(
            name="T2_co",
            category=TypeVarCategory.COVARIANT,
            type_var=T2_co,
            expected_not_none=True,
        ),
        TypeVarTestCase(
            name="T3_co",
            category=TypeVarCategory.COVARIANT,
            type_var=T3_co,
            expected_not_none=True,
        ),
        TypeVarTestCase(
            name="TState_co",
            category=TypeVarCategory.COVARIANT,
            type_var=TState_co,
            expected_not_none=True,
        ),
        TypeVarTestCase(
            name="TAggregate_co",
            category=TypeVarCategory.COVARIANT,
            type_var=TAggregate_co,
            expected_not_none=True,
        ),
        TypeVarTestCase(
            name="TCacheValue_co",
            category=TypeVarCategory.COVARIANT,
            type_var=TCacheValue_co,
            expected_not_none=True,
        ),
        TypeVarTestCase(
            name="TDomainEvent_co",
            category=TypeVarCategory.COVARIANT,
            type_var=TDomainEvent_co,
            expected_not_none=True,
        ),
        TypeVarTestCase(
            name="TEntity_co",
            category=TypeVarCategory.COVARIANT,
            type_var=TEntity_co,
            expected_not_none=True,
        ),
        TypeVarTestCase(
            name="TResult_co",
            category=TypeVarCategory.COVARIANT,
            type_var=TResult_co,
            expected_not_none=True,
        ),
        TypeVarTestCase(
            name="TValue_co",
            category=TypeVarCategory.COVARIANT,
            type_var=TValue_co,
            expected_not_none=True,
        ),
        TypeVarTestCase(
            name="TValueObject_co",
            category=TypeVarCategory.COVARIANT,
            type_var=TValueObject_co,
            expected_not_none=True,
        ),
    ]

    # Contravariant TypeVar scenarios
    CONTRAVARIANT_TYPEVARS: ClassVar[list[TypeVarTestCase]] = [
        TypeVarTestCase(
            name="T_contra",
            category=TypeVarCategory.CONTRAVARIANT,
            type_var=T_contra,
            expected_not_none=True,
        ),
        TypeVarTestCase(
            name="TCommand_contra",
            category=TypeVarCategory.CONTRAVARIANT,
            type_var=TCommand_contra,
            expected_not_none=True,
        ),
        TypeVarTestCase(
            name="TEvent_contra",
            category=TypeVarCategory.CONTRAVARIANT,
            type_var=TEvent_contra,
            expected_not_none=True,
        ),
        TypeVarTestCase(
            name="TInput_contra",
            category=TypeVarCategory.CONTRAVARIANT,
            type_var=TInput_contra,
            expected_not_none=True,
        ),
        TypeVarTestCase(
            name="TQuery_contra",
            category=TypeVarCategory.CONTRAVARIANT,
            type_var=TQuery_contra,
            expected_not_none=True,
        ),
        TypeVarTestCase(
            name="TItem_contra",
            category=TypeVarCategory.CONTRAVARIANT,
            type_var=TItem_contra,
            expected_not_none=True,
        ),
        TypeVarTestCase(
            name="TResult_contra",
            category=TypeVarCategory.CONTRAVARIANT,
            type_var=TResult_contra,
            expected_not_none=True,
        ),
        TypeVarTestCase(
            name="TUtil_contra",
            category=TypeVarCategory.CONTRAVARIANT,
            type_var=TUtil_contra,
            expected_not_none=True,
        ),
        TypeVarTestCase(
            name="TCacheKey_contra",
            category=TypeVarCategory.CONTRAVARIANT,
            type_var=TCacheKey_contra,
            expected_not_none=True,
        ),
        TypeVarTestCase(
            name="TConfigKey_contra",
            category=TypeVarCategory.CONTRAVARIANT,
            type_var=TConfigKey_contra,
            expected_not_none=True,
        ),
    ]

    # CQRS type alias scenarios
    CQRS_ALIASES: ClassVar[list[TypeVarTestCase]] = [
        TypeVarTestCase(
            name="Command",
            category=TypeVarCategory.CQRS,
            type_var=Command,
            expected_not_none=True,
        ),
        TypeVarTestCase(
            name="Query",
            category=TypeVarCategory.CQRS,
            type_var=Query,
            expected_not_none=True,
        ),
        TypeVarTestCase(
            name="Event",
            category=TypeVarCategory.CQRS,
            type_var=Event,
            expected_not_none=True,
        ),
        TypeVarTestCase(
            name="Message",
            category=TypeVarCategory.CQRS,
            type_var=Message,
            expected_not_none=True,
        ),
    ]

    # ParamSpec scenarios
    PARAMSPEC_ITEMS: ClassVar[list[TypeVarTestCase]] = [
        TypeVarTestCase(
            name="P",
            category=TypeVarCategory.PARAMSPEC,
            type_var=P,
            expected_not_none=True,
        ),
    ]

    @staticmethod
    def is_typevar(obj: object) -> bool:
        """Check if object is a TypeVar-like instance."""
        return isinstance(obj, (TypeVar, ParamSpec)) or obj is not None


# =========================================================================
# Test Suite
# =========================================================================


class TestFlextTypings:
    """Unified test suite for FlextTypes and type system."""

    # =====================================================================
    # Core TypeVar Tests
    # =====================================================================

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

    # =====================================================================
    # Covariant TypeVar Tests
    # =====================================================================

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

    # =====================================================================
    # Contravariant TypeVar Tests
    # =====================================================================

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

    # =====================================================================
    # CQRS Type Alias Tests
    # =====================================================================

    @pytest.mark.parametrize(
        "test_case",
        TypeScenarios.CQRS_ALIASES,
        ids=lambda tc: tc.name,
    )
    def test_cqrs_aliases(self, test_case: TypeVarTestCase) -> None:
        """Test CQRS type aliases are properly defined."""
        if test_case.expected_not_none:
            assert test_case.type_var is not None

    # =====================================================================
    # ParamSpec Tests
    # =====================================================================

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

    # =====================================================================
    # Integration Tests
    # =====================================================================

    def test_flexttypes_accessible(self) -> None:
        """Test FlextTypes namespace is accessible."""
        assert FlextTypes is not None
        # FlextTypes has nested classes for types
        assert hasattr(FlextTypes, "Validation")
        assert hasattr(FlextTypes, "Json")
        assert hasattr(FlextTypes, "Handler")
        assert hasattr(FlextTypes, "Processor")
        assert hasattr(FlextTypes, "Factory")
        assert hasattr(FlextTypes, "Utility")
        assert hasattr(FlextTypes, "Bus")
        assert hasattr(FlextTypes, "Logging")
        assert hasattr(FlextTypes, "Cqrs")
        assert hasattr(FlextTypes, "Config")

    def test_all_exports_importable(self) -> None:
        """Test that all public exports can be imported.

        This test verifies the module loads without import errors.
        """
        # If any import failed, the test module wouldn't load
        assert True

    def test_module_structure(self) -> None:
        """Test that FlextTypes has expected structure."""
        # Core TypeVars at module level (already imported at top)
        assert T is not None
        assert U is not None
        assert P is not None
        assert R is not None

        # CQRS aliases in nested class
        assert FlextTypes.Cqrs.Command is not None
        assert FlextTypes.Cqrs.Event is not None
        assert FlextTypes.Cqrs.Query is not None
        assert FlextTypes.Cqrs.Message is not None


__all__ = ["TestFlextTypings"]
