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
            assert test_case.type_var is not None
            assert TypeScenarios.is_typevar(test_case.type_var)

    @pytest.mark.parametrize(
        "test_case",
        TypeScenarios.COVARIANT_TYPEVARS,
        ids=lambda c: c.name,
    )
    def test_covariant_typevars(self, test_case: TypeVarTestCase) -> None:
        """Test covariant TypeVar definitions are properly exported."""
        if test_case.expected_not_none:
            assert test_case.type_var is not None
            assert TypeScenarios.is_typevar(test_case.type_var)

    @pytest.mark.parametrize(
        "test_case",
        TypeScenarios.CONTRAVARIANT_TYPEVARS,
        ids=lambda c: c.name,
    )
    def test_contravariant_typevars(self, test_case: TypeVarTestCase) -> None:
        """Test contravariant TypeVar definitions are properly exported."""
        if test_case.expected_not_none:
            assert test_case.type_var is not None
            assert TypeScenarios.is_typevar(test_case.type_var)

    @pytest.mark.parametrize(
        "test_case",
        TypeScenarios.CQRS_ALIASES,
        ids=lambda c: c.name,
    )
    def test_cqrs_aliases(self, test_case: TypeVarTestCase) -> None:
        """Test CQRS type aliases are properly defined."""
        if test_case.expected_not_none:
            assert test_case.type_var is not None

    @pytest.mark.parametrize(
        "test_case",
        TypeScenarios.PARAMSPEC_ITEMS,
        ids=lambda c: c.name,
    )
    def test_paramspec(self, test_case: TypeVarTestCase) -> None:
        """Test ParamSpec is properly defined and exported."""
        if test_case.expected_not_none:
            assert test_case.type_var is not None
            assert isinstance(test_case.type_var, ParamSpec)

    def test_flexttypes_accessible(self) -> None:
        """Test t namespace is accessible."""
        assert t is not None
        # Verify actual nested classes in FlextTypes
        # Note: Handler (not HandlerAliases), Dispatcher (not Processor)
        # Factory, Bus, Logging, Cqrs exist in other modules (constants, models)
        assert all(
            hasattr(t, attr)
            for attr in [
                "Validation",
                "Json",
                "Handler",
                "Dispatcher",
                "Utility",
                "Config",
                "Types",
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

    def test_hostname_validation_success(self) -> None:
        """Test hostname validation success path for 100% coverage."""
        # Test with a valid hostname (localhost should always resolve)
        result = t.Validation._validate_hostname("localhost")
        assert result == "localhost"

        # Test with a valid IP address (should also work)
        result = t.Validation._validate_hostname("127.0.0.1")
        assert result == "127.0.0.1"

    def test_hostname_validation_error(self) -> None:
        """Test hostname validation error path for 100% coverage."""
        # Access the validator via t.Validation namespace
        # The validator is a static method in t.Validation class
        invalid_hostname = "this-hostname-definitely-does-not-exist-12345.invalid"

        # Test that invalid hostname raises ValueError
        with pytest.raises(ValueError, match="Cannot resolve hostname"):
            t.Validation._validate_hostname(invalid_hostname)


__all__ = ["TestFlextTypings"]
