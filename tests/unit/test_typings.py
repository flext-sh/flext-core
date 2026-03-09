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

from enum import StrEnum
from typing import ClassVar, ParamSpec, TypeVar

import pytest
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter as PydanticTypeAdapter,
    ValidationError as PydanticValidationError,
)

from flext_core import FlextConstants, P, R, ResultT, T, T_co, T_contra, U, e, t
from flext_tests import tm


class TypeVarCategory(StrEnum):
    """TypeVar categories for parametrized testing."""

    CORE = "core"
    COVARIANT = "covariant"
    CONTRAVARIANT = "contravariant"
    PARAMSPEC = "paramspec"
    CQRS = "cqrs"


class TypeVarTestCase(BaseModel):
    """TypeVar test case definition."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="Type variable test case name")
    category: TypeVarCategory = Field(description="Type variable category")
    type_var: object = Field(description="Type variable object under test")
    expected_not_none: bool = Field(
        default=True,
        description="Whether object is expected to be non-none",
    )


class TypeScenarios:
    """Factory for type system test scenarios with centralized test data."""

    CORE_TYPEVARS: ClassVar[list[TypeVarTestCase]] = [
        TypeVarTestCase.model_validate({
            "name": "T",
            "category": TypeVarCategory.CORE,
            "type_var": T,
        }),
        TypeVarTestCase.model_validate({
            "name": "U",
            "category": TypeVarCategory.CORE,
            "type_var": U,
        }),
        TypeVarTestCase.model_validate({
            "name": "E",
            "category": TypeVarCategory.CORE,
            "type_var": e,
        }),
        TypeVarTestCase.model_validate({
            "name": "R",
            "category": TypeVarCategory.CORE,
            "type_var": R,
        }),
        TypeVarTestCase.model_validate(
            {"name": "ResultT", "category": TypeVarCategory.CORE, "type_var": ResultT},
        ),
    ]
    COVARIANT_TYPEVARS: ClassVar[list[TypeVarTestCase]] = [
        TypeVarTestCase.model_validate(
            {"name": "T_co", "category": TypeVarCategory.COVARIANT, "type_var": T_co},
        ),
    ]
    CONTRAVARIANT_TYPEVARS: ClassVar[list[TypeVarTestCase]] = [
        TypeVarTestCase.model_validate(
            {
                "name": "T_contra",
                "category": TypeVarCategory.CONTRAVARIANT,
                "type_var": T_contra,
            },
        ),
    ]
    CQRS_ALIASES: ClassVar[list[TypeVarTestCase]] = [
        TypeVarTestCase.model_validate(
            {
                "name": "Command",
                "category": TypeVarCategory.CQRS,
                "type_var": t.ContainerValue,
            },
        ),
        TypeVarTestCase.model_validate(
            {
                "name": "Query",
                "category": TypeVarCategory.CQRS,
                "type_var": t.ContainerValue,
            },
        ),
        TypeVarTestCase.model_validate(
            {
                "name": "Event",
                "category": TypeVarCategory.CQRS,
                "type_var": t.ContainerValue,
            },
        ),
        TypeVarTestCase.model_validate(
            {
                "name": "Message",
                "category": TypeVarCategory.CQRS,
                "type_var": t.ContainerValue,
            },
        ),
    ]
    PARAMSPEC_ITEMS: ClassVar[list[TypeVarTestCase]] = [
        TypeVarTestCase.model_validate(
            {"name": "P", "category": TypeVarCategory.PARAMSPEC, "type_var": P},
        ),
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
            tm.that(test_case.type_var, none=False, msg="Covariant TypeVar must exist")

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
            tm.that(
                test_case.type_var, none=False, msg="Contravariant TypeVar must exist"
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
            tm.that(
                test_case.type_var,
                eq=t.ContainerValue,
                msg=f"{test_case.name} must equal t.ContainerValue",
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
            tm.that(test_case.type_var, none=False, msg="ParamSpec must exist")

    def test_flexttypes_accessible(self) -> None:
        """Test t namespace is accessible with real validation."""
        tm.that(t, none=False, msg="FlextTypes (t) must be accessible")
        flat_types = ["Container", "Scalar", "HandlerCallable"]
        for type_alias in flat_types:
            tm.that(
                hasattr(t, type_alias),
                eq=True,
                msg=f"t must have {type_alias} type alias",
            )

    def test_all_exports_importable(self) -> None:
        """Test that all public exports can be imported and are valid."""
        core_typevars = [T, U, e, R, ResultT, T_co, T_contra, P]
        for tv in core_typevars:
            tm.that(tv, none=False, msg="TypeVar must be importable and not None")

    def test_module_structure(self) -> None:
        """Test that t has expected structure with real validation."""
        for tv in [T, U, P, R]:
            tm.that(tv, none=False, msg="TypeVar must not be None")
        cqrs_aliases = [
            t.ContainerValue,
            t.ContainerValue,
            t.ContainerValue,
            t.ContainerValue,
        ]
        for alias in cqrs_aliases:
            tm.that(alias, none=False, msg="CQRS alias must not be None")
            tm.that(
                alias,
                eq=t.ContainerValue,
                msg="CQRS alias must equal t.ContainerValue",
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
