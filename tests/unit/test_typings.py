from __future__ import annotations

from enum import StrEnum, unique
from typing import ClassVar

import pytest
from flext_tests import tm
from pydantic import (
    TypeAdapter as PydanticTypeAdapter,
    ValidationError as PydanticValidationError,
)

from flext_core import FlextConstants, P, R, ResultT, T, T_co, T_contra, U, e, t


class TestTypings:
    @unique
    class TypeVarCategory(StrEnum):
        """TypeVar category enumeration."""

        CORE = "core"
        COVARIANT = "covariant"
        CONTRAVARIANT = "contravariant"
        PARAMSPEC = "paramspec"

    TYPEVAR_CASES: ClassVar[tuple[tuple[str, TypeVarCategory, str], ...]] = (
        ("T", TypeVarCategory.CORE, "T"),
        ("U", TypeVarCategory.CORE, "U"),
        ("E", TypeVarCategory.CORE, "E"),
        ("R", TypeVarCategory.CORE, "R"),
        ("ResultT", TypeVarCategory.CORE, "ResultT"),
        ("T_co", TypeVarCategory.COVARIANT, "T_co"),
        ("T_contra", TypeVarCategory.CONTRAVARIANT, "T_contra"),
        ("P", TypeVarCategory.PARAMSPEC, "P"),
    )

    CORE_ALIAS_NAMES: ClassVar[tuple[str, ...]] = (
        "Container",
        "Scalar",
        "HandlerCallable",
    )

    @staticmethod
    def _symbol_repr(name: str) -> str:
        symbol_map = {
            "T": T,
            "U": U,
            "E": e,
            "R": R,
            "ResultT": ResultT,
            "T_co": T_co,
            "T_contra": T_contra,
            "P": P,
        }
        symbol = symbol_map[name]
        return repr(symbol)

    @pytest.mark.parametrize(("name", "category", "token"), TYPEVAR_CASES)
    def test_typevars_exported(
        self, name: str, category: TypeVarCategory, token: str
    ) -> None:
        tm.that(category, none=False)
        tm.that(name, none=False, empty=False)
        tm.that(self._symbol_repr(name), has=token)

    def test_flexttypes_namespace_accessible(self) -> None:
        tm.that(t, none=False)
        for alias_name in self.CORE_ALIAS_NAMES:
            tm.that(hasattr(t, alias_name), eq=True)

    def test_hostname_validation_success(self) -> None:
        hostname_adapter: PydanticTypeAdapter[str] = PydanticTypeAdapter(
            t.Validation.HostnameStr,
        )
        localhost = hostname_adapter.validate_python(FlextConstants.Network.LOCALHOST)
        tm.that(localhost, eq=FlextConstants.Network.LOCALHOST)
        loopback = hostname_adapter.validate_python(FlextConstants.Network.LOOPBACK_IP)
        tm.that(loopback, eq=FlextConstants.Network.LOOPBACK_IP)

    def test_hostname_validation_error(self) -> None:
        hostname_adapter: PydanticTypeAdapter[str] = PydanticTypeAdapter(
            t.Validation.HostnameStr,
        )
        with pytest.raises(PydanticValidationError):
            hostname_adapter.validate_python("")


__all__ = ["TestTypings"]
