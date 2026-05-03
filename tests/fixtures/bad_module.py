"""Bad fixture — each top-level class intentionally violates ONE rule.

Importing this module MUST emit enforcement warnings identifiable by the
violating class name. Used by ``test_enforcement_integration.py`` to
verify every rule actually fires through the real
``__pydantic_init_subclass__`` / ``FlextModelsNamespace.__init_subclass__``
hook chain on real code.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import typing
from typing import Annotated, ClassVar

from flext_core import FlextModels, FlextModelsNamespace
from tests import m, u

# --- Pydantic hook rules ------------------------------------------------


class TestsFlextBadAnyField(FlextModels.ArbitraryTypesModel):
    """Violates ``no_any`` — field annotated as ``typing.Any``."""

    data: Annotated[typing.Any, u.Field(description="Intentionally Any.")] = None




class TestsFlextBadMutableDefault(FlextModels.ArbitraryTypesModel):
    """Violates ``no_mutable_default`` — mutable default instance."""

    tags: Annotated[
        list[str],
        u.Field(description="Non-empty mutable default."),
    ] = ["a"]


class TestsFlextBadMissingDesc(FlextModels.ArbitraryTypesModel):
    """Violates ``missing_description`` — field without ``description=``."""

    undocumented: str = ""


class TestsFlextBadInlineUnion(FlextModels.ArbitraryTypesModel):
    """Violates ``no_inline_union`` — inline union with > max arms."""

    value: Annotated[
        str | int | float | bool | bytes,
        u.Field(description="Five-arm inline union."),
    ] = ""


class TestsFlextBadFrozen(FlextModels.ImmutableValueModel):
    """Violates ``value_not_frozen`` — value-object base with ``frozen=False``."""

    model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=False)

    payload: Annotated[str, u.Field(description="Data payload.")] = ""


# --- Namespace hook rules (inherit FlextModelsNamespace) ----------------


class TestsFlextBadAccessors(FlextModelsNamespace):
    """Violates ``no_accessor_methods`` — public ``get_*``/``set_*``/``is_*``."""

    def get_value(self) -> int:
        return 0

    def set_value(self, value: int) -> None:
        return None

    def is_ready(self) -> bool:
        return True


class TestsFlextBadWorkerSettings(FlextModelsNamespace):
    """Violates ``settings_inheritance`` — Settings name, no FlextSettings base."""


class TestsFlextBadConstants(FlextModelsNamespace):
    """Violates ``const_mutable`` + ``const_lowercase``."""

    items: ClassVar[list[str]] = ["a", "b"]  # mutable + lowercase
