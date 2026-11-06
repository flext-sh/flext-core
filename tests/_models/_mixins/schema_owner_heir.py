"""Subclass, in another module, of a model whose field defers its schema owner.

The module binds neither ``p`` nor ``t``: the inherited field is complete at its
declaration, so it is never re-resolved in this namespace.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Annotated

from flext_core import m
from tests._models._mixins import TestsFlextModelsSchemaOwner


class TestsFlextModelsSchemaOwnerHeir(TestsFlextModelsSchemaOwner):
    """Cross-module heir of the deferred-owner declarations."""

    class SchemaOwnerHeirDeclarations(
        TestsFlextModelsSchemaOwner.SchemaOwnerDeclarations
    ):
        """Declarations extended outside the module that declares their fields."""

        label: Annotated[str, m.Field(description="Heir-owned label")]


__all__: list[str] = ["TestsFlextModelsSchemaOwnerHeir"]
