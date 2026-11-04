# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
tests/_protocols/_mixins/schema_owner_heir
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from tests._protocols._mixins import TestsFlextProtocolsMixinsSchemaOwner


class TestsFlextProtocolsMixinsSchemaOwnerHeir:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class SchemaOwnerHeirDeclarations(
        TestsFlextProtocolsMixinsSchemaOwner.SchemaOwnerDeclarations, Protocol
    ):
        """Structural data contract projected from the owning model."""

        @property
        def label(self) -> str:
            """Heir-owned label."""
            ...


__all__: list[str] = [
    "TestsFlextProtocolsMixinsSchemaOwnerHeir",
]
