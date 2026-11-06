# AUTO-GENERATED FILE — Regenerate with: make gen
"""Source-derived structural model contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
tests/_protocols/_mixins/schema_owner
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from flext_tests import FlextTestsProtocols


class TestsFlextProtocolsMixinsSchemaOwner:
    """Own the structural contracts declared by its model source."""

    @runtime_checkable
    class SchemaOwnerText(FlextTestsProtocols.BaseModel, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def kind(self) -> Literal["text"]:
            """Variant tag."""
            ...

        @property
        def text(self) -> str:
            """Text payload."""
            ...

    @runtime_checkable
    class SchemaOwnerCount(FlextTestsProtocols.BaseModel, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def kind(self) -> Literal["count"]:
            """Variant tag."""
            ...

        @property
        def count(self) -> int:
            """Count payload."""
            ...

    @runtime_checkable
    class SchemaOwnerVariantCell(FlextTestsProtocols.BaseModel, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def variant(self) -> TestsFlextProtocolsMixinsSchemaOwner.SchemaOwnerVariant:
            """Variant validated by the model its tag names."""
            ...

    @runtime_checkable
    class SchemaOwnerLeaf(FlextTestsProtocols.BaseModel, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def name(self) -> str:
            """Leaf identity."""
            ...

    @runtime_checkable
    class SchemaOwnerDeclarations(FlextTestsProtocols.BaseModel, Protocol):
        """Structural data contract projected from the owning model."""

        @property
        def leaves(
            self,
        ) -> tuple[TestsFlextProtocolsMixinsSchemaOwner.SchemaOwnerLeaf, ...]:
            """Sibling leaves validated by their real model."""
            ...

    type SchemaOwnerVariant = SchemaOwnerText | SchemaOwnerCount


__all__: list[str] = [
    "TestsFlextProtocolsMixinsSchemaOwner",
]
