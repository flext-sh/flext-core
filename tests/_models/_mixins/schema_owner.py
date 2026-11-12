"""Holder whose nested models reference a sibling through a simplified model.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Annotated, Literal

from flext_core import m


class TestsFlextModelsSchemaOwner:
    """Sibling models declared in one holder."""

    class SchemaOwnerText(m.FrozenModel):
        """Text variant of a discriminated sibling union."""

        kind: Literal["text"] = m.Field(default="text", description="Variant tag")
        text: Annotated[str, m.Field(description="Text payload")]

    class SchemaOwnerCount(m.FrozenModel):
        """Count variant of a discriminated sibling union."""

        kind: Literal["count"] = m.Field(default="count", description="Variant tag")
        count: Annotated[int, m.Field(description="Count payload")]

    type SchemaOwnerVariant = Annotated[
        SchemaOwnerText | SchemaOwnerCount,
        m.Field(discriminator="kind"),
    ]

    class SchemaOwnerVariantCell(m.FrozenModel):
        """Payload typed by the variant union, validated by its models."""

        variant: TestsFlextModelsSchemaOwner.SchemaOwnerVariant = m.Field(
            description="Variant validated by the model its tag names"
        )

    class SchemaOwnerLeaf(m.FrozenModel):
        """Leaf payload referenced by a sibling of the same holder."""

        name: Annotated[str, m.Field(description="Leaf identity")]

    class SchemaOwnerDeclarations(m.FrozenModel):
        """Payload naming its sibling through the holder."""

        leaves: tuple[TestsFlextModelsSchemaOwner.SchemaOwnerLeaf, ...] = ()


__all__: list[str] = ["TestsFlextModelsSchemaOwner"]
