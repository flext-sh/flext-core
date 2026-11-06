"""Holder whose nested models reference a sibling through a deferred schema owner.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Annotated, Literal

from flext_core import m, t
from tests import p


class TestsFlextModelsSchemaOwner:
    """Sibling models declared in one holder (ADR-018 deferred schema owner)."""

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

        variant: Annotated[
            p.Tests.SchemaOwnerVariant,
            t.schema_for_model(lambda: TestsFlextModelsSchemaOwner.SchemaOwnerVariant),
        ] = m.Field(description="Variant validated by the model its tag names")

    class SchemaOwnerLeaf(m.FrozenModel):
        """Leaf payload referenced by a sibling of the same holder."""

        name: Annotated[str, m.Field(description="Leaf identity")]

    class SchemaOwnerDeclarations(m.FrozenModel):
        """Payload naming its sibling through the holder with a deferred owner."""

        leaves: Annotated[
            tuple[
                Annotated[
                    p.Tests.SchemaOwnerLeaf,
                    t.schema_for_model(
                        lambda: TestsFlextModelsSchemaOwner.SchemaOwnerLeaf
                    ),
                ],
                ...,
            ],
            m.Field(description="Sibling leaves validated by their real model"),
        ] = ()


__all__: list[str] = ["TestsFlextModelsSchemaOwner"]
