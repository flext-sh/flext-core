"""Holder whose nested models reference a sibling through a deferred schema owner.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Annotated

from flext_core import m, t
from tests import p


class TestsFlextModelsSchemaOwner:
    """Sibling models declared in one holder (ADR-018 deferred schema owner)."""

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
