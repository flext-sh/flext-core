"""Behavioral tests for the deferred schema owner of ``t.schema_for_model``.

A nested model can only name a sibling through its holder class, which is not
bound while its body executes. The deferred owner keeps such a field complete at
declaration, so a subclass declared in another module validates, serializes and
rejects invalid input through the real model without binding ``p`` or ``t``.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import pydantic

from tests import m, p, tm, u


class TestsFlextCoreTypingsSchemaOwner:
    """Public contract of the deferred schema owner."""

    @staticmethod
    def test_cross_module_heir_validates_through_real_model() -> None:
        """A heir in another module validates the inherited deferred field."""
        heir = m.Tests.SchemaOwnerHeirDeclarations.model_validate({
            "label": "heir",
            "leaves": ({"name": "first"},),
        })
        tm.that(heir.leaves[0].name, eq="first")
        tm.that(isinstance(heir.leaves[0], p.Tests.SchemaOwnerLeaf), eq=True)

    @staticmethod
    def test_cross_module_heir_rejects_invalid_payload() -> None:
        """The real model's validation still rejects an invalid sibling payload."""
        with u.Tests.raises(pydantic.ValidationError):
            m.Tests.SchemaOwnerHeirDeclarations.model_validate({
                "label": "heir",
                "leaves": ({"nome": "first"},),
            })

    @staticmethod
    def test_union_owner_validates_each_variant_through_its_model() -> None:
        """A model-union owner validates a payload through the model its tag names."""
        cell = m.Tests.SchemaOwnerVariantCell.model_validate({
            "variant": {"kind": "count", "count": 3},
        })
        tm.that(isinstance(cell.variant, p.Tests.SchemaOwnerCount), eq=True)
        tm.that(cell.variant.model_dump(), eq={"kind": "count", "count": 3})

    @staticmethod
    def test_union_owner_rejects_an_unknown_tag() -> None:
        """The discriminated union refuses a tag none of its models declares."""
        with u.Tests.raises(pydantic.ValidationError):
            m.Tests.SchemaOwnerVariantCell.model_validate({
                "variant": {"kind": "other"},
            })

    @staticmethod
    def test_union_owner_json_round_trip_and_schema() -> None:
        """JSON ingress, dump and JSON Schema come from the variant models."""
        cell = m.Tests.SchemaOwnerVariantCell.model_validate_json(
            '{"variant": {"kind": "text", "text": "json"}}'
        )
        tm.that(
            m.Tests.SchemaOwnerVariantCell.model_validate_json(cell.model_dump_json()),
            eq=cell,
        )
        schema = m.Tests.SchemaOwnerVariantCell.model_json_schema()
        tm.that(
            sorted(schema["$defs"]),
            eq=["SchemaOwnerCount", "SchemaOwnerText", "SchemaOwnerVariant"],
        )

    @staticmethod
    def test_json_ingress_and_dump_use_native_model_contract() -> None:
        """JSON ingress, dump and JSON Schema come from the sibling model."""
        heir = m.Tests.SchemaOwnerHeirDeclarations.model_validate_json(
            '{"label": "heir", "leaves": [{"name": "json"}]}'
        )
        tm.that(heir.model_dump(), eq={"leaves": ({"name": "json"},), "label": "heir"})
        schema = m.Tests.SchemaOwnerHeirDeclarations.model_json_schema()
        tm.that(sorted(schema["$defs"]), eq=["SchemaOwnerLeaf"])
