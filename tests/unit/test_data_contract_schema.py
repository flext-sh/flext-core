"""Behavior contract for ``p.DataContract``: model-free typing, model validation.

A projected field stays protocol-typed so consumers never import ``m`` merely to
type, while Pydantic still validates, serializes and rejects the payload through
the owning concrete model resolved lazily from ``__contract_model__``.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import ClassVar, Protocol

import pytest
from pydantic import BaseModel, GetCoreSchemaHandler, ValidationError
from pydantic_core import CoreSchema

from flext_core import p
from tests import tm

__all__ = ["TestsDataContractSchema"]


class OwnedModel(BaseModel):
    """Owning model exercised through the protocol-typed boundary."""

    name: str
    count: int = 0


class OwnedContract(p.DataContract, Protocol):
    """Model-free read contract for :class:`OwnedModel`."""

    __contract_model__: ClassVar[tuple[str, ...]] = (__name__, "OwnedModel")

    name: str

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: object, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        """Delegate validation to the owning model resolved by route.

        Returns:
            The owning model's core schema.

        """
        _ = source
        return p.DataContract.contract_schema(cls.__contract_model__, handler)


class Envelope(BaseModel):
    """Boundary that types with the protocol and validates with the model."""

    owned: OwnedContract


class TestsDataContractSchema:
    """Public behavior of the model-free data contract hook."""

    @staticmethod
    def test_validates_payload_through_the_owning_model() -> None:
        """A protocol-typed field produces the owning model instance."""
        value = Envelope.model_validate({"owned": {"name": "x", "count": 2}})
        tm.that(value.owned, is_=OwnedModel)
        tm.that(value.model_dump(), eq={"owned": {"name": "x", "count": 2}})

    @staticmethod
    def test_round_trips_json_and_model_payloads() -> None:
        """Serialization and an existing model instance both stay valid."""
        value = Envelope.model_validate({"owned": {"name": "x", "count": 2}})
        tm.that(Envelope.model_validate_json(value.model_dump_json()), eq=value)
        tm.that(Envelope.model_validate({"owned": value.owned}), eq=value)

    @staticmethod
    def test_rejects_payload_the_owning_model_rejects() -> None:
        """Missing owned fields raise the owning model's validation error."""
        with pytest.raises(ValidationError) as error:
            Envelope.model_validate({"owned": {}})
        tm.that(
            {item["loc"] for item in error.value.errors()},
            eq={("owned", "name")},
        )
