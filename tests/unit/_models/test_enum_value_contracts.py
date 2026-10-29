"""Public model behavior for the governed enum-value contract.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import pytest

from tests import c, m, tm


class TestsFlextCoreEnumValueContracts:
    """Enum inputs retain strict validation and produce primitive model values."""

    class Managed(m.ManagedModel):
        """Mutable state inherits enum extraction and default validation."""

        kind: c.HandlerType = c.DEFAULT_HANDLER_TYPE

    class Strict(m.StrictManagedModel):
        """Strict mutable state validates every assigned enum member."""

        kind: c.HandlerType

    class Factory(m.ManagedModel):
        """An enum factory is validated through the same preset as a default."""

        kind: c.HandlerType = m.Field(default_factory=lambda: c.DEFAULT_HANDLER_TYPE)

    class Frozen(m.FrozenModel):
        """Frozen state exposes the validated primitive without a writer."""

        kind: c.HandlerType

    class Contract(m.ContractModel):
        """Public immutable contracts share the same enum extraction owner."""

        kind: c.HandlerType

    @staticmethod
    @pytest.mark.parametrize("member", c.HandlerType)
    @pytest.mark.parametrize("model", [Managed, Strict, Factory, Frozen, Contract])
    def test_enum_python_input_and_json_round_trip(
        model: type[m.BaseModel], member: c.HandlerType
    ) -> None:
        """All supplied members validate to primitive values through public APIs."""
        payload = model.model_validate({"kind": member})
        expected = {"kind": member.value}

        tm.that(payload.model_dump(), eq=expected)
        tm.that(type(payload.model_dump()["kind"]), eq=str)
        tm.that(
            model.model_validate_json(payload.model_dump_json()).model_dump(),
            eq=expected,
        )
        schema = model.model_json_schema()
        tm.that(
            schema["$defs"]["HandlerType"]["enum"],
            eq=[member.value for member in c.HandlerType],
        )
        with pytest.raises(m.ValidationError):
            model.model_validate_json('{"kind":null}')

    @staticmethod
    @pytest.mark.parametrize("model", [Managed, Factory])
    def test_default_is_validated_by_the_shared_preset(
        model: type[m.BaseModel],
    ) -> None:
        """Defaults are extracted without a local field-level validation policy."""
        payload = model.model_validate({})

        tm.that(payload.model_dump(), eq={"kind": c.DEFAULT_HANDLER_TYPE.value})
        tm.that(type(payload.model_dump()["kind"]), eq=str)

    @staticmethod
    def test_strict_assignment_preserves_input_validation() -> None:
        """A mutable strict model extracts members and rejects primitive writes."""
        members = iter(c.HandlerType)
        first, second = next(members), next(members)
        payload = TestsFlextCoreEnumValueContracts.Strict(kind=first)
        payload.kind = second

        tm.that(type(payload.kind), eq=str)
        tm.that(payload.model_dump(), eq={"kind": second.value})
        tm.rejects_assignment(payload, "kind", first.value, expected=m.ValidationError)
        tm.that(payload.model_dump(), eq={"kind": second.value})

    @staticmethod
    @pytest.mark.parametrize("model", [Strict, Frozen, Contract])
    def test_strict_python_input_rejects_a_primitive(
        model: type[m.BaseModel],
    ) -> None:
        """JSON enum values do not authorize coercion of a strict Python input."""
        with pytest.raises(m.ValidationError):
            model.model_validate({"kind": c.DEFAULT_HANDLER_TYPE.value})

    @staticmethod
    @pytest.mark.parametrize("model", [Frozen, Contract])
    def test_frozen_models_reject_enum_assignment(model: type[m.BaseModel]) -> None:
        """Frozen contracts expose no mutable enum capability."""
        payload = model.model_validate({"kind": c.DEFAULT_HANDLER_TYPE})

        tm.rejects_assignment(
            payload, "kind", next(iter(c.HandlerType)), expected=m.ValidationError
        )
