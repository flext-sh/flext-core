"""Public model behavior for the governed enum-preservation contract.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import override

import pytest

from tests import c, m, p, r, s, t, tm


class TestsFlextCoreEnumValueContracts:
    """Enum members retain identity while native JSON serialization emits values."""

    class Managed(m.ManagedModel):
        """Mutable state inherits enum preservation and default validation."""

        kind: c.HandlerType = c.DEFAULT_HANDLER_TYPE

    class Strict(m.StrictManagedModel):
        """Strict mutable state validates every assigned enum member."""

        kind: c.HandlerType

    class Factory(m.ManagedModel):
        """An enum factory is validated through the same preset as a default."""

        kind: c.HandlerType = m.Field(default_factory=lambda: c.DEFAULT_HANDLER_TYPE)

    class Frozen(m.FrozenModel):
        """Frozen state exposes the validated enum member without a writer."""

        kind: c.HandlerType

    class Contract(m.ContractModel):
        """Public immutable contracts share the same enum preservation owner."""

        kind: c.HandlerType

    class SettingsValue(m.SettingsValue):
        """A settings branch retains the enum through its actual preset chain."""

        kind: c.HandlerType

    class Params(m.ParamsModel):
        """Exception parameters inherit the same validated enum contract."""

        kind: c.HandlerType

    class Service(s[c.HandlerType]):
        """A real service retains its enum state through the operational MRO."""

        kind: c.HandlerType

        @override
        def execute(self) -> p.Result[c.HandlerType]:
            """Return the validated member without a transport conversion."""
            return r[c.HandlerType].ok(self.kind)

    class InvalidDefault(m.ManagedModel):
        """A statically typed default must still satisfy its native constraint."""

        value: t.NonEmptyStr = ""

    class InvalidFactory(m.ManagedModel):
        """Factory results cross the same native default-validation boundary."""

        value: t.NonEmptyStr = m.Field(default_factory=str)

    @staticmethod
    @pytest.mark.parametrize("member", c.HandlerType)
    @pytest.mark.parametrize(
        "model",
        [Managed, Strict, Factory, Frozen, Contract, SettingsValue, Params, Service],
    )
    def test_enum_python_input_and_json_round_trip(
        model: type[m.BaseModel], member: c.HandlerType
    ) -> None:
        """Python retains members; the native JSON round trip restores them."""
        payload = model.model_validate({"kind": member})

        tm.ok(payload.model_dump()["kind"] is member)
        tm.ok(model.model_validate(payload) is payload)
        tm.that(
            payload.model_dump(mode="json"),
            eq={"kind": member.value},
        )
        tm.ok(
            model.model_validate_json(payload.model_dump_json()).model_dump()["kind"]
            is member
        )
        schema = model.model_json_schema()
        tm.that(
            set(schema["$defs"]["HandlerType"]["enum"]),
            eq={member.value for member in c.HandlerType},
        )
        with pytest.raises(m.ValidationError):
            model.model_validate_json('{"kind":null}')

    @staticmethod
    @pytest.mark.parametrize("model", [Managed, Factory])
    def test_default_is_validated_by_the_shared_preset(
        model: type[m.BaseModel],
    ) -> None:
        """Defaults retain members without a local field-level validation policy."""
        payload = model.model_validate({})

        tm.ok(payload.model_dump()["kind"] is c.DEFAULT_HANDLER_TYPE)
        tm.that(
            payload.model_dump(mode="json"),
            eq={"kind": c.DEFAULT_HANDLER_TYPE.value},
        )

    @staticmethod
    @pytest.mark.parametrize("model", [InvalidDefault, InvalidFactory])
    def test_defaults_must_satisfy_their_declared_constraints(
        model: type[m.BaseModel],
    ) -> None:
        """Defaults and factory results cannot bypass a constrained data contract."""
        with pytest.raises(m.ValidationError):
            model.model_validate({})

    @staticmethod
    def test_strict_assignment_preserves_input_validation() -> None:
        """A mutable strict model retains members and rejects primitive writes."""
        members = iter(c.HandlerType)
        first, second = next(members), next(members)
        payload = TestsFlextCoreEnumValueContracts.Strict(kind=first)
        payload.kind = second

        tm.ok(payload.kind is second)
        tm.ok(payload.model_dump()["kind"] is second)
        tm.rejects_assignment(payload, "kind", first.value, expected=m.ValidationError)
        tm.ok(payload.kind is second)

    @staticmethod
    def test_service_result_preserves_its_validated_member() -> None:
        """The actual service/result boundary retains the validated domain value."""
        service = TestsFlextCoreEnumValueContracts.Service(kind=c.DEFAULT_HANDLER_TYPE)

        result = service.execute()

        tm.ok(result.success)
        tm.ok(result.value is c.DEFAULT_HANDLER_TYPE)

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
