"""Model coverage smoke tests aligned to current model facade."""

from __future__ import annotations

import math

from tests import m


class TestCoverageModels:
    def test_value_model_instantiation(self) -> None:
        class Money(m.Value):
            amount: float
            currency: str

        value = Money(amount=100.0, currency="USD")
        assert math.isclose(value.amount, 100.0)
        assert value.currency == "USD"

    def test_entity_model_instantiation(self) -> None:
        class User(m.Entity):
            name: str

        user = User(name="alice", domain_events=[])
        assert user.name == "alice"
        assert user.unique_id is not None
