from __future__ import annotations

from examples import c
from flext_core import m


class Ex08User(m.Entity):
    name: str
    email: str


class Ex08Order(m.AggregateRoot):
    customer_id: str
    status: c.Status = c.Status.ACTIVE
