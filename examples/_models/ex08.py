from __future__ import annotations

from flext_core import c, m


class Ex08User(m.Entity):
    name: str
    email: str


class Ex08Order(m.AggregateRoot):
    customer_id: str
    status: c.Domain.Status = c.Domain.Status.ACTIVE
