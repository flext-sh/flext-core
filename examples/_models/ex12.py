from __future__ import annotations

from flext_core import m


class Ex12CommandA(m.Command):
    value: str


class Ex12CommandB(m.Command):
    amount: int
