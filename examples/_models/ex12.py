from __future__ import annotations

from typing import Annotated

from flext_core import m, u


class Ex12CommandA(m.Command):
    command_type: str = "ex12_command_a"
    value: Annotated[
        str,
        u.Field(description="String payload used by registry example command A."),
    ]


class Ex12CommandB(m.Command):
    command_type: str = "ex12_command_b"
    amount: Annotated[
        int,
        u.Field(description="Numeric payload used by registry example command B."),
    ]
