# AUTO-GENERATED FILE — Regenerate with: make gen
"""Public examples protocols facade for flext-core.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from examples._protocols import (
    ExamplesFlextProtocolsEx00,
    ExamplesFlextProtocolsEx01,
    ExamplesFlextProtocolsEx02,
    ExamplesFlextProtocolsEx03,
    ExamplesFlextProtocolsEx04,
    ExamplesFlextProtocolsEx05,
    ExamplesFlextProtocolsEx07,
    ExamplesFlextProtocolsEx08,
    ExamplesFlextProtocolsEx10,
    ExamplesFlextProtocolsEx11,
    ExamplesFlextProtocolsEx12,
    ExamplesFlextProtocolsEx14,
    ExamplesFlextProtocolsHandle,
    ExamplesFlextProtocolsPerson,
)
from flext_core import FlextProtocols


class ExamplesFlextProtocols(
    FlextProtocols,
    ExamplesFlextProtocolsEx00,
    ExamplesFlextProtocolsEx01,
    ExamplesFlextProtocolsEx02,
    ExamplesFlextProtocolsEx03,
    ExamplesFlextProtocolsEx04,
    ExamplesFlextProtocolsEx05,
    ExamplesFlextProtocolsEx07,
    ExamplesFlextProtocolsEx08,
    ExamplesFlextProtocolsEx10,
    ExamplesFlextProtocolsEx11,
    ExamplesFlextProtocolsEx12,
    ExamplesFlextProtocolsEx14,
    ExamplesFlextProtocolsHandle,
    ExamplesFlextProtocolsPerson,
):
    """Public examples protocol facade — composes all _protocols/* via MRO."""

    class Examples(
        ExamplesFlextProtocolsEx00,
        ExamplesFlextProtocolsEx01,
        ExamplesFlextProtocolsEx02,
        ExamplesFlextProtocolsEx03,
        ExamplesFlextProtocolsEx04,
        ExamplesFlextProtocolsEx05,
        ExamplesFlextProtocolsEx07,
        ExamplesFlextProtocolsEx08,
        ExamplesFlextProtocolsEx10,
        ExamplesFlextProtocolsEx11,
        ExamplesFlextProtocolsEx12,
        ExamplesFlextProtocolsEx14,
        ExamplesFlextProtocolsHandle,
        ExamplesFlextProtocolsPerson,
    ):
        """Canonical namespace for all example structural contracts.

        Access via: from examples import p; p.Examples.<ClassName>
        """


p = ExamplesFlextProtocols
__all__: list[str] = ["ExamplesFlextProtocols", "p"]
