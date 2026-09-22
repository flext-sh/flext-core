"""Constants mixin for result.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import ClassVar, Literal


class TestsFlextConstantsResult:
    """Flat result and railway constants for flext-core tests."""

    type RailwayOperation = Literal[
        "get_email", "send_email", "get_status", "double", "square", "negate"
    ]
    RAILWAY_OPERATION_GET_EMAIL: ClassVar[RailwayOperation] = "get_email"
    RAILWAY_OPERATION_SEND_EMAIL: ClassVar[RailwayOperation] = "send_email"
    RAILWAY_OPERATION_GET_STATUS: ClassVar[RailwayOperation] = "get_status"
    RAILWAY_OPERATION_DOUBLE: ClassVar[RailwayOperation] = "double"
    RAILWAY_OPERATION_SQUARE: ClassVar[RailwayOperation] = "square"
    RAILWAY_OPERATION_NEGATE: ClassVar[RailwayOperation] = "negate"
    RAILWAY_OPERATIONS: ClassVar[frozenset[RailwayOperation]] = frozenset({
        RAILWAY_OPERATION_GET_EMAIL,
        RAILWAY_OPERATION_SEND_EMAIL,
        RAILWAY_OPERATION_GET_STATUS,
        RAILWAY_OPERATION_DOUBLE,
        RAILWAY_OPERATION_SQUARE,
        RAILWAY_OPERATION_NEGATE,
    })


__all__: list[str] = ["TestsFlextConstantsResult"]
