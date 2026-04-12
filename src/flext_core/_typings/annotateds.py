"""Centralized generic Annotated aliases for FLEXT typings.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Annotated

from annotated_types import Ge, Gt, Le, Len

from flext_core._typings.pydantic import FlextTypesPydantic


class FlextTypesAnnotateds:
    """Generic Annotated aliases shared through the ``t`` facade."""

    type MessageUnion[CommandT, QueryT, EventT] = Annotated[
        CommandT | QueryT | EventT,
        FlextTypesPydantic.Discriminator("message_type"),
    ]

    type OperationResult[SuccessT, FailureT, PartialT] = Annotated[
        SuccessT | FailureT | PartialT,
        FlextTypesPydantic.Discriminator("result_type"),
    ]

    type ValidationOutcome[ValidT, InvalidT, WarningT] = Annotated[
        ValidT | InvalidT | WarningT,
        FlextTypesPydantic.Discriminator("outcome_type"),
    ]

    # -- string constraints --------------------------------------------------
    type NonEmptyStr = Annotated[str, Len(1)]
    type StrippedStr = Annotated[str, Len(1)]
    type BoundedStr = Annotated[str, Len(1, 255)]
    type HostnameStr = Annotated[str, Len(1)]
    type UriString = Annotated[str, Len(1)]
    type TimestampStr = Annotated[str, Len(1)]

    # -- integer constraints --------------------------------------------------
    type PositiveInt = Annotated[int, Gt(0)]
    type NonNegativeInt = Annotated[int, Ge(0)]
    type PortNumber = Annotated[int, Ge(1), Le(65535)]
    type RetryCount = Annotated[int, Ge(0), Le(10)]
    type WorkerCount = Annotated[int, Ge(1), Le(100)]
    type HttpStatusCode = Annotated[int, Ge(100), Le(599)]
    type BatchSize = Annotated[int, Ge(1), Le(10000)]
    type MaxLength = Annotated[int, Ge(1)]

    # -- float constraints ----------------------------------------------------
    type PositiveFloat = Annotated[float, Gt(0.0)]
    type NonNegativeFloat = Annotated[float, Ge(0.0)]
    type PositiveTimeout = Annotated[float, Gt(0.0), Le(300.0)]
    type BackoffMultiplier = Annotated[float, Ge(1.0)]
    type Percentage = Annotated[float, Ge(0.0), Le(100.0)]
    type DecimalFraction = Annotated[float, Ge(0.0), Le(1.0)]


__all__: list[str] = ["FlextTypesAnnotateds"]
