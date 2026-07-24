"""Centralized generic Annotated aliases for FLEXT typings.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Annotated

from annotated_types import Ge, Gt, Le, Len

from .pydantic import FlextTypesPydantic


class FlextTypesAnnotateds:
    """Generic Annotated aliases shared through the ``t`` facade."""

    type MessageUnion[CommandT, QueryT, EventT] = Annotated[
        CommandT | QueryT | EventT, FlextTypesPydantic.Discriminator("message_type")
    ]

    type OperationResult[SuccessT, FailureT, PartialT] = Annotated[
        SuccessT | FailureT | PartialT, FlextTypesPydantic.Discriminator("result_type")
    ]

    type ValidationOutcome[ValidT, InvalidT, WarningT] = Annotated[
        ValidT | InvalidT | WarningT, FlextTypesPydantic.Discriminator("outcome_type")
    ]

    # -- string constraints --------------------------------------------------
    type NonEmptyStr = Annotated[str, Len(1)]
    # NOTE (multi-agent, bead mro-wfc8): StrippedStr strips surrounding whitespace and
    # rejects blank/whitespace-only (was Annotated[str, Len(1)] — a copy of NonEmptyStr
    # that neither stripped nor rejected "   "). Canonical reuse point for the workspace
    # declaration-purity campaign (models drop validate_business_rules empty-string checks).
    type StrippedStr = Annotated[
        str, FlextTypesPydantic.StringConstraints(strip_whitespace=True, min_length=1)
    ]
    type BoundedStr = Annotated[str, Len(1, 255)]
    type HostnameStr = Annotated[str, Len(1)]
    type UriString = Annotated[str, Len(1)]
    type TimestampStr = Annotated[str, Len(1)]

    # -- integer constraints (PositiveInt / NonNegativeInt in FlextTypesPydantic) -
    type PortNumber = Annotated[int, Ge(1), Le(65535)]
    type RetryCount = Annotated[int, Ge(0), Le(10)]
    type WorkerCount = Annotated[int, Ge(1), Le(100)]
    type HttpStatusCode = Annotated[int, Ge(100), Le(599)]
    type BatchSize = Annotated[int, Ge(1), Le(10000)]
    type MaxLength = Annotated[int, Ge(1)]

    # -- float constraints (PositiveFloat / NonNegativeFloat in FlextTypesPydantic) -
    type PositiveTimeout = Annotated[float, Gt(0.0), Le(300.0)]
    type BackoffMultiplier = Annotated[float, Ge(1.0)]
    type Percentage = Annotated[float, Ge(0.0), Le(100.0)]
    type DecimalFraction = Annotated[float, Ge(0.0), Le(1.0)]


__all__: list[str] = ["FlextTypesAnnotateds"]
