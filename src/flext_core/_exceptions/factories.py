"""Fail DSL factory facade.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
)

from flext_core._exceptions._factories import FlextExceptionsFactoriesFail


class FlextExceptionsFactories(FlextExceptionsFactoriesFail):
    """Public failure factory namespace."""


__all__: list[str] = ["FlextExceptionsFactories"]
