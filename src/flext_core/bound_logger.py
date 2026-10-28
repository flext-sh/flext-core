"""Public facade for the stateful bound logger owned by the logging family.

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

from flext_core._utilities import FlextUtilitiesLoggingBoundLogger

FlextBoundLogger = FlextUtilitiesLoggingBoundLogger

__all__: list[str] = ["FlextBoundLogger"]
