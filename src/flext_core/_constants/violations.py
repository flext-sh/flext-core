"""Runtime violation exception types of the core public contract.

The static-enforcement machinery that once raised these moved to
flext-infra (legacy rules preserved under flext-infra/legacy/); the
exception contract stays in the core public surface for the future
runtime reactivation of beartype claw.

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


class FlextMroViolation(Exception):
    """Raised when facade MRO composition violates the declared layer order."""


class FlextSmellViolation(Exception):
    """Raised when a code-smell rule violation blocks composition."""


__all__: list[str] = ["FlextMroViolation", "FlextSmellViolation"]
