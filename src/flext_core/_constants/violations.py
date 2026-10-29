"""Runtime violation exception types of the core public contract.

The static-enforcement machinery that once raised these moved to
flext-infra (legacy rules preserved under flext-infra/legacy/); the
exception contract stays in the core public surface for the future
runtime reactivation of beartype claw.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from warnings import deprecated


class FlextMroViolationError(UserWarning):
    """Raised when facade MRO composition violates the declared layer order.

    ``UserWarning`` base is the ENFORCE-022 runtime-warning contract: the
    enforcement dispatcher captures these categories through
    ``warnings.warn``, which admits ``Warning`` subclasses only.
    """


class FlextSmellViolationError(UserWarning):
    """Raised when a code-smell rule violation blocks composition.

    Shares the ENFORCE-022 ``UserWarning`` base with the MRO violation.
    """


FlextMroViolation = deprecated(
    "compat: FlextMroViolation -> FlextMroViolationError; fix: rewire"
)(FlextMroViolationError)
FlextSmellViolation = deprecated(
    "compat: FlextSmellViolation -> FlextSmellViolationError; fix: rewire"
)(FlextSmellViolationError)

__all__: list[str] = [
    "FlextMroViolation",
    "FlextMroViolationError",
    "FlextSmellViolation",
    "FlextSmellViolationError",
]
