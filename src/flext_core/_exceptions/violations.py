"""Runtime violation warning categories of the core public contract.

The static-enforcement machinery that once raised these moved to
flext-infra; the warning contract stays in the core public surface for the
runtime reactivation of beartype claw. They are exception types, so the
exceptions family owns them and ``e`` publishes them through its MRO.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations


class FlextExceptionsViolations:
    """Warning categories raised for violated composition contracts."""

    class MroViolationError(UserWarning):
        """Raised when facade MRO composition violates the declared layer order.

        ``UserWarning`` base is the ENFORCE-022 runtime-warning contract: the
        enforcement dispatcher captures these categories through
        ``warnings.warn``, which admits ``Warning`` subclasses only.
        """

    class SmellViolationError(UserWarning):
        """Raised when a code-smell rule violation blocks composition.

        Shares the ENFORCE-022 ``UserWarning`` base with the MRO violation.
        """


__all__: list[str] = ["FlextExceptionsViolations"]
