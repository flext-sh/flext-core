"""Timeout helpers used by ``FlextDispatcher``.

Expose ``TimeoutEnforcer`` to provide deterministic timeout enforcement for
dispatcher-managed handlers. The helper keeps executor configuration isolated
from orchestration code while retaining the same behavior as the consolidated
dispatcher.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._models.dispatcher import FlextModelsDispatcher

TimeoutEnforcer = FlextModelsDispatcher.TimeoutEnforcer


__all__ = ["TimeoutEnforcer"]
