"""Exceptions for gate contract validation.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations


class GateContractUsageError(Exception):
    """Invalid CLI usage for gate contract validation."""


class GateContractInfraError(Exception):
    """Infrastructure failure during gate contract validation."""


__all__: list[str] = ["GateContractInfraError", "GateContractUsageError"]
