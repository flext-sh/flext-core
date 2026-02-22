"""Public API for flext-infra.

Provides access to infrastructure services for workspace management, validation,
dependency handling, and build orchestration in the FLEXT ecosystem.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_infra.__version__ import __version__, __version_info__
from flext_infra.constants import InfraConstants, ic
from flext_infra.models import FlextInfraModels, InfraModels, im, m
from flext_infra.protocols import FlextInfraProtocols, p

__all__ = [
    "InfraConstants",
    "InfraModels",
    "FlextInfraModels",
    "FlextInfraProtocols",
    "__version__",
    "__version_info__",
    "ic",
    "im",
    "m",
    "p",
]
