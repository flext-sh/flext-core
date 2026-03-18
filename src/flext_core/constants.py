"""FLEXT Core Constants - Thin MRO Facade.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Final

from flext_core._constants import (
    FlextConstantsBase,
    FlextConstantsCqrs,
    FlextConstantsDomain,
    FlextConstantsInfrastructure,
    FlextConstantsMixins,
    FlextConstantsPlatform,
    FlextConstantsSettings,
    FlextConstantsValidation,
)

PROJECT_KIND_LIBRARY: Final[str] = "library"
PROJECT_KIND_APPLICATION: Final[str] = "application"
PROJECT_KIND_SERVICE: Final[str] = "service"


class FlextConstants(
    FlextConstantsBase,
    FlextConstantsValidation,
    FlextConstantsSettings,
    FlextConstantsPlatform,
    FlextConstantsDomain,
    FlextConstantsCqrs,
    FlextConstantsInfrastructure,
    FlextConstantsMixins,
):
    """Centralized constants for the FLEXT ecosystem (Layer 0)."""

    TIMEOUT: Final[int] = FlextConstantsBase.Network.DEFAULT_TIMEOUT
    VALIDATION_ERROR: Final[str] = FlextConstantsValidation.Errors.VALIDATION_ERROR
    NOT_FOUND: Final[str] = FlextConstantsValidation.Errors.NOT_FOUND
    ENCODING: Final[str] = FlextConstantsSettings.Utilities.DEFAULT_ENCODING
    PAGE_SIZE: Final[int] = FlextConstantsInfrastructure.Pagination.DEFAULT_PAGE_SIZE
    MAX_RETRIES: Final[int] = FlextConstantsPlatform.Reliability.MAX_RETRY_ATTEMPTS
    PROJECT_KIND_LIBRARY: Final[str] = PROJECT_KIND_LIBRARY
    PROJECT_KIND_APPLICATION: Final[str] = PROJECT_KIND_APPLICATION
    PROJECT_KIND_SERVICE: Final[str] = PROJECT_KIND_SERVICE


c = FlextConstants
__all__ = [
    "PROJECT_KIND_APPLICATION",
    "PROJECT_KIND_LIBRARY",
    "PROJECT_KIND_SERVICE",
    "FlextConstants",
    "c",
]
