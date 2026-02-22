"""Core infrastructure services.

Provides foundational services for inventory management, validation rules,
base.mk sync checking, pytest diagnostics, pattern scanning, skill validation,
and stub supply chain management.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_infra.core.__main__ import main
from flext_infra.core.basemk_validator import BaseMkValidator
from flext_infra.core.inventory import InventoryService
from flext_infra.core.pytest_diag import PytestDiagExtractor
from flext_infra.core.scanner import TextPatternScanner
from flext_infra.core.skill_validator import SkillValidator
from flext_infra.core.stub_chain import StubSupplyChain

__all__ = [
    "BaseMkValidator",
    "InventoryService",
    "PytestDiagExtractor",
    "SkillValidator",
    "StubSupplyChain",
    "TextPatternScanner",
    "main",
]
