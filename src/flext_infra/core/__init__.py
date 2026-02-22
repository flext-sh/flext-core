"""Core infrastructure services.

Provides foundational services for inventory management, validation rules,
base.mk sync checking, pytest diagnostics, pattern scanning, skill validation,
and stub supply chain management.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations


def __getattr__(name: str):
    if name == "main":
        from flext_infra.core.__main__ import main

        return main
    if name == "BaseMkValidator":
        from flext_infra.core.basemk_validator import BaseMkValidator

        return BaseMkValidator
    if name == "InventoryService":
        from flext_infra.core.inventory import InventoryService

        return InventoryService
    if name == "PytestDiagExtractor":
        from flext_infra.core.pytest_diag import PytestDiagExtractor

        return PytestDiagExtractor
    if name == "TextPatternScanner":
        from flext_infra.core.scanner import TextPatternScanner

        return TextPatternScanner
    if name == "SkillValidator":
        from flext_infra.core.skill_validator import SkillValidator

        return SkillValidator
    if name == "StubSupplyChain":
        from flext_infra.core.stub_chain import StubSupplyChain

        return StubSupplyChain
    raise AttributeError(name)


__all__ = [
    "BaseMkValidator",
    "InventoryService",
    "PytestDiagExtractor",
    "SkillValidator",
    "StubSupplyChain",
    "TextPatternScanner",
    "main",
]
