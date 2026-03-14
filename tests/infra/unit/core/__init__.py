# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make codegen
#
"""Core package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import cleanup_submodule_namespace, lazy_getattr

if TYPE_CHECKING:
    from flext_core.typings import FlextTypes
    from tests.infra.unit.core.basemk_validator import (
        TestBaseMkValidatorCore,
        TestBaseMkValidatorEdgeCases,
        TestBaseMkValidatorSha256,
        v,
    )
    from tests.infra.unit.core.init import TestCoreModuleInit
    from tests.infra.unit.core.inventory import (
        TestInventoryServiceCore,
        TestInventoryServiceReports,
        TestInventoryServiceScripts,
    )
    from tests.infra.unit.core.main import (
        TestMainBaseMkValidate,
        TestMainCliRouting,
        TestMainInventory,
        TestMainScan,
    )
    from tests.infra.unit.core.pytest_diag import (
        TestPytestDiagExtractorCore,
        TestPytestDiagLogParsing,
        TestPytestDiagParseXml,
    )
    from tests.infra.unit.core.scanner import (
        TestScannerCore,
        TestScannerHelpers,
        TestScannerMultiFile,
    )
    from tests.infra.unit.core.skill_validator import (
        TestNormalizeStringList,
        TestSafeLoadYaml,
        TestSkillValidatorAstGrepCount,
        TestSkillValidatorCore,
        TestSkillValidatorRenderTemplate,
    )
    from tests.infra.unit.core.stub_chain import (
        TestStubChainAnalyze,
        TestStubChainCore,
        TestStubChainDiscoverProjects,
        TestStubChainIsInternal,
        TestStubChainStubExists,
        TestStubChainValidate,
    )

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "TestBaseMkValidatorCore": (
        "tests.infra.unit.core.basemk_validator",
        "TestBaseMkValidatorCore",
    ),
    "TestBaseMkValidatorEdgeCases": (
        "tests.infra.unit.core.basemk_validator",
        "TestBaseMkValidatorEdgeCases",
    ),
    "TestBaseMkValidatorSha256": (
        "tests.infra.unit.core.basemk_validator",
        "TestBaseMkValidatorSha256",
    ),
    "TestCoreModuleInit": ("tests.infra.unit.core.init", "TestCoreModuleInit"),
    "TestInventoryServiceCore": (
        "tests.infra.unit.core.inventory",
        "TestInventoryServiceCore",
    ),
    "TestInventoryServiceReports": (
        "tests.infra.unit.core.inventory",
        "TestInventoryServiceReports",
    ),
    "TestInventoryServiceScripts": (
        "tests.infra.unit.core.inventory",
        "TestInventoryServiceScripts",
    ),
    "TestMainBaseMkValidate": ("tests.infra.unit.core.main", "TestMainBaseMkValidate"),
    "TestMainCliRouting": ("tests.infra.unit.core.main", "TestMainCliRouting"),
    "TestMainInventory": ("tests.infra.unit.core.main", "TestMainInventory"),
    "TestMainScan": ("tests.infra.unit.core.main", "TestMainScan"),
    "TestNormalizeStringList": (
        "tests.infra.unit.core.skill_validator",
        "TestNormalizeStringList",
    ),
    "TestPytestDiagExtractorCore": (
        "tests.infra.unit.core.pytest_diag",
        "TestPytestDiagExtractorCore",
    ),
    "TestPytestDiagLogParsing": (
        "tests.infra.unit.core.pytest_diag",
        "TestPytestDiagLogParsing",
    ),
    "TestPytestDiagParseXml": (
        "tests.infra.unit.core.pytest_diag",
        "TestPytestDiagParseXml",
    ),
    "TestSafeLoadYaml": ("tests.infra.unit.core.skill_validator", "TestSafeLoadYaml"),
    "TestScannerCore": ("tests.infra.unit.core.scanner", "TestScannerCore"),
    "TestScannerHelpers": ("tests.infra.unit.core.scanner", "TestScannerHelpers"),
    "TestScannerMultiFile": ("tests.infra.unit.core.scanner", "TestScannerMultiFile"),
    "TestSkillValidatorAstGrepCount": (
        "tests.infra.unit.core.skill_validator",
        "TestSkillValidatorAstGrepCount",
    ),
    "TestSkillValidatorCore": (
        "tests.infra.unit.core.skill_validator",
        "TestSkillValidatorCore",
    ),
    "TestSkillValidatorRenderTemplate": (
        "tests.infra.unit.core.skill_validator",
        "TestSkillValidatorRenderTemplate",
    ),
    "TestStubChainAnalyze": (
        "tests.infra.unit.core.stub_chain",
        "TestStubChainAnalyze",
    ),
    "TestStubChainCore": ("tests.infra.unit.core.stub_chain", "TestStubChainCore"),
    "TestStubChainDiscoverProjects": (
        "tests.infra.unit.core.stub_chain",
        "TestStubChainDiscoverProjects",
    ),
    "TestStubChainIsInternal": (
        "tests.infra.unit.core.stub_chain",
        "TestStubChainIsInternal",
    ),
    "TestStubChainStubExists": (
        "tests.infra.unit.core.stub_chain",
        "TestStubChainStubExists",
    ),
    "TestStubChainValidate": (
        "tests.infra.unit.core.stub_chain",
        "TestStubChainValidate",
    ),
    "v": ("tests.infra.unit.core.basemk_validator", "v"),
}

__all__ = [
    "TestBaseMkValidatorCore",
    "TestBaseMkValidatorEdgeCases",
    "TestBaseMkValidatorSha256",
    "TestCoreModuleInit",
    "TestInventoryServiceCore",
    "TestInventoryServiceReports",
    "TestInventoryServiceScripts",
    "TestMainBaseMkValidate",
    "TestMainCliRouting",
    "TestMainInventory",
    "TestMainScan",
    "TestNormalizeStringList",
    "TestPytestDiagExtractorCore",
    "TestPytestDiagLogParsing",
    "TestPytestDiagParseXml",
    "TestSafeLoadYaml",
    "TestScannerCore",
    "TestScannerHelpers",
    "TestScannerMultiFile",
    "TestSkillValidatorAstGrepCount",
    "TestSkillValidatorCore",
    "TestSkillValidatorRenderTemplate",
    "TestStubChainAnalyze",
    "TestStubChainCore",
    "TestStubChainDiscoverProjects",
    "TestStubChainIsInternal",
    "TestStubChainStubExists",
    "TestStubChainValidate",
    "v",
]


def __getattr__(name: str) -> FlextTypes.ModuleExport:
    """Lazy-load module attributes on first access (PEP 562)."""
    return lazy_getattr(name, _LAZY_IMPORTS, globals(), __name__)


def __dir__() -> list[str]:
    """Return list of available attributes for dir() and autocomplete."""
    return sorted(__all__)


cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)
