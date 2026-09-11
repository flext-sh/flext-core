"""Validate-command CLI route ownership."""

from __future__ import annotations

from typing import ClassVar

from flext_infra import m
from flext_infra.services.cli_route_base import CliRouteBase
from flext_infra.validate.cprofile_report import FlextInfraCProfileReport
from flext_infra.validate.fresh_import import FlextInfraValidateFreshImport
from flext_infra.validate.import_cycles import FlextInfraValidateImportCycles
from flext_infra.validate.inventory import FlextInfraInventoryService
from flext_infra.validate.lazy_map_freshness import FlextInfraValidateLazyMapFreshness
from flext_infra.validate.manual_command import FlextInfraManualCommandValidator
from flext_infra.validate.metadata_discipline import (
    FlextInfraValidateMetadataDiscipline,
)
from flext_infra.validate.namespace_validator import FlextInfraNamespaceValidator
from flext_infra.validate.pytest_diag import FlextInfraPytestDiagExtractor
from flext_infra.validate.runtime_census import FlextInfraRuntimeCensusValidator
from flext_infra.validate.scanner import FlextInfraTextPatternScanner
from flext_infra.validate.silent_failure import FlextInfraSilentFailureValidator
from flext_infra.validate.skill_validator import FlextInfraSkillValidator
from flext_infra.validate.stub_chain import FlextInfraStubSupplyChain
from flext_infra.validate.tier_whitelist import FlextInfraValidateTierWhitelist


class ValidationCommandRoutes(CliRouteBase):
    """Own the complete validate command tuple."""

    validate_command_routes: ClassVar[tuple[m.Cli.ResultCommandRoute, ...]] = tuple(
        m.Cli.ResultCommandRoute(
            name=route_name,
            help_text=help_text,
            model_cls=model_cls,
            handler=CliRouteBase.result_handler(handler),
        )
        for route_name, help_text, model_cls, handler in (
            (
                "cprofile-report",
                "Render a bounded cProfile report",
                FlextInfraCProfileReport,
                FlextInfraCProfileReport.execute_command,
            ),
            (
                "inventory",
                "Generate scripts inventory",
                FlextInfraInventoryService,
                FlextInfraInventoryService.execute_command,
            ),
            (
                "runtime-census",
                "Post-import Beartype enforcement census for flext_* modules",
                FlextInfraRuntimeCensusValidator,
                FlextInfraRuntimeCensusValidator.execute_command,
            ),
            (
                "pytest-diag",
                "Extract pytest diagnostics",
                FlextInfraPytestDiagExtractor,
                FlextInfraPytestDiagExtractor.execute_command,
            ),
            (
                "scan",
                "Scan text files for patterns",
                FlextInfraTextPatternScanner,
                FlextInfraTextPatternScanner.execute_command,
            ),
            (
                "skill-validate",
                "Validate a skill",
                FlextInfraSkillValidator,
                FlextInfraSkillValidator.execute_command,
            ),
            (
                "silent-failure",
                "Validate silent failure sentinel returns",
                FlextInfraSilentFailureValidator,
                FlextInfraSilentFailureValidator.execute_command,
            ),
            (
                "stub-validate",
                "Validate stub supply chain",
                FlextInfraStubSupplyChain,
                FlextInfraStubSupplyChain.execute_command,
            ),
            (
                "fresh-import",
                "Guard 7: fresh-process import smoke test",
                FlextInfraValidateFreshImport,
                FlextInfraValidateFreshImport.execute_command,
            ),
            (
                "import-cycles",
                "Guard 1: ROPE-backed import cycle detector",
                FlextInfraValidateImportCycles,
                FlextInfraValidateImportCycles.execute_command,
            ),
            (
                "lazy-map-freshness",
                "Guard 2/3: lazy-map freshness validator",
                FlextInfraValidateLazyMapFreshness,
                FlextInfraValidateLazyMapFreshness.execute_command,
            ),
            (
                "namespace",
                "Guard: static namespace rules (NS-000..003) via rope",
                FlextInfraNamespaceValidator,
                FlextInfraNamespaceValidator.execute_command,
            ),
            (
                "tier-whitelist",
                "Guard 5: tier-whitelist/abstraction-boundary enforcer",
                FlextInfraValidateTierWhitelist,
                FlextInfraValidateTierWhitelist.execute_command,
            ),
            (
                "metadata-discipline",
                "Guard 8: centralized metadata parser discipline",
                FlextInfraValidateMetadataDiscipline,
                FlextInfraValidateMetadataDiscipline.execute_command,
            ),
            (
                "manual-cmd",
                "Manual-command blocker (§5): pre-commit config drift gate",
                FlextInfraManualCommandValidator,
                FlextInfraManualCommandValidator.execute_command,
            ),
        )
    )


__all__: list[str] = ["ValidationCommandRoutes"]
