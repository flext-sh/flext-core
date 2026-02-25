"""Public API for flext-infra.

Provides access to infrastructure services for workspace management, validation,
dependency handling, and build orchestration in the FLEXT ecosystem.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""
# Re-exports and aliases for infra API

from __future__ import annotations

from flext_infra.__version__ import __version__, __version_info__
from flext_infra.basemk import BaseMkGenerator
from flext_infra.basemk import TemplateEngine as BaseMkTemplateEngine
from flext_infra.constants import FlextInfraConstants, c
from flext_infra.discovery import DiscoveryService
from flext_infra.git import GitService
from flext_infra.json_io import JsonService
from flext_infra.models import FlextInfraModels, m
from flext_infra.paths import PathResolver
from flext_infra.patterns import FlextInfraPatterns
from flext_infra.protocols import FlextInfraProtocols, p
from flext_infra.release import ReleaseOrchestrator
from flext_infra.reporting import ReportingService
from flext_infra.selection import ProjectSelector
from flext_infra.subprocess import CommandRunner
from flext_infra.templates import TemplateEngine
from flext_infra.toml_io import TomlService
from flext_infra.typings import FlextInfraTypes, t
from flext_infra.utilities import FlextInfraUtilities, u
from flext_infra.versioning import VersioningService

FlextInfraBaseMkGenerator = BaseMkGenerator
FlextInfraCommandRunner = CommandRunner
FlextInfraDiscoveryService = DiscoveryService
FlextInfraGitService = GitService
FlextInfraJsonService = JsonService
FlextInfraPathResolver = PathResolver
FlextInfraProjectSelector = ProjectSelector
FlextInfraReleaseOrchestrator = ReleaseOrchestrator
FlextInfraReportingService = ReportingService
FlextInfraTemplateEngine = TemplateEngine
FlextInfraTomlService = TomlService
FlextInfraVersioningService = VersioningService

__all__ = [
    "BaseMkGenerator",
    "BaseMkTemplateEngine",
    "CommandRunner",
    "DiscoveryService",
    "FlextInfraBaseMkGenerator",
    "FlextInfraCommandRunner",
    "FlextInfraConstants",
    "FlextInfraDiscoveryService",
    "FlextInfraGitService",
    "FlextInfraJsonService",
    "FlextInfraModels",
    "FlextInfraPathResolver",
    "FlextInfraPatterns",
    "FlextInfraProjectSelector",
    "FlextInfraProtocols",
    "FlextInfraReleaseOrchestrator",
    "FlextInfraReportingService",
    "FlextInfraTemplateEngine",
    "FlextInfraTomlService",
    "FlextInfraTypes",
    "FlextInfraUtilities",
    "FlextInfraVersioningService",
    "GitService",
    "JsonService",
    "PathResolver",
    "ProjectSelector",
    "ReleaseOrchestrator",
    "ReportingService",
    "TemplateEngine",
    "TomlService",
    "VersioningService",
    "__version__",
    "__version_info__",
    "c",
    "m",
    "p",
    "t",
    "u",
]
