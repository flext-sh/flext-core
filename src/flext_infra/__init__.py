"""Public API for flext-infra.

Provides access to infrastructure services for workspace management, validation,
dependency handling, and build orchestration in the FLEXT ecosystem.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_infra.__version__ import __version__, __version_info__
from flext_infra.basemk import BaseMkGenerator, TemplateEngine as BaseMkTemplateEngine
from flext_infra.constants import InfraConstants, ic
from flext_infra.discovery import DiscoveryService
from flext_infra.git import GitService
from flext_infra.json_io import JsonService
from flext_infra.models import FlextInfraModels, InfraModels, im, m
from flext_infra.paths import PathResolver
from flext_infra.patterns import InfraPatterns
from flext_infra.protocols import FlextInfraProtocols, p
from flext_infra.release import ReleaseOrchestrator
from flext_infra.reporting import ReportingService
from flext_infra.selection import ProjectSelector
from flext_infra.subprocess import CommandRunner
from flext_infra.templates import TemplateEngine
from flext_infra.toml_io import TomlService
from flext_infra.versioning import VersioningService

__all__ = [
    "BaseMkGenerator",
    "BaseMkTemplateEngine",
    "CommandRunner",
    "DiscoveryService",
    "FlextInfraModels",
    "FlextInfraProtocols",
    "GitService",
    "InfraConstants",
    "InfraModels",
    "InfraPatterns",
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
    "ic",
    "im",
    "m",
    "p",
]
