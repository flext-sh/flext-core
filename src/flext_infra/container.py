"""DI Container utilities for flext-infra using flext-core patterns.

This module provides dependency injection container utilities following
flext-core foundation patterns, eliminating code duplication and ensuring
consistent dependency management across the infrastructure implementation.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import FlextContainer, r
from flext_infra import t
from flext_infra.basemk import FlextInfraBaseMkGenerator, FlextInfraBaseMkTemplateEngine
from flext_infra.discovery import FlextInfraDiscoveryService
from flext_infra.git import FlextInfraGitService
from flext_infra.json_io import FlextInfraJsonService
from flext_infra.maintenance import FlextInfraPythonVersionEnforcer
from flext_infra.output import output
from flext_infra.paths import FlextInfraPathResolver
from flext_infra.release import FlextInfraReleaseOrchestrator
from flext_infra.reporting import FlextInfraReportingService
from flext_infra.selection import FlextInfraProjectSelector
from flext_infra.subprocess import FlextInfraCommandRunner
from flext_infra.toml_io import FlextInfraTomlService
from flext_infra.versioning import FlextInfraVersioningService
from flext_infra.workspace import (
    FlextInfraOrchestratorService,
    FlextInfraProjectMigrator,
    FlextInfraSyncService,
    FlextInfraWorkspaceDetector,
)


def get_flext_infra_container() -> FlextContainer:
    """Get the global FLEXT DI container.

    Returns:
        The global FlextContainer singleton instance.

    """
    return FlextContainer.get_global()


def get_flext_infra_service(service_name: str) -> r[t.RegisterableService]:
    """Get service from FLEXT DI container.

    Args:
        service_name: Name of the service to retrieve.

    Returns:
        r containing the service or error.

    """
    container = get_flext_infra_container()
    return container.get(service_name)


def configure_flext_infra_dependencies() -> None:
    """Register all FlextInfra services in the DI container.

    Registers the following services:
    - git_service: FlextInfraGitService
    - json_io: FlextInfraJsonService
    - toml_io: FlextInfraTomlService
    - path_resolver: FlextInfraPathResolver
    - command_runner: FlextInfraCommandRunner
    - discovery: FlextInfraDiscoveryService
    - selection: FlextInfraProjectSelector
    - reporting: FlextInfraReportingService
    - versioning: FlextInfraVersioningService
    - output: FlextInfraOutput singleton instance
    - basemk_engine: FlextInfraBaseMkTemplateEngine
    - basemk_generator: FlextInfraBaseMkGenerator
    - workspace_detector: FlextInfraWorkspaceDetector
    - workspace_migrator: FlextInfraProjectMigrator
    - workspace_orchestrator: FlextInfraOrchestratorService
    - workspace_sync: FlextInfraSyncService
    - release_orchestrator: FlextInfraReleaseOrchestrator
    - python_version_enforcer: FlextInfraPythonVersionEnforcer

    """
    container = get_flext_infra_container()
    _ = container.register(
        "git_service", lambda: FlextInfraGitService(), kind="factory",
    )
    _ = container.register("json_io", lambda: FlextInfraJsonService(), kind="factory")
    _ = container.register("toml_io", lambda: FlextInfraTomlService(), kind="factory")
    _ = container.register(
        "path_resolver", lambda: FlextInfraPathResolver(), kind="factory",
    )
    _ = container.register(
        "command_runner", lambda: FlextInfraCommandRunner(), kind="factory",
    )
    _ = container.register(
        "discovery", lambda: FlextInfraDiscoveryService(), kind="factory",
    )
    _ = container.register(
        "selection", lambda: FlextInfraProjectSelector(), kind="factory",
    )
    _ = container.register(
        "reporting", lambda: FlextInfraReportingService(), kind="factory",
    )
    _ = container.register(
        "versioning", lambda: FlextInfraVersioningService(), kind="factory",
    )
    _ = container.register("output", output)
    _ = container.register(
        "basemk_engine", lambda: FlextInfraBaseMkTemplateEngine(), kind="factory",
    )
    _ = container.register(
        "basemk_generator", lambda: FlextInfraBaseMkGenerator(), kind="factory",
    )
    _ = container.register(
        "workspace_detector", lambda: FlextInfraWorkspaceDetector(), kind="factory",
    )
    _ = container.register(
        "workspace_migrator", lambda: FlextInfraProjectMigrator(), kind="factory",
    )
    _ = container.register(
        "workspace_orchestrator",
        lambda: FlextInfraOrchestratorService(),
        kind="factory",
    )
    _ = container.register(
        "workspace_sync", lambda: FlextInfraSyncService(), kind="factory",
    )
    _ = container.register(
        "release_orchestrator", lambda: FlextInfraReleaseOrchestrator(), kind="factory",
    )
    _ = container.register(
        "python_version_enforcer",
        lambda: FlextInfraPythonVersionEnforcer(),
        kind="factory",
    )


configure_flext_infra_dependencies()
