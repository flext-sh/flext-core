"""Workspace Python file iteration helpers.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_infra._iteration_directory import FlextInfraUtilitiesIterationDirectory
from flext_infra._iteration_matching import FlextInfraUtilitiesIterationMatching
from flext_infra._iteration_project import FlextInfraUtilitiesIterationProject
from flext_infra._iteration_workspace import FlextInfraUtilitiesIterationWorkspace


class FlextInfraUtilitiesIteration(
    FlextInfraUtilitiesIterationMatching,
    FlextInfraUtilitiesIterationDirectory,
    FlextInfraUtilitiesIterationWorkspace,
    FlextInfraUtilitiesIterationProject,
):
    """Static helpers for discovering and iterating Python files in workspace."""


__all__: list[str] = ["FlextInfraUtilitiesIteration"]
