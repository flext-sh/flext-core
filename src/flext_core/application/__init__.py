"""Application layer - Use cases and commands.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

Orchestrates domain objects.
"""

from __future__ import annotations

import warnings

# Commands are still in flext_core
from flext_core.application.commands import Command

# Use local imports directly since flext.services.application doesn't exist yet
from flext_core.application.handlers import CommandHandler
from flext_core.application.handlers import EventHandler
from flext_core.application.handlers import QueryHandler
from flext_core.application.handlers import SimpleQueryHandler
from flext_core.application.handlers import VoidCommandHandler
from flext_core.application.pipeline import CreatePipelineCommand
from flext_core.application.pipeline import ExecutePipelineCommand
from flext_core.application.pipeline import GetPipelineQuery
from flext_core.application.pipeline import PipelineService

# Note: Deprecation warnings are issued at the module level in
# handlers.py and pipeline.py

__all__ = [
    # Base commands
    "Command",
    # Base handlers
    "CommandHandler",
    # Pipeline commands
    "CreatePipelineCommand",
    "EventHandler",
    "ExecutePipelineCommand",
    "GetPipelineQuery",
    "PipelineService",
    "QueryHandler",
    "SimpleQueryHandler",
    "VoidCommandHandler",
]
