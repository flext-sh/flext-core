"""Application layer for the FLX platform."""

# Import commands from their canonical location
from flx_core.application.application import (
    FlextEnterpriseApplication as FlxApplication,
)
from flx_core.commands.pipeline import (
    CreatePipelineCommand,
    ExecutePipelineCommand,
    UpdatePipelineCommand,
)

__all__ = [
    "CreatePipelineCommand",
    "ExecutePipelineCommand",
    "FlxApplication",
    "UpdatePipelineCommand",
]
