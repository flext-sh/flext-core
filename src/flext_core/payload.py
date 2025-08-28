"""Payload system - FACADE LAYER to consolidated FlextModels.

Following complete consolidation request:
- All functionality moved to FlextModels.Payload, FlextModels.Message, FlextModels.Event
- This module now provides compatibility facade only
- All classes are aliases to FlextModels nested classes

Usage:
    # These are now facades to FlextModels
    payload = FlextPayload(data={"key": "value"}, message_type="test", source_service="api")
    message = FlextMessage(data={"content": "hello"}, message_type="greeting", source_service="web")
    event = FlextEvent(
        data={"user_id": "123"},
        message_type="user_created",
        source_service="users",
        aggregate_id="user_123",
        aggregate_type="User"
    )
"""

from __future__ import annotations

from flext_core.models import FlextModels

# =============================================================================
# PAYLOAD FACADES - All functionality in FlextModels now
# =============================================================================

# Main payload classes - facades to FlextModels
FlextPayload = FlextModels.Payload
FlextMessage = FlextModels.Message
FlextEvent = FlextModels.Event

# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "FlextEvent",
    "FlextMessage",
    "FlextPayload",
]
