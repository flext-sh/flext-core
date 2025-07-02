"""Domain events module for enterprise event sourcing.

This module provides compatibility imports for domain events after consolidation
with the enterprise event bus system.
"""

from __future__ import annotations

from typing import Any

# ZERO TOLERANCE CONSOLIDATION: DomainEvent moved to flx_core.events.event_bus
# Import canonical implementation from event_bus
from flx_core.events.event_bus import DomainEvent

# Python 3.13 type aliases
EventData = dict[str, Any]

# Export consolidated interface
__all__ = [
    "DomainEvent",
    "EventData",
]
