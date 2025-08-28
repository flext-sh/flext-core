"""Domain-Driven Design aggregate root - FACADE LAYER to consolidated FlextModels.

Following complete consolidation request:
- All functionality moved to FlextModels.AggregateRoot
- This module now provides compatibility facade only
- All classes are aliases to FlextModels nested classes

Usage:
    # These are now facades to FlextModels
    class UserAggregate(FlextAggregateRoot):
        name: str
        email: str

        def validate_business_rules(self) -> FlextResult[None]:
            return FlextResult[None].ok(None)
"""

from __future__ import annotations

from flext_core.models import FlextModels

# =============================================================================
# AGGREGATE ROOT FACADES - All functionality in FlextModels now
# =============================================================================

# Main aggregate classes - facades to FlextModels
FlextAggregateRoot = FlextModels.AggregateRoot
FlextEntity = FlextModels.Entity  # For backward compatibility

# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "FlextAggregateRoot",
    "FlextEntity",  # Legacy compatibility
]
