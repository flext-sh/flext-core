"""LEGACY: Value objects - Use FlextValue from models.py instead.

This module provides compatibility for projects that import
FlextValueObject directly. All new development should use FlextValue
from flext_core.models which provides modern Pydantic patterns.
"""

from __future__ import annotations

# All imports at top for E402 compliance
from flext_core.models import FlextFactory, FlextValue

# Legacy compatibility aliases
FlextValueObject = FlextValue
FlextValueObjectFactory = FlextFactory

# Export API for compatibility
__all__: list[str] = ["FlextValueObject", "FlextValueObjectFactory"]
