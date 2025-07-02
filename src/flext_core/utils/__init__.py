"""Utility modules for FLEXT Core.

This package provides utility functions and patterns used throughout
the FLEXT framework for common operations and integrations.
"""

from .import_fallback_patterns import SQLALCHEMY_DEPENDENCY

__all__ = [
    "SQLALCHEMY_DEPENDENCY"
]
