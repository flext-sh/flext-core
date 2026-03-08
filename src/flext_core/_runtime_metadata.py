"""Metadata model for FlextRuntime — isolated for lazy pydantic loading.

This module is imported lazily by FlextRuntime.__getattr__ to defer
pydantic dependency loading until Metadata is first accessed.

Used by exceptions.py and other low-level modules that cannot import
from _models.base to maintain proper architecture layering.
"""

from __future__ import annotations

from flext_core.models import m

Metadata = m.Metadata
__all__ = ["Metadata"]
