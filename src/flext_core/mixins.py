"""Backward-compat shim -- canonical location is _utilities/mixins.py."""

from __future__ import annotations

from flext_core._utilities.mixins import FlextMixins

x = FlextMixins
__all__ = ["FlextMixins", "x"]
