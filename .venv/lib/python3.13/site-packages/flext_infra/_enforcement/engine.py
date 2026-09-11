"""Shared catalog-backed enforcement rule engine."""

from __future__ import annotations

from flext_infra._enforcement.collection_sources import (
    FlextInfraEnforcementSourceCollectors,
)


class FlextInfraEnforcementEngine(FlextInfraEnforcementSourceCollectors):
    """Single SSOT-backed collector for validation, census, and fix flows."""


__all__: list[str] = ["FlextInfraEnforcementEngine"]
