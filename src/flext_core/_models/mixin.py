"""Mixin state models for FlextMixins infrastructure.

Provides typed Pydantic v2 models for the internal state tracked by
FlextMixins, replacing ad-hoc ConfigMap usage with validated models.

The runtime triple (config, context, container) is already modeled by
``FlextModels.ServiceRuntime`` — this module adds mixin-specific state
that ServiceRuntime does not cover.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pydantic import Field, computed_field

from flext_core._models.base import FlextModelFoundation


class FlextModelsMixin:
    """Namespace for mixin state models.

    Accessed via ``m.Mixin.*`` through the FlextModels facade.

    Models:
        OperationStats: Accumulated metrics from ``FlextMixins.track()`` calls.

    The runtime triple used by ``FlextMixins._runtime`` is ``m.ServiceRuntime``
    (config, context, container) — not duplicated here.
    """

    class OperationStats(FlextModelFoundation.ArbitraryTypesModel):
        """Tracked operation statistics accumulated by FlextMixins.track().

        Replaces the ad-hoc ``m.ConfigMap`` pattern used in ``track()`` with
        a validated, typed model. Computed ``success_rate`` derives from
        ``operation_count`` and ``error_count``.
        """

        operation_count: int = Field(
            default=0,
            ge=0,
            description="Total operations executed.",
        )
        error_count: int = Field(
            default=0,
            ge=0,
            description="Total operations that raised errors.",
        )
        total_duration_ms: float = Field(
            default=0.0,
            ge=0.0,
            description="Cumulative duration in milliseconds.",
        )
        avg_duration_ms: float = Field(
            default=0.0,
            ge=0.0,
            description="Average duration per operation in milliseconds.",
        )

        @computed_field
        def success_rate(self) -> float:
            """Fraction of successful operations (0.0-1.0)."""
            if self.operation_count == 0:
                return 0.0
            return (self.operation_count - self.error_count) / self.operation_count
