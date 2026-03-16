"""FlextProtocolsMetrics - metrics and context stack protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from flext_core import t
from flext_core._protocols.base import FlextProtocolsBase
from flext_core._protocols.result import FlextProtocolsResult


class FlextProtocolsMetrics:
    """Protocols for metrics tracking and execution context stacks."""

    @runtime_checkable
    class MetricsTracker(FlextProtocolsBase.Base, Protocol):
        """Metrics tracking protocol for handler execution metrics.

        Reflects real implementations like FlextMixins.CQRS.MetricsTracker which
        tracks handler execution metrics (latency, success/failure counts, etc.).
        """

        def get_metrics(self) -> FlextProtocolsResult.Result[t.ConfigMap]:
            """Get current metrics dictionary.

            Returns:
                Result[ConfigMap]: Success result with metrics collection

            """
            ...

        def record_metric(
            self, name: str, value: t.Container
        ) -> FlextProtocolsResult.Result[bool]:
            """Record a metric value.

            Args:
                name: Metric name
                value: Metric value to record

            Returns:
                Result[bool]: Success result

            """
            ...

    @runtime_checkable
    class ContextStack(FlextProtocolsBase.Base, Protocol):
        """Execution context stack protocol for CQRS operations.

        Reflects real implementations like FlextMixins.CQRS.ContextStack which
        manages a stack of execution contexts for nested handler invocations.
        """

        def current_context(self) -> FlextProtocolsBase.Model | None:
            """Get current execution context without popping.

            Returns:
                ExecutionContext | None: Current context or None if stack is empty

            """
            ...

        def pop_context(self) -> FlextProtocolsResult.Result[FlextProtocolsBase.Model]:
            """Pop execution context from the stack.

            Returns:
                Result[Model]: Success result with popped context

            """
            ...

        def push_context(
            self, ctx: FlextProtocolsBase.Model
        ) -> FlextProtocolsResult.Result[bool]:
            """Push execution context onto the stack.

            Args:
                ctx: Execution context to push

            Returns:
                Result[bool]: Success result

            """
            ...


__all__ = ["FlextProtocolsMetrics"]
