"""Domain service abstractions supporting the 1.0.0 alignment pillar.

These bases codify the service ergonomics described in ``README.md`` and
``docs/architecture.md``: immutable models, context-aware logging, and
``FlextResult`` contracts that remain stable throughout the 1.x lifecycle.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import cast

from pydantic import BaseModel, ConfigDict

from flext_core.mixins import FlextMixins
from flext_core.models import FlextModels
from flext_core.result import FlextResult
from flext_core.typings import FlextTypes, TDomainResult
from flext_core.utilities import FlextUtilities


class FlextDomainService[TDomainResult](
    FlextModels.TimestampedModel,
    FlextMixins.Serializable,
    FlextMixins.Loggable,
    ABC,
):
    """Optimized domain service base using advanced railway patterns.

    Reduced complexity through monadic composition while maintaining
    identical functionality and API compatibility for FLEXT 1.0.0.
    """

    model_config = ConfigDict(
        frozen=True,
        validate_assignment=True,
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    def to_json(self, indent: int | None = None) -> str:
        """Convert to JSON string while preserving modernization metadata."""
        return FlextMixins.to_json(self, indent)

    def is_valid(self) -> bool:
        """Check service validity using railway pattern composition."""
        return self.validate_business_rules().is_success

    def validate_business_rules(self) -> FlextResult[object | None]:
        """Validate domain service business rules using railway contract."""
        return FlextResult[object | None].ok(None)

    @abstractmethod
    def execute(self) -> FlextResult[TDomainResult]:
        """Execute the main domain service operation with result contract."""
        ...

    def validate_config(self) -> FlextResult[object | None]:
        """Validate service configuration using railway guardrails."""
        return FlextResult[object | None].ok(None)

    def execute_operation(
        self,
        operation_name: str,
        operation: object,
        *args: object,
        **kwargs: object,
    ) -> FlextResult[object]:
        """Execute operation using optimized railway composition."""
        return (
            FlextResult.ok(operation)
            >> (lambda op: self._validate_callable(op, operation_name))
            >> (lambda _: self._safe_execute_operation(operation, *args, **kwargs))
        )

    def _validate_callable(
        self, operation: object, operation_name: str
    ) -> FlextResult[object]:
        """Validate callable using filter pattern."""
        return FlextResult.ok(operation).filter(
            callable, f"Operation {operation_name} is not callable"
        )

    def _safe_execute_operation(
        self, operation: object, *args: object, **kwargs: object
    ) -> FlextResult[object]:
        """Execute operation using FlextResult.from_exception pattern."""

        def execute() -> object:
            if not callable(operation):
                msg = "Operation is not callable"
                raise TypeError(msg)
            return operation(*args, **kwargs)

        return FlextResult.from_exception(execute)

    def get_service_info(self) -> FlextTypes.Core.Dict:
        """Return service metadata using railway validation composition."""
        validation_results = FlextResult.chain_validations(
            self.validate_config, self.validate_business_rules
        )

        return {
            "service_type": self.__class__.__name__,
            "service_id": f"service_{self.__class__.__name__.lower()}_{FlextUtilities.Generators.generate_id()}",
            "config_valid": self.validate_config().is_success,
            "business_rules_valid": self.validate_business_rules().is_success,
            "configuration": cast("BaseModel", self).model_dump(),
            "is_valid": validation_results.is_success,
            "timestamp": FlextUtilities.Generators.generate_iso_timestamp(),
        }

    def execute_with_full_validation(self) -> FlextResult[TDomainResult]:
        """Execute with complete validation pipeline using railway composition."""
        return FlextResult.chain_validations(
            self.validate_config, self.validate_business_rules
        ) >> (lambda _: self.execute())

    def execute_batch_operations(
        self, operations: list[Callable[[], FlextResult[object]]]
    ) -> FlextResult[list[object]]:
        """Execute multiple operations using traverse pattern."""
        return FlextResult.traverse(operations, lambda op: op())

    def retry_execute(self, max_attempts: int = 3) -> FlextResult[TDomainResult]:
        """Retry execution with railway pattern backoff."""
        return FlextUtilities.Reliability.retry_with_backoff(
            self.execute, max_retries=max_attempts
        )


__all__: FlextTypes.Core.StringList = [
    "FlextDomainService",
]
