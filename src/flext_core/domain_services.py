"""Domain service abstractions supporting the 1.0.0 alignment pillar.

These bases codify the service ergonomics described in ``README.md`` and
``docs/architecture.md``: immutable models, context-aware logging, and
``FlextResult`` contracts that remain stable throughout the 1.x lifecycle.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import cast

from pydantic import BaseModel, ConfigDict

from flext_core.constants import FlextConstants
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
    """Base class for domain services following the modernization guidance.

    Subclasses inherit logging, serialization, and validation utilities so they
    match the domain-service ergonomics promised for the FLEXT 1.0.0 rollout.
    The class keeps contracts strict—``execute`` returns ``FlextResult`` and
    configuration checks are routed through helper validators.
    """

    model_config = ConfigDict(
        frozen=True,
        validate_assignment=True,
        extra="forbid",
        arbitrary_types_allowed=True,  # Allow non-Pydantic types like FlextDbOracleApi
    )

    # Override to_json to resolve inheritance conflict
    def to_json(self, indent: int | None = None) -> str:
        """Convert to JSON string while preserving modernization metadata."""
        return FlextMixins.to_json(self, indent)

    # Mixin functionality is now inherited via FlextMixins.Serializable
    def is_valid(self) -> bool:
        """Check service validity using 1.0.0-aligned validators."""
        try:
            validation_result = self.validate_business_rules()
            return validation_result.is_success
        except Exception:
            # Use FlextUtilities for error logging if needed
            return False

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate domain service business rules per modernization policy."""
        return FlextResult[None].ok(None)

    @abstractmethod
    def execute(self) -> FlextResult[TDomainResult]:
        """Execute the main domain service operation with result contract."""
        ...

    def validate_config(self) -> FlextResult[None]:
        """Validate service configuration using shared configuration guardrails."""
        return FlextResult[None].ok(None)

    def execute_operation(
        self,
        operation_name: str,
        operation: object,
        *args: object,
        **kwargs: object,
    ) -> FlextResult[object]:
        """Execute an operation with modernization-compliant error handling."""
        try:
            # Validate configuration first
            config_result = self.validate_config()
            if config_result.is_failure:
                error_message = (
                    config_result.error
                    or f"{FlextConstants.Messages.VALIDATION_FAILED}: Configuration validation failed"
                )
                return FlextResult[object].fail(
                    error_message,
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )

            # Validate operation is callable and execute
            if not callable(operation):
                return FlextResult[object].fail(
                    f"{FlextConstants.Messages.OPERATION_FAILED}: Operation {operation_name} is not callable",
                    error_code=FlextConstants.Errors.OPERATION_ERROR,
                )

            # Execute the callable operation - MyPy should understand this is reachable
            result = operation(*args, **kwargs)
            return FlextResult[object].ok(result)

        except (RuntimeError, ValueError, TypeError) as e:
            return FlextResult[object].fail(
                f"{FlextConstants.Messages.OPERATION_FAILED}: Operation {operation_name} failed: {e}",
                error_code=FlextConstants.Errors.EXCEPTION_ERROR,
            )
        except Exception as e:
            # Catch any other exceptions using FlextConstants
            return FlextResult[object].fail(
                f"{FlextConstants.Messages.UNKNOWN_ERROR}: Unexpected error in {operation_name}: {e}",
                error_code=FlextConstants.Errors.UNKNOWN_ERROR,
            )

    def get_service_info(self) -> FlextTypes.Core.Dict:
        """Return service metadata for modernization monitoring dashboards."""
        config_result = self.validate_config()
        rules_result = self.validate_business_rules()
        is_valid = config_result.is_success and rules_result.is_success

        return {
            "service_type": self.__class__.__name__,
            "service_id": f"service_{self.__class__.__name__.lower()}_{FlextUtilities.Generators.generate_id()}",
            "config_valid": config_result.is_success,
            "business_rules_valid": rules_result.is_success,
            "configuration": cast(
                "BaseModel", self,
            ).model_dump(),  # Add configuration as expected by tests
            "is_valid": is_valid,  # Add overall validity as expected by tests
            "timestamp": FlextUtilities.Generators.generate_iso_timestamp(),
        }


__all__: FlextTypes.Core.StringList = [
    "FlextDomainService",
]
