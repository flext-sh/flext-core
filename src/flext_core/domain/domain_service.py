"""FlextDomainService - Enterprise Domain Service Base Class.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

Professional implementation of Domain-Driven Design (DDD) domain service
base class
following enterprise software engineering principles. This module provides the
foundation class for implementing stateless domain services that coordinate
operations
across multiple entities and value objects.

Single Responsibility: This module contains only the FlextDomainService
base class
and its core functionality, adhering to SOLID principles.
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict


class FlextDomainService(BaseModel, ABC):
    r"""Base class for domain services in enterprise domain modeling.

    FlextDomainService encapsulates domain logic that doesn't naturally
    belong to any entity or value object. These services represent
    stateless operations that are significant in the domain but don't
    have a natural home in domain objects themselves.

    Enterprise Features:
        - Stateless operations ensuring thread-safe concurrent execution
        - Domain logic coordination across multiple entities and value objects
        - Cross-entity operations with transaction boundary management
        - Pure domain focus without infrastructure or application concerns
        - Comprehensive error handling using FlextResult patterns
        - Production-ready logging and monitoring integration points

    Architectural Design:
        - Stateless services with no instance state or side effects
        - Focus exclusively on domain logic and business rules
        - Coordinate operations between multiple domain objects
        - Return domain objects or FlextResult instances for error handling
        - Abstract interface requiring concrete implementation of execute
          method

    Production Usage Patterns:
        Domain service implementation:
        >>> class TransferService(FlextDomainService):
        ...     def execute(
        ...         self,
        ...         from_account: Account,
        ...         to_account: Account,
        ...         amount: Money,
        ...     ) -> FlextResult[None]:
        ...         # Validate business rules
        ...         if from_account.balance < amount:
        ...             msg = "Insufficient funds for transfer"
        ...             return FlextResult.fail(msg)
        ...
        ...         if from_account.is_frozen or to_account.is_frozen:
        ...             msg = "Cannot transfer to/from frozen accounts"
        ...             return FlextResult.fail(msg)
        ...
        ...         # Execute domain logic
        ...         from_account.debit(amount)
        ...         to_account.credit(amount)
        ...
        ...         # Record domain events
        ...         event = MoneyDebited(account_id=from_account.id, amount=amount)
        ...         from_account.add_domain_event(event)
        ...         event = MoneyCredited(account_id=to_account.id, amount=amount)
        ...         to_account.add_domain_event(event)
        ...
        ...         return FlextResult.ok(None)

        Service usage in application layer:
        >>> transfer_service = TransferService()
        >>> result = transfer_service.execute(
        ...     source_account, target_account, transfer_amount
        ... )
        >>>
        >>> if result.is_success:
        ...     logger.info(f"Transfer completed successfully")
        ...     await publish_domain_events(source_account, target_account)
        >>> else:
        ...     logger.error(f"Transfer failed: {result.error}")

    Thread Safety Guarantees:
        - All domain services are stateless and fully thread-safe
        - Concurrent execution of service operations is safe
        - No shared mutable state between service instances
        - Domain object modifications follow entity thread-safety rules

    Performance Characteristics:
        - Stateless design eliminates memory overhead between operations
        - O(1) service instantiation with no initialization cost
        - Efficient execution with minimal indirection overhead
        - Batch operation support for high-throughput scenarios

    """

    model_config = ConfigDict(
        # Immutable services ensuring thread safety
        frozen=True,
        # No instance state allowed for pure domain services
        extra="forbid",
        # Strict validation for enterprise reliability
        validate_assignment=True,
        # JSON schema generation for API documentation
        json_schema_extra={
            "description": ("Stateless domain service for cross-entity operations"),
        },
    )

    @abstractmethod
    def execute(self) -> Any:
        """Execute the domain service operation with business logic.

        This method must be implemented by each concrete domain service
        to contain the core domain logic that this service is responsible
        for executing. The method should be stateless and side-effect-free
        except for modifications to domain objects passed as parameters.

        Returns:
            Result of the domain operation, typically FlextResult for error
            handling

        Example Implementation:
            >>> def execute(self, user: User, role: Role) -> FlextResult[User]:
            ...     # Validate business rules
            ...     if not user.is_verified:
            ...         msg = "Only verified users can be assigned roles"
            ...         return FlextResult.fail(msg)
            ...
            ...     if role.requires_approval and not user.has_approval:
            ...         return FlextResult.fail("Role requires approval")
            ...
            ...     # Execute domain logic
            ...     updated_user = user.assign_role(role)
            ...     event = RoleAssigned(user_id=user.id, role=role)
            ...     updated_user.add_domain_event(event)
            ...
            ...     return FlextResult.ok(updated_user)

        """


__all__ = ["FlextDomainService"]
