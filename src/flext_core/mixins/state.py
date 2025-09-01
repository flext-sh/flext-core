"""FLEXT State - Object lifecycle and state management functionality.

Provides comprehensive state management capabilities through hierarchical organization
of state utilities and mixin classes. Built for object lifecycle tracking, state
transitions, and history management with enterprise-grade patterns.

Module Role in Architecture:
    FlextState serves as the state management foundation providing lifecycle
    tracking patterns for object-oriented applications. Integrates with
    FlextResult for type-safe state transitions and FlextLogging for state change auditing.
"""

from __future__ import annotations

from flext_core.protocols import FlextProtocols
from flext_core.result import FlextResult

# =============================================================================
# TIER 1 MODULE PATTERN - SINGLE MAIN EXPORT
# =============================================================================


class FlextState:
    """Unified state management system implementing single class pattern.

    This class serves as the single main export consolidating ALL state
    functionality with enterprise-grade patterns. Provides comprehensive
    object lifecycle and state management capabilities while maintaining clean API.

    Tier 1 Module Pattern: state.py -> FlextState
    All state functionality is accessible through this single interface.
    """

    # =============================================================================
    # CORE STATE OPERATIONS
    # =============================================================================

    @staticmethod
    def initialize_state(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
        initial_state: str = "created",
    ) -> None:
        """Initialize object state.

        Args:
            obj: Object to initialize state on
            initial_state: Initial state name

        """
        obj._state = initial_state
        obj._state_history = [initial_state]
        obj._state_initialized = True

    @staticmethod
    def get_state(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
    ) -> str:
        """Get current state.

        Args:
            obj: Object to get state from

        Returns:
            Current state name

        """
        if not hasattr(obj, "_state_initialized"):
            FlextState.initialize_state(obj)
        return getattr(obj, "_state", "created")

    @staticmethod
    def set_state(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
        new_state: str,
    ) -> FlextResult[None] | None:
        """Set new state with validation.

        Args:
            obj: Object to set state on
            new_state: New state name

        Returns:
            FlextResult with error if validation fails, None if successful

        """
        from flext_core.mixins.logging import FlextLogging
        from flext_core.result import FlextResult

        if not new_state or len(new_state.strip()) == 0:
            return FlextResult[None].fail(
                f"Invalid state: {new_state}", error_code="INVALID_STATE"
            )

        if not hasattr(obj, "_state_initialized"):
            FlextState.initialize_state(obj)

        old_state = FlextState.get_state(obj)
        obj._state = new_state.strip()

        # Update state history
        history = getattr(obj, "_state_history", [])
        history.append(new_state.strip())
        obj._state_history = history

        FlextLogging.log_operation(
            obj, "state_change", old_state=old_state, new_state=new_state
        )
        return None

    @staticmethod
    def get_state_history(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
    ) -> list[str]:
        """Get state change history.

        Args:
            obj: Object to get state history from

        Returns:
            List of state changes in chronological order

        """
        if not hasattr(obj, "_state_initialized"):
            FlextState.initialize_state(obj)
        return list(getattr(obj, "_state_history", ["created"]))

    # =============================================================================
    # MIXIN CLASS
    # =============================================================================

    class Stateful:
        """Mixin class providing state management functionality.

        This mixin adds state lifecycle management to any class, including
        state tracking and history.
        """

        def __init__(self, *args: object, **kwargs: object) -> None:
            """Initialize state management."""
            super().__init__(*args, **kwargs)
            FlextState.initialize_state(self)

        @property
        def state(self) -> str:
            """Get current state."""
            return FlextState.get_state(self)

        @state.setter
        def state(self, new_state: str) -> None:
            """Set new state with validation."""
            result = FlextState.set_state(self, new_state)
            if (
                result is not None
                and hasattr(result, "is_failure")
                and result.is_failure
            ):
                from flext_core.exceptions import FlextExceptions

                raise FlextExceptions.ValidationError(result.error or "Invalid state")

        @property
        def state_history(self) -> list[str]:
            """Get state change history."""
            return FlextState.get_state_history(self)


__all__ = [
    "FlextState",
]
