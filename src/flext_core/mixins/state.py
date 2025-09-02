"""FLEXT State - Object lifecycle and state management functionality.

Provides state management capabilities for object lifecycle tracking, state transitions,
and history management with production-ready patterns.

Usage:
    # Basic state management
    obj = MyObject()
    FlextState.initialize_state(obj)
    state = FlextState.get_state(obj)

    # State transitions with validation
    transition_result = FlextState.transition_state(obj, "active", "inactive")
    if transition_result.success:
        print("State transitioned successfully")

    # State history tracking
    history = FlextState.get_state_history(obj)
"""

from __future__ import annotations

from flext_core.protocols import FlextProtocols
from flext_core.result import FlextResult

# =============================================================================
# TIER 1 MODULE PATTERN - SINGLE MAIN EXPORT
# =============================================================================


class FlextState:
    """Unified state management system implementing single class pattern.

    Single main export consolidating state functionality with production-ready patterns.
    Provides object lifecycle and state management capabilities with clean API.

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

    @staticmethod
    def set_attribute(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
        key: str,
        value: object,
    ) -> None:
        """Set a state attribute with key-value pair.

        Args:
            obj: Object to set state on
            key: Attribute key
            value: Attribute value

        """
        if not hasattr(obj, "_state_attributes"):
            obj._state_attributes = {}

        state_attrs = getattr(obj, "_state_attributes", {})
        state_attrs[key] = value
        obj._state_attributes = state_attrs

    @staticmethod
    def get_attribute(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
        key: str,
    ) -> object:
        """Get a state attribute value.

        Args:
            obj: Object to get state from
            key: Attribute key

        Returns:
            Attribute value or None if not found

        """
        state_attrs = getattr(obj, "_state_attributes", {})
        return state_attrs.get(key)

    @staticmethod
    def has_attribute(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
        key: str,
    ) -> bool:
        """Check if state attribute exists.

        Args:
            obj: Object to check state on
            key: Attribute key

        Returns:
            True if attribute exists

        """
        state_attrs = getattr(obj, "_state_attributes", {})
        return key in state_attrs

    @staticmethod
    def update_state(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
        updates: dict[str, object],
    ) -> None:
        """Update multiple state attributes.

        Args:
            obj: Object to update state on
            updates: Dictionary of key-value pairs to update

        """
        for key, value in updates.items():
            FlextState.set_attribute(obj, key, value)

    @staticmethod
    def validate_state(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
    ) -> bool:
        """Validate current state.

        Args:
            obj: Object to validate state on

        Returns:
            True if state is valid

        """
        # Basic validation - ensure state is initialized
        return hasattr(obj, "_state_initialized") or hasattr(obj, "_state_attributes")

    @staticmethod
    def clear_state(
        obj: FlextProtocols.Foundation.SupportsDynamicAttributes,
    ) -> None:
        """Clear all state attributes.

        Args:
            obj: Object to clear state from

        """
        if hasattr(obj, "_state_attributes"):
            obj._state_attributes = {}

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
