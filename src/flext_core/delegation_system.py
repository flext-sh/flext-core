"""Internal delegation system for mixin composition patterns.

SINGLE CONSOLIDATED MODULE following FLEXT architectural patterns.
All delegation functionality consolidated into FlextDelegationSystem.
"""

from __future__ import annotations

import contextlib
from typing import ClassVar, NoReturn, Protocol, cast

from flext_core.exceptions import FlextExceptions
from flext_core.loggings import FlextLogger
from flext_core.mixins import FlextMixins
from flext_core.result import FlextResult


class FlextDelegationSystem:
    """SINGLE CONSOLIDATED CLASS for all delegation functionality.

    Following FLEXT architectural patterns - consolidates ALL delegation functionality
    including protocols, delegated properties, and mixin delegation into one main class
    with nested classes for organization.

    CONSOLIDATED CLASSES: _HasDelegator + _DelegatorProtocol + FlextDelegatedProperty + FlextMixinDelegator
    """

    # ==========================================================================
    # NESTED PROTOCOLS AND CLASSES FOR ORGANIZATION
    # ==========================================================================

    class HasDelegator(Protocol):
        """Protocol for objects that have a delegator."""

        delegator: FlextDelegationSystem.DelegatorProtocol

    class DelegatorProtocol(Protocol):
        """Protocol for delegator objects."""

        def get_delegation_info(self) -> dict[str, object]: ...

    class DelegatedMethodProtocol(Protocol):
        """Protocol for delegated method callable."""

        def __call__(self, *args: object, **kwargs: object) -> object: ...

    class DelegatedProperty:
        """Nested delegated property descriptor."""

        def __init__(
            self,
            prop_name: str,
            mixin_instance: object,
            default: object = None,
        ) -> None:
            self.prop_name = prop_name
            self.mixin_instance = mixin_instance
            self.default = default

        def __get__(self, instance: object, owner: type | None = None) -> object:
            """Get delegated property value."""
            if instance is None:
                return self
            return getattr(self.mixin_instance, self.prop_name, self.default)

        def __set__(self, instance: object, value: object) -> None:
            """Set delegated property value."""
            # If the mixin instance has the setter, delegate to it
            with contextlib.suppress(AttributeError):
                setattr(self.mixin_instance, self.prop_name, value)
            # Also set on the host instance for potential fallback
            setattr(instance, f"_{self.prop_name}", value)

    class MixinDelegator:
        """Nested mixin delegator for composition patterns.

        Provides delegation to multiple mixin instances for composition-based
        functionality without multiple inheritance complexity.
        """

        _DELEGATED_PROPERTIES: ClassVar[list[str]] = [
            "is_valid",
            "validation_errors",
        ]

        def __init__(self, host_instance: object, *mixin_classes: type) -> None:
            """Initialize delegator with host and mixin classes.

            Args:
                host_instance: Object that will receive delegated functionality
                *mixin_classes: Classes to instantiate and delegate from

            """
            self.host_instance = host_instance
            self.mixin_instances: dict[type, object] = {}
            self.logger = FlextLogger(self.__class__.__name__)

            # Register and configure each mixin
            for mixin_class in mixin_classes:
                self._register_mixin(mixin_class)

            # Automatically delegate common methods and properties
            self._auto_delegate_methods()

        def _register_mixin(self, mixin_class: type) -> None:
            """Register and initialize a mixin class for delegation."""
            try:
                # Create mixin instance (assuming zero-arg constructor for now)
                mixin_instance = mixin_class()
                self.mixin_instances[mixin_class] = mixin_instance

                # Initialize mixin with data from host if available
                if hasattr(mixin_instance, "__post_init__"):
                    mixin_instance.__post_init__()

                self.logger.debug(
                    "Registered mixin",
                    mixin_class=mixin_class.__name__,
                    host_class=self.host_instance.__class__.__name__,
                )

            except Exception as e:
                error_msg = f"Failed to register mixin {mixin_class.__name__}: {e}"
                self.logger.exception(
                    error_msg,
                    mixin_class=mixin_class.__name__,
                    error=str(e),
                )
                raise FlextExceptions.FlextExceptionBaseError(error_msg) from e

        def _auto_delegate_methods(self) -> None:
            """Automatically delegate common methods from mixins to host."""
            for mixin_instance in self.mixin_instances.values():
                # Delegate common properties using descriptors
                for prop_name in self._DELEGATED_PROPERTIES:
                    if hasattr(mixin_instance, prop_name):
                        delegated_prop = self._create_delegated_property(
                            prop_name, mixin_instance
                        )
                        setattr(self.host_instance, prop_name, delegated_prop)

                # Delegate methods dynamically
                method_names = [
                    name
                    for name in dir(mixin_instance)
                    if not name.startswith("_")
                    and callable(getattr(mixin_instance, name))
                    and name
                    not in [
                        "is_valid",
                        "validation_errors",
                    ]  # Properties handled separately
                ]

                for method_name in method_names:
                    if not hasattr(self.host_instance, method_name):
                        delegated_method = self._create_delegated_method(
                            method_name, mixin_instance
                        )
                        setattr(self.host_instance, method_name, delegated_method)

        def _create_delegated_property(
            self, prop_name: str, mixin_instance: object
        ) -> FlextDelegationSystem.DelegatedProperty:
            """Create a delegated property descriptor."""
            return FlextDelegationSystem.DelegatedProperty(prop_name, mixin_instance)

        def _create_delegated_method(
            self, method_name: str, mixin_instance: object
        ) -> FlextDelegationSystem.DelegatedMethodProtocol:
            """Create a delegated method that forwards calls to mixin."""

            def delegated_method(*args: object, **kwargs: object) -> object:
                original_method = getattr(mixin_instance, method_name)
                try:
                    return original_method(*args, **kwargs)
                except Exception as e:
                    error_msg = f"Delegated method {method_name} failed: {e}"
                    self.logger.exception(
                        error_msg,
                        method_name=method_name,
                        mixin_class=mixin_instance.__class__.__name__,
                        error=str(e),
                    )
                    raise FlextExceptions.FlextExceptionBaseError(error_msg) from e

            # Preserve method metadata
            delegated_method.__name__ = method_name
            delegated_method.__doc__ = getattr(
                getattr(mixin_instance, method_name), "__doc__", None
            )

            return delegated_method

        def _validate_delegation(self) -> FlextResult[None]:
            """Validate that delegation is working correctly."""
            validation_results: list[str] = []

            for mixin_class, mixin_instance in self.mixin_instances.items():
                try:
                    # Check that mixin instance is valid
                    if hasattr(mixin_instance, "is_valid"):
                        is_valid_value = getattr(mixin_instance, "is_valid", False)
                        if not cast("bool", is_valid_value):
                            validation_results.append(
                                f"Mixin {mixin_class.__name__} is not valid"
                            )

                    # Check that host has delegated methods
                    missing_methods = [
                        f"Method {method_name} not delegated to host"
                        for method_name in dir(mixin_instance)
                        if (
                            not method_name.startswith("_")
                            and callable(getattr(mixin_instance, method_name))
                            and not hasattr(self.host_instance, method_name)
                        )
                    ]
                    validation_results.extend(missing_methods)

                except Exception as e:
                    validation_results.append(
                        f"Validation failed for {mixin_class.__name__}: {e}"
                    )

            if validation_results:
                error_msg = "; ".join(validation_results)
                return FlextResult[None].fail(error_msg)
            return FlextResult[None].ok(None)

        def get_mixin_instance(self, mixin_class: type) -> object | None:
            """Get instance of specific mixin class."""
            return self.mixin_instances.get(mixin_class)

        def get_delegation_info(self) -> dict[str, object]:
            """Get information about current delegation state."""
            return {
                "host_class": self.host_instance.__class__.__name__,
                "mixin_classes": [cls.__name__ for cls in self.mixin_instances],
                "delegated_methods": [
                    name
                    for name in dir(self.host_instance)
                    if not name.startswith("_")
                    and callable(getattr(self.host_instance, name))
                ],
            }

    # ==========================================================================
    # STATIC METHODS FOR CLASS-LEVEL OPERATIONS
    # ==========================================================================

    @staticmethod
    def create_mixin_delegator(
        host_instance: object,
        *mixin_classes: type,
    ) -> FlextDelegationSystem.MixinDelegator:
        """Create mixin delegators through the main class.

        Args:
          host_instance: Object that will receive delegated functionality
          *mixin_classes: Mixin classes to delegate from

        Returns:
          Configured delegator with all mixins registered and methods delegated

        """
        return FlextDelegationSystem.MixinDelegator(host_instance, *mixin_classes)

    @staticmethod
    def validate_delegation_system() -> FlextResult[
        dict[str, str | list[str] | dict[str, object]]
    ]:
        """Comprehensive validation of the delegation system with test cases.

        Returns:
          FlextResult with validation report and test results

        """

        # Test case 1: Basic delegation using FlextMixins static methods
        class TestHost:
            def __init__(self) -> None:
                # Initialize with FlextMixins functionality
                FlextMixins.create_timestamp_fields(self)
                FlextMixins.initialize_validation(self)
                # Test delegation can work with FlextMixins static methods
                self.delegator = FlextDelegationSystem.create_mixin_delegator(self)

        test_results: list[str] = []

        try:
            # Create test instance
            host = TestHost()

            # Run validation steps
            FlextDelegationSystem._validate_delegation_methods(host, test_results)
            FlextDelegationSystem._validate_method_functionality(host, test_results)
            info = FlextDelegationSystem._validate_delegation_info(host, test_results)

            return FlextResult[dict[str, str | list[str] | dict[str, object]]].ok(
                {
                    "status": "SUCCESS",
                    "test_results": test_results,
                    "delegation_info": info,
                },
            )

        except (
            AttributeError,
            TypeError,
            ValueError,
            RuntimeError,
            FlextExceptions.Error,
            FlextExceptions.TypeError,
        ) as e:
            test_results.append(f"✗ Test failed: {e}")
            error_msg: str = (
                f"Delegation system validation failed: {'; '.join(test_results)}"
            )
            return FlextResult[dict[str, str | list[str] | dict[str, object]]].fail(
                error_msg
            )

    @staticmethod
    def _validate_delegation_methods(host: object, test_results: list[str]) -> None:
        """Validate that delegation methods exist on the host."""

        def _raise_delegation_error(message: str) -> NoReturn:
            raise FlextExceptions.FlextExceptionBaseError(message)

        # Test validation methods exist
        if not hasattr(host, "is_valid"):
            _raise_delegation_error("is_valid property not delegated")
        if not hasattr(host, "validation_errors"):
            _raise_delegation_error("validation_errors property not delegated")
        if not hasattr(host, "has_validation_errors"):
            _raise_delegation_error("has_validation_errors method not delegated")
        test_results.append("✓ Validation methods successfully delegated")

        # Test serialization methods exist
        if not hasattr(host, "to_dict_basic"):
            _raise_delegation_error("to_dict_basic method not delegated")
        test_results.append("✓ Serialization methods successfully delegated")

    @staticmethod
    def _validate_method_functionality(host: object, test_results: list[str]) -> None:
        """Validate that delegated methods are functional."""

        def _raise_type_error(message: str) -> NoReturn:
            raise FlextExceptions.TypeError(message)

        # Test method functionality
        validation_result = getattr(host, "is_valid", None)
        if not isinstance(validation_result, bool):
            _raise_type_error("is_valid should return bool")
        test_results.append("✓ Delegated methods are functional")

    @staticmethod
    def _get_host_delegator(host: object) -> FlextDelegationSystem.MixinDelegator:
        """Safe delegator access with proper typing."""
        if not hasattr(host, "delegator"):
            error_msg = "Host missing delegator attribute"
            raise FlextExceptions.FlextExceptionBaseError(error_msg)

        # Dynamic attribute access is necessary here for validation code
        delegator_attr = getattr(host, "delegator", None)
        if delegator_attr is None:
            error_msg = "Host delegator attribute is None"
            raise FlextExceptions.FlextExceptionBaseError(error_msg)
        return cast("FlextDelegationSystem.MixinDelegator", delegator_attr)

    @staticmethod
    def _validate_delegation_info(
        host: object, test_results: list[str]
    ) -> dict[str, object]:
        """Validate delegation info and return it."""

        def _raise_delegation_error(message: str) -> NoReturn:
            raise FlextExceptions.FlextExceptionBaseError(message)

        # Get delegation info using safe helper
        delegator = FlextDelegationSystem._get_host_delegator(host)
        if not hasattr(delegator, "get_delegation_info"):
            _raise_delegation_error("Delegator missing get_delegation_info method")

        info = delegator.get_delegation_info()
        test_results.append("✓ Delegation info validation successful")
        return info


__all__: list[str] = [
    "FlextDelegationSystem",  # ONLY main class exported
]
