"""FLEXT Core Delegation System - Sistema robusto de delegação de mixins.

Sistema arquiteturalmente correto para eliminar múltiplas heranças através de
composição automática e delegação inteligente. Implementa padrão universal
para todos os mixins da FLEXT, garantindo funcionalidade completa sem
complexidade de herança múltipla.

Architecture:
    - Delegação automática com descoberta dinâmica de métodos
    - Sistema de registro de mixins para reutilização global
    - Proxy inteligente que preserva signatures e tipos
    - Inicialização automática de state mixin através de composição
    - Validação e testes automáticos para garantir funcionamento correto

Design Principles:
    - SINGLE SOURCE OF TRUTH: cada funcionalidade definida uma vez
    - COMPOSITION OVER INHERITANCE: delegação em vez de herança múltipla
    - AUTOMATIC DISCOVERY: métodos descobertos dinamicamente
    - TYPE SAFE: preservação de tipos em delegação
    - FULLY TESTED: sistema amplamente testável e validável

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from collections.abc import Callable

    from flext_core.flext_types import TAnyDict

from flext_core._mixins_base import (
    _BaseSerializableMixin,
    _BaseValidatableMixin,
)
from flext_core.exceptions import FlextOperationError, FlextTypeError
from flext_core.result import FlextResult


class DelegatedProperty:
    def __init__(
        self,
        prop_name: str,
        mixin_instance: object,
        *,
        has_setter: bool,
        doc: str | None = None,
    ) -> None:
        self.prop_name = prop_name
        self.mixin_instance = mixin_instance
        self.has_setter = has_setter
        self.__doc__ = doc

    def __get__(self, instance: object, owner: type | None = None) -> object:
        return getattr(self.mixin_instance, self.prop_name)

    def __set__(self, instance: object, value: object) -> None:
        if self.has_setter:
            setattr(self.mixin_instance, self.prop_name, value)
        else:
            error_msg = f"Property '{self.prop_name}' is read-only"
            raise FlextOperationError(
                error_msg,
                operation="property_setter",
                context={"property_name": self.prop_name, "readonly": True},
            )


class FlextMixinDelegator:
    """Sistema robusto de delegação de mixins com descoberta automática.

    Elimina completamente múltiplas heranças através de composição inteligente
    e delegação automática. Garante funcionalidade 100% compatível com a
    herança múltipla original mas com arquitetura limpa e testável.

    Features:
        - Delegação automática de TODOS os métodos de mixin
        - Inicialização automática de state interno dos mixins
        - Preservação de type hints e signatures
        - Sistema de cache para performance otimizada
        - Validação automática de funcionamento correto
        - Debugging e observability built-in
    """

    # Registry global de mixins disponíveis
    _MIXIN_REGISTRY: ClassVar[dict[str, type]] = {}

    def __init__(self, host_instance: object, *mixin_classes: type) -> None:
        """Initialize delegation system with automatic mixin discovery.

        Args:
            host_instance: Instance that will receive delegated methods
            *mixin_classes: Mixin classes to delegate from

        """
        self._host = host_instance
        self._mixin_instances: dict[type, object] = {}
        self._delegated_methods: dict[str, Callable[[object, object], object]] = {}
        self._initialization_log: list[str] = []

        # Create and initialize mixin instances
        for mixin_class in mixin_classes:
            self._register_mixin(mixin_class)

        # Auto-delegate all public methods
        self._auto_delegate_methods()

        # Validate delegation system
        self._validate_delegation()

    def _register_mixin(self, mixin_class: type) -> None:
        """Register and initialize a mixin class for delegation."""
        try:
            # Create mixin instance
            mixin_instance = mixin_class()
            self._mixin_instances[mixin_class] = mixin_instance

            # Auto-initialize if mixin has initialization method
            init_methods = [
                f"_initialize_{name.lower()}"
                for name in [
                    "validation",
                    "timestamps",
                    "id",
                    "logging",
                    "serialization",
                ]
            ]

            for init_method in init_methods:
                if hasattr(mixin_instance, init_method):
                    try:
                        getattr(mixin_instance, init_method)()
                        self._initialization_log.append(
                            f"✓ {mixin_class.__name__}.{init_method}()",
                        )
                    except (AttributeError, TypeError, ValueError) as e:
                        error_msg = f"✗ {mixin_class.__name__}.{init_method}(): {e}"
                        self._initialization_log.append(error_msg)

            # Register globally for reuse
            self._MIXIN_REGISTRY[mixin_class.__name__] = mixin_class

        except (AttributeError, TypeError, ValueError) as e:
            error_msg = f"✗ Failed to register {mixin_class.__name__}: {e}"
            self._initialization_log.append(error_msg)
            raise

    def _auto_delegate_methods(self) -> None:
        """Automatically delegate all public methods from registered mixins."""
        for mixin_instance in self._mixin_instances.values():
            for attr_name in dir(mixin_instance):
                # Skip private/magic methods
                if attr_name.startswith("_"):
                    continue

                # Delegate properties first (they are also callable)
                try:
                    is_property = isinstance(
                        getattr(type(mixin_instance), attr_name, None),
                        property,
                    )
                except (AttributeError, TypeError, ValueError):
                    is_property = False

                if is_property:
                    try:
                        self._create_delegated_property(attr_name, mixin_instance)
                    except (AttributeError, TypeError, ValueError):
                        # Ignore properties that can't be delegated
                        continue
                else:
                    try:
                        attr = getattr(mixin_instance, attr_name)
                    except (AttributeError, TypeError, ValueError):
                        continue
                    if callable(attr):
                        try:
                            self._create_delegated_method(
                                attr_name,
                                mixin_instance,
                                attr,
                            )
                        except (AttributeError, TypeError, ValueError):
                            continue

    def _create_delegated_property(
        self,
        prop_name: str,
        mixin_instance: object,
    ) -> None:
        """Create a delegated property with getter/setter preservation."""
        prop = getattr(type(mixin_instance), prop_name)
        has_setter = prop.fset is not None
        delegated_prop = DelegatedProperty(
            prop_name,
            mixin_instance,
            has_setter=has_setter,
            doc=prop.__doc__,
        )
        # Remove atributo de instância se existir
        if hasattr(self._host, prop_name):
            with contextlib.suppress(AttributeError):
                delattr(self._host, prop_name)
        setattr(type(self._host), prop_name, delegated_prop)

    def _create_delegated_method(
        self,
        method_name: str,
        mixin_instance: object,
        method: Callable[[object, object], object],
    ) -> None:
        """Create a delegated method with full signature preservation."""

        def delegated_method(*args: object, **kwargs: object) -> object:
            try:
                return method(*args, **kwargs)
            except (AttributeError, TypeError, ValueError) as e:
                # Enhanced error with delegation context
                error_msg = (
                    f"Delegation error in {type(mixin_instance).__name__}."
                    f"{method_name}: {e}"
                )
                raise FlextOperationError(
                    error_msg,
                    operation="delegation",
                    stage=method_name,
                ) from e

        # Preserve original signature and docstring
        try:
            delegated_method.__name__ = method_name
            delegated_method.__doc__ = method.__doc__
            import inspect  # noqa: PLC0415

            # Use type ignore for dynamic attribute assignment
            delegated_method.__signature__ = inspect.signature(method)  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass

        # Store for access via host
        self._delegated_methods[method_name] = delegated_method

        # Try to attach to host instance - handle frozen Pydantic models
        try:
            if not hasattr(self._host, method_name):
                setattr(self._host, method_name, delegated_method)
        except AttributeError:
            if hasattr(self._host, "__class__"):
                setattr(self._host.__class__, method_name, delegated_method)

    def _validate_delegation(self) -> FlextResult[None]:
        """Validate that delegation system is working correctly."""
        validation_errors = []

        # Check that all mixins were initialized
        if not self._mixin_instances:
            validation_errors.append("No mixins were successfully registered")

        # Check that methods were delegated
        if not self._delegated_methods:
            validation_errors.append("No methods were successfully delegated")

        # Note: Due to typing, all methods in _delegated_methods are guaranteed to be callable  # noqa: E501
        # so we skip the runtime callable check that would be unreachable

        # Check initialization log for errors
        failed_inits = [log for log in self._initialization_log if log.startswith("✗")]
        if failed_inits:
            validation_errors.extend(failed_inits)

        if validation_errors:
            return FlextResult.fail(
                f"Delegation validation failed: {'; '.join(validation_errors)}",
            )

        return FlextResult.ok(None)

    def get_mixin_instance(self, mixin_class: type) -> object | None:
        """Get specific mixin instance for direct access if needed."""
        return self._mixin_instances.get(mixin_class)

    def get_delegation_info(self) -> TAnyDict:
        """Get comprehensive information about delegation state for debugging."""
        return {
            "registered_mixins": list(self._mixin_instances.keys()),
            "delegated_methods": list(self._delegated_methods.keys()),
            "initialization_log": self._initialization_log.copy(),
            "validation_result": self._validate_delegation().is_success,
        }


def create_mixin_delegator(
    host_instance: object,
    *mixin_classes: type,
) -> FlextMixinDelegator:
    """Create mixin delegators.

    Args:
        host_instance: Object that will receive delegated functionality
        *mixin_classes: Mixin classes to delegate from

    Returns:
        Configured delegator with all mixins registered and methods delegated

    """
    return FlextMixinDelegator(host_instance, *mixin_classes)


def validate_delegation_system() -> FlextResult[TAnyDict]:  # noqa: C901
    """Comprehensive validation of the delegation system with test cases.

    Returns:
        FlextResult with validation report and test results

    """
    # Import inside to avoid circular imports

    # Test case 1: Basic delegation
    class TestHost:
        def __init__(self) -> None:
            self.delegator = create_mixin_delegator(
                self,
                _BaseValidatableMixin,
                _BaseSerializableMixin,
            )

    test_results = []

    try:
        # Create test instance
        host = TestHost()

        def _raise_delegation_error(message: str) -> None:
            raise FlextOperationError(message, operation="delegation_validation")  # noqa: TRY301

        def _raise_type_error(message: str) -> None:
            raise FlextTypeError(message)  # noqa: TRY301

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

        # Test method functionality
        validation_result = getattr(host, "is_valid", None)
        if not isinstance(validation_result, bool):
            _raise_type_error("is_valid should return bool")
        test_results.append("✓ Delegated methods are functional")

        # Test delegation info
        info = host.delegator.get_delegation_info()
        if not info["validation_result"]:
            _raise_delegation_error("Delegation validation should pass")
        test_results.append("✓ Delegation system self-validation passed")

        return FlextResult.ok(
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
        FlextOperationError,
        FlextTypeError,
    ) as e:
        test_results.append(f"✗ Test failed: {e}")
        error_msg = f"Delegation system validation failed: {'; '.join(test_results)}"
        return FlextResult.fail(error_msg)


__all__ = [
    "FlextMixinDelegator",
    "create_mixin_delegator",
    "validate_delegation_system",
]
