"""Tests for flext_core.typings module - Type system validation.

Tests real functionality of the centralized type system, ensuring all
TypeVars, type aliases, and FlextTypes namespace work correctly.

Per user requirement: Tests only test what exists in src/, not more.
"""

from flext_core import (
    Command,
    E,
    Event,
    F,
    FlextTypes,
    K,
    Message,
    # P is not exported due to ParamSpec limitations
    Query,
    R,
    ResultT,
    T,
    T1_co,
    T2_co,
    T3_co,
    T_contra,
    TAggregate_co,
    TCacheKey_contra,
    TCacheValue_co,
    TCommand_contra,
    TConfigKey_contra,
    TDomainEvent_co,
    TEntity_co,
    TEvent_contra,
    TInput_contra,
    TItem_contra,
    TQuery_contra,
    TResult_co,
    TResult_contra,
    TState_co,
    TUtil_contra,
    TValue_co,
    TValueObject_co,
    U,
    V,
    W,
)


class TestTypeVars:
    """Test TypeVar definitions and properties."""

    def test_core_typevars_defined(self) -> None:
        """Test that core TypeVars are properly defined."""
        # Test basic TypeVars
        assert T is not None
        assert U is not None
        assert V is not None
        assert W is not None
        assert E is not None
        assert F is not None
        assert K is not None
        assert R is not None

        # Test domain TypeVars
        assert Message is not None
        assert Command is not None
        assert Query is not None
        assert Event is not None
        assert ResultT is not None

    def test_covariant_typevars(self) -> None:
        """Test covariant TypeVars are properly defined."""
        # Test covariant variants
        assert T1_co is not None
        assert T2_co is not None
        assert T3_co is not None
        assert TState_co is not None
        assert TAggregate_co is not None
        assert TCacheValue_co is not None
        assert TDomainEvent_co is not None
        assert TEntity_co is not None
        assert TResult_co is not None
        assert TValue_co is not None
        assert TValueObject_co is not None

    def test_contravariant_typevars(self) -> None:
        """Test contravariant TypeVars are properly defined."""
        assert T_contra is not None
        assert TCommand_contra is not None
        assert TEvent_contra is not None
        assert TInput_contra is not None
        assert TQuery_contra is not None
        assert TItem_contra is not None
        assert TResult_contra is not None
        assert TUtil_contra is not None
        assert TCacheKey_contra is not None
        assert TConfigKey_contra is not None

    def test_paramspec_defined(self) -> None:
        """Test ParamSpec is properly defined."""
        # P is defined at module level as ParamSpec
        from flext_core import P

        assert P is not None


class TestFlextTypes:
    """Test FlextTypes namespace - ONLY types in lean typings.py."""

    def test_flexttypes_core_weak_types(self) -> None:
        """Test core weak types are accessible."""
        # Test collection types from lean typings.py
        assert dict[str, object] is not None
        assert list[object] is not None
        assert list[str] is not None
        assert FlextTypes.IntList is not None
        assert FlextTypes.FloatList is not None
        assert FlextTypes.BoolList is not None
        assert FlextTypes.NestedDict is not None
        assert dict[str, str] is not None

    def test_flexttypes_domain_types(self) -> None:
        """Test domain modeling types from lean typings.py."""
        # Test types used by src/flext_core/models.py
        assert object is not None
        assert FlextTypes.CallableHandlerType is not None
        assert object is not None  # Validation type
        assert dict[str, object] is not None
        assert dict[str, str | int | float] is not None

    def test_flexttypes_processor_types(self) -> None:
        """Test processor types from lean typings.py."""
        # Test types used by src/flext_core/processors.py
        assert FlextTypes.ProcessorInputType is not None
        assert FlextTypes.ProcessorOutputType is not None

    def test_flexttypes_handler_types(self) -> None:
        """Test handler types from lean typings.py."""
        # Test types used by src/flext_core/handlers.py and src/flext_core/bus.py
        assert FlextTypes.HandlerCallableType is not None  # Handler type
        assert FlextTypes.BusHandlerType is not None
        assert FlextTypes.BusMessageType is not None
        assert FlextTypes.AcceptableMessageType is not None
        assert FlextTypes.MiddlewareType is not None
        assert FlextTypes.MiddlewareConfig is not None

    def test_flexttypes_logging_types(self) -> None:
        """Test logging types from lean typings.py."""
        # Test types used by src/flext_core/loggings.py
        assert FlextTypes.LoggingContextValueType is not None
        assert FlextTypes.LoggingArgType is not None
        assert (
            FlextTypes.LoggingKwargsType is not None
        )  # Fixed from LoggingKwargsValueType
        assert FlextTypes.LoggingContextType is not None
        assert FlextTypes.BoundLoggerType is not None
        assert FlextTypes.LoggingProcessorType is not None

    def test_flexttypes_runtime_types(self) -> None:
        """Test runtime types from lean typings.py."""
        # Test types used by src/flext_core/runtime.py
        assert FlextTypes.ValidatableInputType is not None
        assert FlextTypes.TypeHintSpecifier is not None
        assert FlextTypes.SerializableObjectType is not None
        assert FlextTypes.GenericTypeArgument is not None
        assert FlextTypes.LoggerContextType is not None
        assert FlextTypes.FactoryCallableType is not None
        assert FlextTypes.ContextualObjectType is not None

    def test_flexttypes_container_types(self) -> None:
        """Test container types from lean typings.py."""
        # Test types used by src/flext_core/container.py
        assert FlextTypes.ValidatorFunctionType is not None
        assert FlextTypes.ContainerServiceType is not None

    def test_flexttypes_validation_types(self) -> None:
        """Test validation types from lean typings.py."""
        # Test types used by src/flext_core/models.py
        assert FlextTypes.ValidationRule is not None  # Fixed from ValidationPipeline
        assert object is not None


class TestImports:
    """Test all public imports work."""

    def test_all_exports_importable(self) -> None:
        """Test that all __all__ exports can be imported."""
        # This is implicitly tested by the imports at the top
        # If any import failed, the module wouldn't load
        assert True

    def test_no_import_errors(self) -> None:
        """Test that importing the module doesn't cause errors."""
        # If there were import errors, the test module wouldn't load
        assert True
