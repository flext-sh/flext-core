"""Comprehensive tests for FLEXT Core Types Module.

Tests all consolidated types functionality including:
- FlextTypes consolidated class and type system organization
- Core type variables (T, U, V, R, E, P, F) for generic programming
- Domain type variables (TEntity, TValue, TService) for business logic
- CQRS type variables (TCommand, TQuery, TEvent) for architectural patterns
- Protocol definitions (FlextValidatable, FlextSerializable, etc.)
- Type aliases (TAnyDict, TEntityId, TErrorMessage) for specific domains
- Type guards and runtime type checking utilities
- Legacy compatibility aliases and type system metadata
"""

from datetime import UTC
from typing import Any, TypeVar

from flext_core.types import (
    Comparable,
    E,
    F,
    FlextCacheable,
    FlextConfigurable,
    FlextEntityId,
    FlextErrorCode,
    FlextErrorMessage,
    FlextExecutable,
    FlextHandler,
    FlextIdentifiable,
    FlextSerializable,
    FlextTimestamped,
    FlextTransformer,
    FlextTypes,
    FlextUserId,
    FlextValidatable,
    FlextValidator,
    Identifiable,
    P,
    R,
    Serializable,
    T,
    TAnyDict,
    TAnyList,
    TAnyMapping,
    TAnySequence,
    TBusinessCode,
    TBusinessId,
    TBusinessName,
    TBusinessStatus,
    TBusinessType,
    TCacheKey,
    TCacheTTL,
    TCacheValue,
    TCommand,
    TConfigDict,
    TConfigValue,
    TConnectionString,
    TContextDict,
    TCorrelationId,
    TData,
    TDirectoryPath,
    TEntity,
    TEntityId,
    TErrorCode,
    TErrorHandler,
    TErrorMessage,
    TEvent,
    TFactory,
    TFieldId,
    TFieldName,
    TFilePath,
    THandler,
    Timestamped,
    TMapper,
    TMessage,
    TProcessor,
    TQuery,
    TRequest,
    TRequestId,
    TResponse,
    TResult,
    TService,
    TSessionId,
    TTransformer,
    TUserId,
    TValidator,
    TValue,
    U,
    V,
    Validatable,
)


class TestCoreTypeVariables:
    """Test core type variables for generic programming."""

    def test_core_type_variables_existence(self) -> None:
        """Test that all core type variables exist and are TypeVar instances."""
        # Test core type variables exist
        assert T is not None
        assert U is not None
        assert V is not None
        assert R is not None
        assert E is not None
        assert P is not None
        assert F is not None

        # Test they are TypeVar instances
        assert isinstance(T, TypeVar)
        assert isinstance(U, TypeVar)
        assert isinstance(V, TypeVar)
        assert isinstance(R, TypeVar)
        assert isinstance(E, TypeVar)
        assert isinstance(P, TypeVar)
        assert isinstance(F, TypeVar)

    def test_core_type_variables_names(self) -> None:
        """Test that core type variables have correct names."""
        assert T.__name__ == "T"
        assert U.__name__ == "U"
        assert V.__name__ == "V"
        assert R.__name__ == "R"
        assert E.__name__ == "E"
        assert P.__name__ == "P"
        assert F.__name__ == "F"

    def test_exception_type_variable_bound(self) -> None:
        """Test that E type variable is bound to Exception."""
        # Test E is bound to Exception
        assert E.__bound__ is Exception

    def test_type_variables_generic_usage(self) -> None:
        """Test that type variables can be used in generic contexts."""
        # This is more of a static analysis test, but we can verify they work
        from typing import Generic

        class TestGeneric(Generic[T]):
            def __init__(self, value: T) -> None:
                self.value = value

        # These should work without errors
        test_str = TestGeneric[str]("test")
        test_int = TestGeneric[int](42)

        assert test_str.value == "test"
        assert test_int.value == 42


class TestDomainTypeVariables:
    """Test domain-specific type variables."""

    def test_domain_type_variables_exist(self) -> None:
        """Test that domain type variables exist."""
        from flext_core.types import (
            TCommand,
            TData,
            TEntity,
            TEvent,
            TMessage,
            TResult,
            TService,
            TValue,
        )

        # Test domain variables exist
        assert TEntity is not None
        assert TValue is not None
        assert TService is not None
        assert TCommand is not None
        assert TData is not None
        assert TEvent is not None
        assert TMessage is not None
        assert TRequest is not None
        assert TResponse is not None
        assert TResult is not None
        assert TQuery is not None

        # Test they are TypeVar instances
        assert isinstance(TEntity, TypeVar)
        assert isinstance(TValue, TypeVar)
        assert isinstance(TService, TypeVar)
        assert isinstance(TCommand, TypeVar)
        assert isinstance(TData, TypeVar)
        assert isinstance(TEvent, TypeVar)
        assert isinstance(TMessage, TypeVar)
        assert isinstance(TRequest, TypeVar)
        assert isinstance(TResponse, TypeVar)
        assert isinstance(TResult, TypeVar)
        assert isinstance(TQuery, TypeVar)

    def test_domain_type_variables_names(self) -> None:
        """Test that domain type variables have correct names."""
        from flext_core.types import (
            TCommand,
            TData,
            TEntity,
            TEvent,
            TMessage,
            TResult,
            TService,
            TValue,
        )

        assert TEntity.__name__ == "TEntity"
        assert TValue.__name__ == "TValue"
        assert TService.__name__ == "TService"
        assert TCommand.__name__ == "TCommand"
        assert TData.__name__ == "TData"
        assert TEvent.__name__ == "TEvent"
        assert TMessage.__name__ == "TMessage"
        assert TRequest.__name__ == "TRequest"
        assert TResponse.__name__ == "TResponse"
        assert TResult.__name__ == "TResult"
        assert TQuery.__name__ == "TQuery"


class TestTypeAliases:
    """Test type aliases for common patterns."""

    def test_basic_type_aliases_exist(self) -> None:
        """Test that basic type aliases exist and have correct types."""
        # Test basic collection aliases
        assert TAnyDict is not None
        assert TAnyList is not None
        assert TAnyMapping is not None
        assert TAnySequence is not None

    def test_domain_specific_aliases(self) -> None:
        """Test domain-specific type aliases."""
        # Test ID aliases
        assert TEntityId is not None
        assert TUserId is not None
        assert TRequestId is not None
        assert TSessionId is not None
        assert TFieldId is not None

        # Test message aliases
        assert TErrorCode is not None
        assert TErrorMessage is not None
        assert TFieldName is not None
        assert TContextDict is not None

        # Test business aliases
        assert TBusinessId is not None
        assert TBusinessName is not None
        assert TBusinessCode is not None
        assert TBusinessStatus is not None
        assert TBusinessType is not None

    def test_functional_type_aliases(self) -> None:
        """Test functional programming type aliases."""
        # Test functional types exist
        assert TValidator is not None
        assert TTransformer is not None
        assert THandler is not None
        assert TProcessor is not None
        assert TMapper is not None
        assert TFactory is not None
        assert TErrorHandler is not None

    def test_infrastructure_type_aliases(self) -> None:
        """Test infrastructure-related type aliases."""
        # Test cache types
        assert TCacheKey is not None
        assert TCacheValue is not None
        assert TCacheTTL is not None

        # Test config types
        assert TConfigValue is not None
        assert TConfigDict is not None

        # Test path types
        assert TFilePath is not None
        assert TDirectoryPath is not None

        # Test database types
        assert TConnectionString is not None

        # Test correlation types
        assert TCorrelationId is not None


class TestProtocolDefinitions:
    """Test protocol definitions for structural typing."""

    def test_flext_validatable_protocol(self) -> None:
        """Test FlextValidatable protocol structure."""
        # Test protocol has required method
        assert hasattr(FlextValidatable, "validate")

        # Test we can create a class that implements the protocol
        class TestValidatable:
            def validate(self) -> bool:
                return True

        test_obj = TestValidatable()
        assert test_obj.validate() is True

    def test_flext_serializable_protocol(self) -> None:
        """Test FlextSerializable protocol structure."""
        # Test protocol has required method
        assert hasattr(FlextSerializable, "to_dict")

        # Test we can create a class that implements the protocol
        class TestSerializable:
            def to_dict(self) -> dict[str, object]:
                return {"test": "value"}

        test_obj = TestSerializable()
        assert test_obj.to_dict() == {"test": "value"}

    def test_flext_identifiable_protocol(self) -> None:
        """Test FlextIdentifiable protocol structure."""
        # Test protocol has required property
        assert hasattr(FlextIdentifiable, "id")

        # Test we can create a class that implements the protocol
        class TestIdentifiable:
            @property
            def id(self) -> str:
                return "test_id"

        test_obj = TestIdentifiable()
        assert test_obj.id == "test_id"

    def test_flext_timestamped_protocol(self) -> None:
        """Test FlextTimestamped protocol structure."""
        # Test protocol has required properties
        assert hasattr(FlextTimestamped, "created_at")
        assert hasattr(FlextTimestamped, "updated_at")

        # Test we can create a class that implements the protocol
        from datetime import datetime

        class TestTimestamped:
            @property
            def created_at(self) -> datetime:
                return datetime.now(UTC)

            @property
            def updated_at(self) -> datetime:
                return datetime.now(UTC)

        test_obj = TestTimestamped()
        assert isinstance(test_obj.created_at, datetime)
        assert isinstance(test_obj.updated_at, datetime)

    def test_flext_cacheable_protocol(self) -> None:
        """Test FlextCacheable protocol structure."""
        # Test protocol has required method
        assert hasattr(FlextCacheable, "cache_key")

        # Test we can create a class that implements the protocol
        class TestCacheable:
            def cache_key(self) -> str:
                return "cache_key_123"

        test_obj = TestCacheable()
        assert test_obj.cache_key() == "cache_key_123"

    def test_flext_configurable_protocol(self) -> None:
        """Test FlextConfigurable protocol structure."""
        # Test protocol has required method
        assert hasattr(FlextConfigurable, "configure")

        # Test we can create a class that implements the protocol
        class TestConfigurable:
            def __init__(self) -> None:
                self.config: dict[str, Any] = {}

            def configure(self, config: dict[str, Any]) -> None:
                self.config = config

        test_obj = TestConfigurable()
        test_config = {"setting": "value"}
        test_obj.configure(test_config)
        assert test_obj.config == test_config

    def test_operational_protocols(self) -> None:
        """Test operational protocols (Executable, Validator, Transformer, Handler)."""
        # Test FlextExecutable
        assert hasattr(FlextExecutable, "execute")

        # Test FlextValidator
        assert hasattr(FlextValidator, "validate")

        # Test FlextTransformer
        assert hasattr(FlextTransformer, "transform")

        # Test FlextHandler
        assert hasattr(FlextHandler, "handle")

    def test_comparable_protocol(self) -> None:
        """Test Comparable protocol structure."""
        # Test protocol has required comparison methods
        assert hasattr(Comparable, "__lt__")
        assert hasattr(Comparable, "__le__")
        assert hasattr(Comparable, "__gt__")
        assert hasattr(Comparable, "__ge__")


class TestFlextTypesMainClass:
    """Test FlextTypes consolidated class."""

    def test_flext_types_structure(self) -> None:
        """Test FlextTypes class has all required attributes."""
        # Test type variables
        assert hasattr(FlextTypes, "T")
        assert hasattr(FlextTypes, "U")
        assert hasattr(FlextTypes, "V")
        assert hasattr(FlextTypes, "R")
        assert hasattr(FlextTypes, "E")
        assert hasattr(FlextTypes, "P")
        assert hasattr(FlextTypes, "F")

        # Test domain variables
        assert hasattr(FlextTypes, "TEntity")
        assert hasattr(FlextTypes, "TValue")
        assert hasattr(FlextTypes, "TService")
        assert hasattr(FlextTypes, "TCommand")
        assert hasattr(FlextTypes, "TQuery")

        # Test type aliases
        assert hasattr(FlextTypes, "TAnyDict")
        assert hasattr(FlextTypes, "TEntityId")
        assert hasattr(FlextTypes, "TErrorMessage")

    def test_flext_types_nested_classes(self) -> None:
        """Test FlextTypes nested classes."""
        # Test TypeGuards nested class exists
        assert hasattr(FlextTypes, "TypeGuards")
        assert hasattr(FlextTypes.TypeGuards, "is_instance_of")

        # Test Meta nested class exists
        assert hasattr(FlextTypes, "Meta")
        assert hasattr(FlextTypes.Meta, "TYPE_COUNT")
        assert hasattr(FlextTypes.Meta, "VERSION")
        assert hasattr(FlextTypes.Meta, "COMPATIBILITY")

    def test_flext_types_type_guards(self) -> None:
        """Test FlextTypes.TypeGuards functionality."""
        # Test is_instance_of method
        result = FlextTypes.TypeGuards.is_instance_of("test", str)
        assert result is True

        result = FlextTypes.TypeGuards.is_instance_of("test", int)
        assert result is False

        result = FlextTypes.TypeGuards.is_instance_of(42, int)
        assert result is True

    def test_flext_types_meta_information(self) -> None:
        """Test FlextTypes.Meta information."""
        # Test meta attributes exist and have reasonable values
        assert isinstance(FlextTypes.Meta.TYPE_COUNT, int)
        assert FlextTypes.Meta.TYPE_COUNT > 0

        assert isinstance(FlextTypes.Meta.VERSION, str)
        assert len(FlextTypes.Meta.VERSION) > 0

        assert isinstance(FlextTypes.Meta.COMPATIBILITY, str)
        assert "Python" in FlextTypes.Meta.COMPATIBILITY

    def test_flext_types_value_consistency(self) -> None:
        """Test FlextTypes values are consistent with direct exports."""
        # Test type variables consistency
        assert FlextTypes.T is T
        assert FlextTypes.U is U
        assert FlextTypes.V is V
        assert FlextTypes.R is R
        assert FlextTypes.E is E
        assert FlextTypes.P is P
        assert FlextTypes.F is F

    def test_flext_types_nested_type_access(self) -> None:
        """Test accessing types through FlextTypes nested structure."""
        # Test we can access all major type categories
        from flext_core.types import TCommand, TEntity, TValue

        assert FlextTypes.TEntity is TEntity
        assert FlextTypes.TValue is TValue
        assert FlextTypes.TCommand is TCommand
        assert FlextTypes.TQuery is TQuery


class TestLegacyCompatibility:
    """Test legacy compatibility aliases."""

    def test_legacy_protocol_aliases(self) -> None:
        """Test legacy protocol aliases work correctly."""
        # Test protocol aliases exist and point to correct protocols
        assert Identifiable is FlextIdentifiable
        assert Serializable is FlextSerializable
        assert Timestamped is FlextTimestamped
        assert Validatable is FlextValidatable

    def test_legacy_type_aliases(self) -> None:
        """Test legacy type aliases work correctly."""
        # Test legacy entity types
        assert FlextEntityId == TEntityId
        assert FlextUserId == TUserId
        assert FlextErrorCode == TErrorCode
        assert FlextErrorMessage == TErrorMessage

        # Test these are string types

        # These should be string type aliases
        assert TEntityId is str
        assert TUserId is str
        assert TErrorCode is str
        assert TErrorMessage is str

    def test_legacy_compatibility_functionality(self) -> None:
        """Test legacy aliases provide same functionality."""

        # Test we can use legacy aliases in the same way
        class TestLegacyIdentifiable:
            @property
            def id(self) -> TEntityId:
                return "legacy_id"

        class TestLegacySerializable:
            def to_dict(self) -> dict[str, object]:
                return {"legacy": True}

        identifiable_obj = TestLegacyIdentifiable()
        serializable_obj = TestLegacySerializable()

        assert identifiable_obj.id == "legacy_id"
        assert serializable_obj.to_dict() == {"legacy": True}


class TestTypeSystemIntegration:
    """Test type system integration scenarios."""

    def test_generic_result_type_usage(self) -> None:
        """Test using type system with generic Result patterns."""
        from typing import Generic

        # Test generic class with type variables
        class TestResult(Generic[T]):
            def __init__(self, value: T) -> None:
                self.value = value

            def get_value(self) -> T:
                return self.value

        # Test with different types
        string_result = TestResult[str]("test")
        int_result = TestResult[int](42)

        assert string_result.get_value() == "test"
        assert int_result.get_value() == 42

    def test_protocol_integration_scenario(self) -> None:
        """Test protocols working together in integration scenarios."""
        from datetime import datetime

        class TestEntity:
            def __init__(self, entity_id: TEntityId) -> None:
                self._id = entity_id
                self._created = datetime.now(UTC)

            @property
            def id(self) -> TEntityId:
                return self._id

            @property
            def created_at(self) -> datetime:
                return self._created

            @property
            def updated_at(self) -> datetime:
                return self._created

            def validate(self) -> bool:
                return len(self._id) > 0

            def to_dict(self) -> dict[str, object]:
                return {
                    "id": self._id,
                    "created_at": self._created.isoformat(),
                }

        # Test entity works with multiple protocols
        entity = TestEntity("test_entity_123")

        # Should work as FlextIdentifiable
        assert entity.id == "test_entity_123"

        # Should work as FlextTimestamped
        assert isinstance(entity.created_at, datetime)
        assert isinstance(entity.updated_at, datetime)

        # Should work as FlextValidatable
        assert entity.validate() is True

        # Should work as FlextSerializable
        entity_dict = entity.to_dict()
        assert entity_dict["id"] == "test_entity_123"
        assert "created_at" in entity_dict

    def test_functional_type_integration(self) -> None:
        """Test functional types in integration scenarios."""

        # Test using functional type aliases
        def create_validator(validation_func: TValidator) -> TValidator:
            return validation_func

        def create_transformer(transform_func: TTransformer) -> TTransformer:
            return transform_func

        def create_handler(handler_func: THandler) -> THandler:
            return handler_func

        # Test creating functions with type aliases
        validator = create_validator(lambda x: isinstance(x, str))
        transformer = create_transformer(lambda x: str(x).upper())
        handler = create_handler(lambda event: f"Handled: {event}")

        assert validator("test") is True
        assert validator(123) is False
        assert transformer("hello") == "HELLO"
        assert handler("test_event") == "Handled: test_event"

    def test_cqrs_pattern_integration(self) -> None:
        """Test CQRS type variables in architectural patterns."""
        from typing import Generic

        class TestCommand(Generic[TData]):
            def __init__(self, data: TData) -> None:
                self.data = data

        class TestQuery(Generic[TResult]):
            def __init__(self, result_type: type[TResult]) -> None:
                self.result_type = result_type

        class TestEvent(Generic[TMessage]):
            def __init__(self, message: TMessage) -> None:
                self.message = message

        # Test CQRS components
        command = TestCommand[dict[str, str]]({"action": "create", "data": "test"})
        query = TestQuery[str](str)
        event = TestEvent[str]("entity_created")

        assert command.data == {"action": "create", "data": "test"}
        assert query.result_type is str
        assert event.message == "entity_created"


class TestTypeSystemEdgeCases:
    """Test edge cases and boundary conditions for type system."""

    def test_type_variable_bounds_and_constraints(self) -> None:
        """Test type variable bounds and constraints."""
        # Test E is bound to Exception
        assert E.__bound__ is Exception

        # Test other type variables are unconstrained
        assert T.__bound__ is None
        assert U.__bound__ is None
        assert V.__bound__ is None
        assert R.__bound__ is None
        assert P.__bound__ is None
        assert F.__bound__ is None

    def test_protocol_method_signatures(self) -> None:
        """Test protocol method signatures are correctly defined."""
        import inspect

        # Test FlextValidatable.validate signature
        validate_sig = inspect.signature(FlextValidatable.validate)
        assert len(validate_sig.parameters) == 1  # only self

        # Test FlextSerializable.to_dict signature
        to_dict_sig = inspect.signature(FlextSerializable.to_dict)
        assert len(to_dict_sig.parameters) == 1  # only self

    def test_type_alias_consistency(self) -> None:
        """Test type alias consistency across the system."""
        # Test ID types are all strings
        assert TEntityId is str
        assert TUserId is str
        assert TRequestId is str
        assert TSessionId is str
        assert TFieldId is str

        # Test error types are strings
        assert TErrorCode is str
        assert TErrorMessage is str

        # Test business types are strings
        assert TBusinessId is str
        assert TBusinessName is str
        assert TBusinessCode is str
        assert TBusinessStatus is str
        assert TBusinessType is str

    def test_type_system_completeness(self) -> None:
        """Test type system provides comprehensive coverage."""
        # Test we have types for major categories

        # ID and identification types
        id_types = [TEntityId, TUserId, TRequestId, TSessionId, TFieldId]
        assert all(t is str for t in id_types)

        # Business domain types
        business_types = [
            TBusinessId,
            TBusinessName,
            TBusinessCode,
            TBusinessStatus,
            TBusinessType,
        ]
        assert all(t is str for t in business_types)

        # Infrastructure types
        infra_types = [TCacheKey, TFilePath, TDirectoryPath, TConnectionString]
        assert all(t is str for t in infra_types)

        # Correlation and context types
        assert TCorrelationId is str
        assert TContextDict == dict[str, object]

    def test_protocol_inheritance_patterns(self) -> None:
        """Test protocol inheritance and composition patterns."""
        # Test we can create classes implementing multiple protocols
        from datetime import datetime

        class FullEntity:
            def __init__(self, entity_id: str) -> None:
                self._id = entity_id
                self._created = datetime.now(UTC)
                self._config: dict[str, object] = {}

            # FlextIdentifiable
            @property
            def id(self) -> str:
                return self._id

            # FlextTimestamped
            @property
            def created_at(self) -> datetime:
                return self._created

            @property
            def updated_at(self) -> datetime:
                return self._created

            # FlextValidatable
            def validate(self) -> bool:
                return len(self._id) > 0

            # FlextSerializable
            def to_dict(self) -> dict[str, object]:
                return {"id": self._id, "created_at": self._created.isoformat()}

            # FlextConfigurable
            def configure(self, config: dict[str, object]) -> None:
                self._config = config

            # FlextCacheable
            def cache_key(self) -> str:
                return f"entity_{self._id}"

        entity = FullEntity("multi_protocol_entity")

        # Test all protocol methods work
        assert entity.id == "multi_protocol_entity"
        assert isinstance(entity.created_at, datetime)
        assert entity.validate() is True
        assert "id" in entity.to_dict()

        entity.configure({"setting": "value"})
        assert entity._config == {"setting": "value"}

        assert entity.cache_key() == "entity_multi_protocol_entity"


class TestTypeSystemDocumentation:
    """Test type system documentation and metadata."""

    def test_type_system_metadata_completeness(self) -> None:
        """Test type system metadata provides complete information."""
        # Test Meta class has comprehensive information
        assert isinstance(FlextTypes.Meta.TYPE_COUNT, int)
        assert FlextTypes.Meta.TYPE_COUNT > 50  # Should have many types

        assert isinstance(FlextTypes.Meta.VERSION, str)
        assert "." in FlextTypes.Meta.VERSION  # Should be semantic version

        assert isinstance(FlextTypes.Meta.COMPATIBILITY, str)
        assert "Python" in FlextTypes.Meta.COMPATIBILITY

    def test_protocol_documentation_exists(self) -> None:
        """Test protocols have proper documentation."""
        # Test major protocols have docstrings
        protocols_with_docs = [
            FlextValidatable,
            FlextSerializable,
            FlextIdentifiable,
            FlextTimestamped,
            FlextCacheable,
            FlextConfigurable,
        ]

        for protocol in protocols_with_docs:
            assert protocol.__doc__ is not None
            assert len(protocol.__doc__) > 0

    def test_type_system_export_completeness(self) -> None:
        """Test type system exports all necessary types."""
        from flext_core import types

        # Test main type categories are exported
        main_exports = [
            "T",
            "U",
            "V",
            "R",
            "E",
            "P",
            "F",  # Core type variables
            "FlextTypes",  # Main type class
            "TAnyDict",
            "TEntityId",
            "TErrorMessage",  # Common aliases
            "FlextValidatable",
            "FlextSerializable",  # Core protocols
        ]

        for export_name in main_exports:
            assert hasattr(types, export_name)

    def test_type_system_naming_consistency(self) -> None:
        """Test type system follows consistent naming patterns."""
        # Test type variables follow T prefix pattern
        type_vars = [TEntity, TValue, TService, TCommand, TData, TEvent, TMessage]
        for type_var in type_vars:
            assert type_var.__name__.startswith("T")

        # Test protocols follow Flext prefix pattern
        protocol_names = [
            FlextValidatable.__name__,
            FlextSerializable.__name__,
            FlextIdentifiable.__name__,
            FlextTimestamped.__name__,
            FlextCacheable.__name__,
            FlextConfigurable.__name__,
        ]

        for name in protocol_names:
            assert name.startswith("Flext")

    def test_type_system_version_compatibility(self) -> None:
        """Test type system maintains version compatibility."""
        # Test version information is accessible
        version = FlextTypes.Meta.VERSION
        compatibility = FlextTypes.Meta.COMPATIBILITY

        assert isinstance(version, str)
        assert len(version.split(".")) >= 2  # At least major.minor

        assert isinstance(compatibility, str)
        assert "3.13" in compatibility or "Python" in compatibility
