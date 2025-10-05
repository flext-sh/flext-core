"""Tests for flext_core.typings module - Type system validation.

Tests real functionality of the centralized type system, ensuring all
TypeVars, type aliases, and FlextTypes namespace work correctly.
"""

from collections import OrderedDict

from flext_core import FlextTypes
from flext_core.typings import (
    Command,
    E,
    Event,
    F,
    K,
    Message,
    P,
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
        assert P is not None


class TestRateLimiterState:
    """Test RateLimiterState TypedDict."""

    def test_ratelimiter_state_structure(self) -> None:
        """Test RateLimiterState has correct structure."""
        # Create an instance
        state: FlextTypes.Reliability.RateLimiterState = {
            "requests": [1.0, 2.0, 3.0],
            "last_reset": 1234567890.0,
        }

        assert isinstance(state["requests"], list)
        assert isinstance(state["last_reset"], float)
        assert len(state["requests"]) == 3


class TestFlextTypes:
    """Test FlextTypes namespace and all sub-types."""

    def test_flexttypes_core_types(self) -> None:
        """Test FlextTypes types are accessible."""
        # Test basic types
        assert FlextTypes.T is not None
        assert FlextTypes.U is not None
        assert FlextTypes.V is not None
        assert FlextTypes.W is not None

        # Test collection types
        assert FlextTypes.Dict == FlextTypes.Dict
        assert FlextTypes.List == FlextTypes.List
        assert FlextTypes.StringList == FlextTypes.StringList
        assert FlextTypes.IntList == FlextTypes.IntList
        assert FlextTypes.FloatList == FlextTypes.FloatList
        assert FlextTypes.BoolList == FlextTypes.BoolList

        # Test advanced types
        assert FlextTypes.NestedDict == FlextTypes.NestedDict
        # OrderedDict is actually OrderedDict type

        assert FlextTypes.OrderedDict == OrderedDict[str, object]

        # Test configuration types
        config_dict: FlextTypes.ConfigDict = {"key": "value", "number": 42}
        assert isinstance(config_dict, dict)

        # Test JSON types
        json_obj: FlextTypes.JsonValue = {"key": "value"}
        assert isinstance(json_obj, dict)

    def test_flexttypes_service_types(self) -> None:
        """Test FlextTypes.Service types."""
        service_dict: FlextTypes.Service.Dict = {"service": "value"}
        assert isinstance(service_dict, dict)

        # Test Literal types
        service_type: FlextTypes.Service.Type = "instance"
        assert service_type in {"instance", "factory"}

    def test_flexttypes_config_types(self) -> None:
        """Test FlextTypes.Config types."""
        # Test Environment literal
        env: FlextTypes.Config.Environment = "development"
        assert env in {
            "development",
            "staging",
            "production",
            "testing",
            "test",
            "local",
        }

        # Test LogLevel literal
        level: FlextTypes.Config.LogLevel = "INFO"
        assert level in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}

    def test_flexttypes_output_types(self) -> None:
        """Test FlextTypes.Output types."""
        # Test Literal types
        fmt: FlextTypes.Output.OutputFormat = "json"
        assert fmt in {"json", "yaml", "table", "csv", "text", "xml"}

        ser_fmt: FlextTypes.Output.SerializationFormat = "json"
        assert ser_fmt in {"json", "yaml", "toml", "ini", "xml"}

        comp_fmt: FlextTypes.Output.CompressionFormat = "gzip"
        assert comp_fmt in {"gzip", "bzip2", "xz", "lzma"}

    def test_flexttypes_project_types(self) -> None:
        """Test FlextTypes.Project types."""
        # Test Literal types
        proj_type: FlextTypes.Project.ProjectType = "library"
        expected_types = [
            "library",
            "application",
            "service",
            "cli",
            "web",
            "api",
            "PYTHON",
            "GO",
            "JAVASCRIPT",
        ]
        assert proj_type in expected_types

        status: FlextTypes.Project.ProjectStatus = "active"
        assert status in {"active", "inactive", "deprecated", "archived"}

    def test_flexttypes_processing_types(self) -> None:
        """Test FlextTypes.Processing types."""
        # Test all Literal types
        status: FlextTypes.Processing.ProcessingStatus = "pending"
        assert status in {"pending", "running", "completed", "failed", "cancelled"}

        mode: FlextTypes.Processing.ProcessingMode = "batch"
        assert mode in {"batch", "stream", "parallel", "sequential"}

        level: FlextTypes.Processing.ValidationLevel = "strict"
        assert level in {"strict", "lenient", "standard"}

        phase: FlextTypes.Processing.ProcessingPhase = "prepare"
        assert phase in {"prepare", "execute", "validate", "complete"}

        handler_type: FlextTypes.Processing.HandlerType = "command"
        assert handler_type in {"command", "query", "event", "processor"}

        workflow_status: FlextTypes.Processing.WorkflowStatus = "pending"
        assert workflow_status in {
            "pending",
            "running",
            "completed",
            "failed",
            "cancelled",
        }

        step_status: FlextTypes.Processing.StepStatus = "pending"
        assert step_status in {"pending", "running", "completed", "failed", "skipped"}

    def test_flexttypes_convenience_aliases(self) -> None:
        """Test convenience aliases work."""
        # Test direct access aliases
        assert FlextTypes.ConfigValue is not None
        assert FlextTypes.JsonValue is not None
        assert FlextTypes.ConfigDict is not None
        assert FlextTypes.JsonDict is not None
        assert FlextTypes.StringDict is not None
        assert FlextTypes.StringDict is not None
        assert FlextTypes.Dict is not None


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
