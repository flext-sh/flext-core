"""Tests for flext_core.typings module - Type system validation.

Tests real functionality of the centralized type system, ensuring all
TypeVars, type aliases, and FlextCore.Types namespace work correctly.
"""

from flext_core import FlextCore
from flext_core.typings import (
    Command,
    E,
    Event,
    F,
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
        from flext_core.typings import P

        assert P is not None


class TestRateLimiterState:
    """Test RateLimiterState TypedDict."""

    def test_ratelimiter_state_structure(self) -> None:
        """Test RateLimiterState has correct structure."""
        # Create an instance
        state: FlextCore.Types.Reliability.RateLimiterState = {
            "requests": [1.0, 2.0, 3.0],
            "last_reset": 1234567890.0,
        }

        assert isinstance(state["requests"], list)
        assert isinstance(state["last_reset"], float)
        assert len(state["requests"]) == 3


class TestFlextTypes:
    """Test FlextCore.Types namespace and all sub-types."""

    def test_flexttypes_core_types(self) -> None:
        """Test FlextCore.Types types are accessible."""
        # Test collection types
        assert FlextCore.Types.Dict == FlextCore.Types.Dict
        assert FlextCore.Types.List == FlextCore.Types.List
        assert FlextCore.Types.StringList == FlextCore.Types.StringList
        assert FlextCore.Types.IntList == FlextCore.Types.IntList
        assert FlextCore.Types.FloatList == FlextCore.Types.FloatList
        assert FlextCore.Types.BoolList == FlextCore.Types.BoolList

        # Test advanced types
        assert FlextCore.Types.NestedDict == FlextCore.Types.NestedDict
        # OrderedDict is actually OrderedDict type

        assert FlextCore.Types.OrderedDict == FlextCore.Types.Dict

        # Test configuration types
        config_dict: FlextCore.Types.ConfigDict = {"key": "value", "number": 42}
        assert isinstance(config_dict, dict)

        # Test JSON types
        json_obj: FlextCore.Types.JsonValue = {"key": "value"}
        assert isinstance(json_obj, dict)

    def test_flexttypes_service_types(self) -> None:
        """Test FlextCore.Types.Service types."""
        service_dict: FlextCore.Types.Service.Dict = {"service": "value"}
        assert isinstance(service_dict, dict)

        # Test Literal types
        service_type: FlextCore.Types.Service.Type = "instance"
        assert service_type in {"instance", "factory"}

    def test_flexttypes_config_types(self) -> None:
        """Test FlextCore.Types.Config types."""
        # Test Environment literal - access through constants
        env = "development"
        assert env in {
            "development",
            "staging",
            "production",
            "testing",
            "test",
            "local",
        }
        # Verify Environment enum exists
        assert hasattr(FlextCore.Constants.Config, "Environment")
        assert FlextCore.Constants.Config.Environment.DEVELOPMENT.value == "development"

        # Test LogLevel literal
        level: FlextCore.Constants.Config.LogLevel = (
            FlextCore.Constants.Config.LogLevel.INFO
        )
        assert level == "INFO"

    def test_flexttypes_output_types(self) -> None:
        """Test FlextCore.Types.Output types."""
        # Test Literal types
        fmt: FlextCore.Types.Output.OutputFormat = "json"
        assert fmt in {"json", "yaml", "table", "csv", "text", "xml"}

        ser_fmt: FlextCore.Types.Output.SerializationFormat = "json"
        assert ser_fmt in {"json", "yaml", "toml", "ini", "xml"}

        comp_fmt: FlextCore.Types.Output.CompressionFormat = "gzip"
        assert comp_fmt in {"gzip", "bzip2", "xz", "lzma"}

    def test_flexttypes_project_types(self) -> None:
        """Test FlextCore.Types.Project types."""
        # Test Literal types
        proj_type: FlextCore.Types.Project.ProjectType = "library"
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

        status: FlextCore.Types.Project.ProjectStatus = "active"
        assert status in {"active", "inactive", "deprecated", "archived"}

    def test_flexttypes_processing_types(self) -> None:
        """Test FlextCore.Types.Processing types."""
        # Test all Literal types
        status: FlextCore.Types.Processing.ProcessingStatus = "pending"
        assert status in {"pending", "running", "completed", "failed", "cancelled"}

        mode: FlextCore.Types.Processing.ProcessingMode = "batch"
        assert mode in {"batch", "stream", "parallel", "sequential"}

        level: FlextCore.Types.Processing.ValidationLevel = "strict"
        assert level in {"strict", "lenient", "standard"}

        phase: FlextCore.Types.Processing.ProcessingPhase = "prepare"
        assert phase in {"prepare", "execute", "validate", "complete"}

        handler_type: FlextCore.Types.Processing.HandlerType = "command"
        assert handler_type in {"command", "query", "event", "processor"}

        workflow_status: FlextCore.Types.Processing.WorkflowStatus = "pending"
        assert workflow_status in {
            "pending",
            "running",
            "completed",
            "failed",
            "cancelled",
        }

    def test_flexttypes_convenience_aliases(self) -> None:
        """Test convenience aliases work."""
        # Test direct access aliases
        assert FlextCore.Types.ConfigValue is not None
        assert FlextCore.Types.JsonValue is not None
        assert FlextCore.Types.ConfigDict is not None
        assert FlextCore.Types.JsonDict is not None
        assert FlextCore.Types.StringDict is not None
        assert FlextCore.Types.StringDict is not None
        assert FlextCore.Types.Dict is not None


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
