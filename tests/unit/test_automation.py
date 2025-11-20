"""Tests for automation.py - coerce_model() function."""

import pytest
from pydantic import BaseModel, Field, ValidationError

from flext_core.automation import coerce_model


class MockConfig(BaseModel):
    """Test model for coerce_model() tests."""

    value: int = Field(ge=0)
    name: str = "default"


class TestCoerceModel:
    """Test coerce_model() quad API functionality."""

    def test_passthrough_model_instance(self) -> None:
        """Test Case 1: Model instance passthrough."""
        opts = MockConfig(value=5, name="test")
        result = coerce_model(MockConfig, opts)

        assert result is opts  # Same instance
        assert result.value == 5
        assert result.name == "test"

    def test_dict_to_model(self) -> None:
        """Test Case 2: Dict conversion to model."""
        result = coerce_model(MockConfig, {"value": 10, "name": "from_dict"})

        assert isinstance(result, MockConfig)
        assert result.value == 10
        assert result.name == "from_dict"

    def test_kwargs_only(self) -> None:
        """Test Case 3: Pure kwargs construction."""
        result = coerce_model(MockConfig, value=15, name="from_kwargs")

        assert isinstance(result, MockConfig)
        assert result.value == 15
        assert result.name == "from_kwargs"

    def test_hybrid_dict_plus_kwargs(self) -> None:
        """Test Case 4a: Hybrid dict + kwargs (kwargs override)."""
        result = coerce_model(MockConfig, {"value": 20}, name="override")

        assert result.value == 20
        assert result.name == "override"  # kwargs override dict

    def test_hybrid_model_plus_kwargs(self) -> None:
        """Test Case 4b: Hybrid model + kwargs (creates new instance)."""
        opts = MockConfig(value=25, name="original")
        result = coerce_model(MockConfig, opts, name="updated")

        assert result is not opts  # New instance created
        assert result.value == 25  # Value preserved
        assert result.name == "updated"  # Name updated

    def test_defaults_used_when_not_provided(self) -> None:
        """Test that model defaults are used."""
        result = coerce_model(MockConfig, value=30)

        assert result.value == 30
        assert result.name == "default"  # Default from model

    def test_validation_error_on_invalid_data(self) -> None:
        """Test that Pydantic validation errors are raised."""
        with pytest.raises(ValidationError):
            coerce_model(MockConfig, {"value": -1})  # value must be >= 0

    def test_type_error_on_invalid_options_type(self) -> None:
        """Test that TypeError is raised for invalid options type."""
        with pytest.raises(TypeError, match="options must be MockConfig, dict, or None"):
            coerce_model(MockConfig, "invalid")  # type: ignore[arg-type]

    def test_empty_dict_uses_defaults(self) -> None:
        """Test that empty dict uses model defaults."""
        result = coerce_model(MockConfig, {}, value=40)

        assert result.value == 40
        assert result.name == "default"
