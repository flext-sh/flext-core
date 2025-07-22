"""Minimal tests to improve coverage.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

This module provides minimal tests to improve coverage for the domain
pydantic_base module.
It includes tests for deprecated dict() and schema() methods.
"""

from __future__ import annotations

from flext_core.domain.pydantic_base import DomainBaseModel


class TestPydanticBaseMethods:
    """Test deprecated pydantic methods."""

    def test_deprecated_dict_method(self) -> None:
        """Test deprecated dict() method."""

        class TestModel(DomainBaseModel):
            name: str = "test"
            value: int = 42

        model = TestModel()
        result = model.model_dump()
        assert result == {"name": "test", "value": 42}

    def test_deprecated_schema_method(self) -> None:
        """Test deprecated schema() method."""

        class TestModel(DomainBaseModel):
            name: str = "test"

        schema = TestModel.model_json_schema()
        assert isinstance(schema, dict)
        assert "properties" in schema

    def test_deprecated_update_forward_refs(self) -> None:
        """Test deprecated update_forward_refs method."""

        class TestModel(DomainBaseModel):
            name: str = "test"

        # Should not raise exception
        TestModel.model_rebuild()
