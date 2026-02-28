"""Tests for FlextInfraJsonService.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

from flext_infra import FlextInfraJsonService
from pydantic import BaseModel


class SampleModel(BaseModel):
    """Sample model for testing."""

    name: str
    value: int


class TestFlextInfraJsonService:
    """Test suite for FlextInfraJsonService."""

    def test_read_existing_file(self, tmp_path: Path) -> None:
        """Test reading an existing JSON file."""
        json_file = tmp_path / "test.json"
        json_file.write_text('{"key": "value", "number": 42}', encoding="utf-8")
        service = FlextInfraJsonService()

        result = service.read(json_file)

        assert result.is_success
        assert result.value["key"] == "value"
        assert result.value["number"] == 42

    def test_read_nonexistent_file(self, tmp_path: Path) -> None:
        """Test reading a nonexistent file returns empty mapping."""
        json_file = tmp_path / "missing.json"
        service = FlextInfraJsonService()

        result = service.read(json_file)

        assert result.is_success
        assert result.value == {}

    def test_read_invalid_json(self, tmp_path: Path) -> None:
        """Test reading invalid JSON returns failure."""
        json_file = tmp_path / "invalid.json"
        json_file.write_text("{invalid json}", encoding="utf-8")
        service = FlextInfraJsonService()

        result = service.read(json_file)

        assert result.is_failure
        assert "JSON read error" in result.error

    def test_read_non_object_root(self, tmp_path: Path) -> None:
        """Test reading JSON with non-object root returns failure."""
        json_file = tmp_path / "array.json"
        json_file.write_text("[1, 2, 3]", encoding="utf-8")
        service = FlextInfraJsonService()

        result = service.read(json_file)

        assert result.is_failure
        assert "must be object" in result.error

    def test_write_dict_payload(self, tmp_path: Path) -> None:
        """Test writing a dict payload to JSON file."""
        json_file = tmp_path / "output.json"
        service = FlextInfraJsonService()
        payload = {"key": "value", "number": 42}

        result = service.write(json_file, payload)

        assert result.is_success
        assert json_file.exists()
        content = json_file.read_text(encoding="utf-8")
        assert "key" in content
        assert "value" in content

    def test_write_model_payload(self, tmp_path: Path) -> None:
        """Test writing a Pydantic model to JSON file."""
        json_file = tmp_path / "model.json"
        service = FlextInfraJsonService()
        model = SampleModel(name="test", value=123)

        result = service.write(json_file, model)

        assert result.is_success
        assert json_file.exists()

    def test_write_creates_parent_directories(self, tmp_path: Path) -> None:
        """Test write creates parent directories."""
        json_file = tmp_path / "nested" / "deep" / "file.json"
        service = FlextInfraJsonService()
        payload = {"key": "value"}

        result = service.write(json_file, payload)

        assert result.is_success
        assert json_file.exists()

    def test_write_with_sorted_keys(self, tmp_path: Path) -> None:
        """Test write with sorted keys."""
        json_file = tmp_path / "sorted.json"
        service = FlextInfraJsonService()
        payload = {"z": 1, "a": 2, "m": 3}

        result = service.write(json_file, payload, sort_keys=True)

        assert result.is_success
        content = json_file.read_text(encoding="utf-8")
        # Check that 'a' appears before 'z' in the output
        assert content.index('"a"') < content.index('"z"')

    def test_write_with_ensure_ascii(self, tmp_path: Path) -> None:
        """Test write with ensure_ascii flag."""
        json_file = tmp_path / "ascii.json"
        service = FlextInfraJsonService()
        payload = {"text": "café"}

        result = service.write(json_file, payload, ensure_ascii=True)

        assert result.is_success
        content = json_file.read_text(encoding="utf-8")
        assert "\\u" in content  # Unicode escape

    def test_write_permission_error(self, tmp_path: Path) -> None:
        """Test write failure on permission error."""
        json_file = tmp_path / "readonly.json"
        json_file.write_text("{}", encoding="utf-8")
        json_file.chmod(0o444)  # Read-only
        service = FlextInfraJsonService()

        try:
            result = service.write(json_file, {"key": "value"})
            assert result.is_failure
        finally:
            json_file.chmod(0o644)  # Restore permissions for cleanup

    def test_write_returns_true_on_success(self, tmp_path: Path) -> None:
        """Test write returns True on success."""
        json_file = tmp_path / "test.json"
        service = FlextInfraJsonService()

        result = service.write(json_file, {"key": "value"})
        assert result.is_success
        assert result.value is True
