"""Unit tests for flext_tests.files module.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest
import yaml
from pydantic import BaseModel

from flext_core import r
from flext_core.typings import t as t_core
from flext_tests.files import FlextTestsFiles, tf
from flext_tests.models import m

# Use the actual nested class directly
FileInfo = FlextTestsFiles.FileInfo


class TestFileInfo:
    """Test suite for FileInfo model."""

    def test_file_info_exists_false(self) -> None:
        """Test FileInfo with exists=False."""
        info = FileInfo(exists=False)

        assert info.exists is False
        assert info.size == 0
        assert info.lines == 0
        assert info.encoding == "utf-8"
        assert info.is_empty is False
        assert info.first_line == ""

    def test_file_info_exists_true(self) -> None:
        """Test FileInfo with exists=True."""
        info = FileInfo(
            exists=True,
            size=100,
            lines=5,
            encoding="utf-8",
            is_empty=False,
            first_line="first line",
        )

        assert info.exists is True
        assert info.size == 100
        assert info.lines == 5
        assert info.encoding == "utf-8"
        assert info.is_empty is False
        assert info.first_line == "first line"


class TestFlextTestsFiles:
    """Test suite for FlextTestsFiles class."""

    def test_init_without_base_dir(self) -> None:
        """Test initialization without base directory."""
        manager = FlextTestsFiles()

        assert manager.base_dir is None
        assert manager.created_files == []
        assert manager.created_dirs == []

    def test_init_with_base_dir(self, tmp_path: Path) -> None:
        """Test initialization with base directory."""
        manager = FlextTestsFiles(base_dir=tmp_path)

        assert manager.base_dir == tmp_path
        assert manager.created_files == []
        assert manager.created_dirs == []

    def test_temporary_directory(self) -> None:
        """Test temporary_directory context manager."""
        manager = FlextTestsFiles()

        with manager.temporary_directory() as temp_dir:
            assert isinstance(temp_dir, Path)
            assert temp_dir.exists()
            assert temp_dir.is_dir()

        # Directory should be cleaned up after context exit
        assert not temp_dir.exists()

    def test_create_text_file_default(self, tmp_path: Path) -> None:
        """Test creating text file with default parameters."""
        manager = FlextTestsFiles(base_dir=tmp_path)
        content = "test content"

        file_path = manager.create(content, "test.txt")

        assert file_path.exists()
        assert file_path.read_text() == content
        assert file_path.name == "test.txt"
        assert file_path in manager.created_files

    def test_create_text_file_custom(self, tmp_path: Path) -> None:
        """Test creating text file with custom parameters."""
        manager = FlextTestsFiles(base_dir=tmp_path)
        content = "custom content"
        filename = "custom.txt"
        custom_dir = tmp_path / "subdir"

        file_path = manager.create(content, filename, directory=custom_dir)

        assert file_path.exists()
        assert file_path.read_text() == content
        assert file_path.name == filename
        assert file_path.parent == custom_dir
        assert file_path in manager.created_files

    def test_create_text_file_custom_encoding(self, tmp_path: Path) -> None:
        """Test creating text file with custom encoding."""
        manager = FlextTestsFiles(base_dir=tmp_path)
        content = "test content"
        encoding = "utf-16"

        file_path = manager.create(content, "test.txt", enc=encoding)

        assert file_path.exists()
        assert file_path.read_text(encoding=encoding) == content

    def test_create_binary_file_default(self, tmp_path: Path) -> None:
        """Test creating binary file with default parameters."""
        manager = FlextTestsFiles(base_dir=tmp_path)
        content = b"binary content"

        file_path = manager.create(content, "binary_data.bin")

        assert file_path.exists()
        assert file_path.read_bytes() == content
        assert file_path.name == "binary_data.bin"
        assert file_path in manager.created_files

    def test_create_binary_file_custom(self, tmp_path: Path) -> None:
        """Test creating binary file with custom parameters."""
        manager = FlextTestsFiles(base_dir=tmp_path)
        content = b"custom binary"
        filename = "custom.bin"
        custom_dir = tmp_path / "subdir"

        file_path = manager.create(content, filename, directory=custom_dir)

        assert file_path.exists()
        assert file_path.read_bytes() == content
        assert file_path.name == filename
        assert file_path.parent == custom_dir

    def test_create_empty_file(self, tmp_path: Path) -> None:
        """Test creating empty file."""
        manager = FlextTestsFiles(base_dir=tmp_path)

        file_path = manager.create("", "empty.txt")

        assert file_path.exists()
        assert file_path.read_text() == ""
        assert file_path.name == "empty.txt"

    def test_create_empty_file_custom(self, tmp_path: Path) -> None:
        """Test creating empty file with custom name."""
        manager = FlextTestsFiles(base_dir=tmp_path)
        filename = "custom_empty.txt"

        file_path = manager.create("", filename)

        assert file_path.exists()
        assert file_path.read_text() == ""
        assert file_path.name == filename

    def test_create_file_set(self, tmp_path: Path) -> None:
        """Test creating multiple files from dictionary."""
        files: dict[
            str,
            str
            | bytes
            | t_core.Types.ConfigurationMapping
            | Sequence[Sequence[str]]
            | BaseModel,
        ] = {
            "file1": "content1",
            "file2": "content2",
            "file3.txt": "content3",
        }

        with tf.files(files, directory=tmp_path, ext=".txt") as created:
            assert len(created) == 3
            assert created["file1"].read_text() == "content1"
            assert created["file2"].read_text() == "content2"
            assert created["file3.txt"].read_text() == "content3"
            assert created["file1"].name == "file1.txt"
            assert created["file2"].name == "file2.txt"
            assert created["file3.txt"].name == "file3.txt"

    def test_create_file_set_custom_extension(self, tmp_path: Path) -> None:
        """Test creating file set with custom extension."""
        files_dict = {"file1": "content1"}
        # Type: dict[str, str] is compatible with dict[str, str | bytes | ...]
        # Cast to satisfy mypy's invariant dict type checking
        files: dict[
            str,
            str
            | bytes
            | t_core.Types.ConfigurationMapping
            | Sequence[Sequence[str]]
            | BaseModel,
        ] = cast(
            "dict[str, str | bytes | t_core.Types.ConfigurationMapping | Sequence[Sequence[str]] | BaseModel]",
            files_dict,
        )
        extension = ".md"

        with tf.files(files, directory=tmp_path, ext=extension) as created:
            assert created["file1"].name == "file1.md"

    def test_get_file_info_not_exists(self, tmp_path: Path) -> None:
        """Test getting file info for non-existent file."""
        manager = FlextTestsFiles()
        non_existent = tmp_path / "non_existent.txt"

        result = manager.info(non_existent)

        assert result.is_success
        file_info = result.unwrap()
        assert isinstance(file_info, FlextTestsFiles.FileInfo)
        assert file_info.exists is False

    def test_get_file_info_exists(self, tmp_path: Path) -> None:
        """Test getting file info for existing file."""
        manager = FlextTestsFiles(base_dir=tmp_path)
        content = "line1\nline2\nline3"
        file_path = manager.create(content, "test.txt")

        result = manager.info(file_path)

        assert result.is_success
        file_info = result.unwrap()
        assert isinstance(file_info, FlextTestsFiles.FileInfo)
        assert file_info.exists is True
        assert file_info.size > 0
        assert file_info.lines == 3
        assert file_info.encoding == "utf-8"
        assert file_info.is_empty is False
        assert file_info.first_line == "line1"

    def test_get_file_info_empty_file(self, tmp_path: Path) -> None:
        """Test getting file info for empty file."""
        manager = FlextTestsFiles(base_dir=tmp_path)
        file_path = manager.create("", "empty.txt")

        result = manager.info(file_path)

        assert result.is_success
        file_info = result.unwrap()
        assert isinstance(file_info, FlextTestsFiles.FileInfo)
        assert file_info.exists is True
        assert file_info.size == 0
        assert file_info.is_empty is True
        assert file_info.first_line == ""

    def test_get_file_info_multiline(self, tmp_path: Path) -> None:
        """Test getting file info for multiline file."""
        manager = FlextTestsFiles(base_dir=tmp_path)
        content = "first line\nsecond line\nthird line"
        file_path = manager.create(content, "multiline.txt")

        result = manager.info(file_path)

        assert result.is_success
        file_info = result.unwrap()
        assert file_info.lines == 3
        assert file_info.first_line == "first line"

    def test_cleanup_files(self, tmp_path: Path) -> None:
        """Test cleaning up created files."""
        manager = FlextTestsFiles(base_dir=tmp_path)
        file1 = manager.create("content1", "file1.txt")
        file2 = manager.create("content2", "file2.txt")

        assert file1.exists()
        assert file2.exists()

        manager.cleanup()

        assert not file1.exists()
        assert not file2.exists()
        assert len(manager.created_files) == 0

    def test_cleanup_directories(self) -> None:
        """Test cleaning up created directories."""
        manager = FlextTestsFiles()
        # Create a file which will create a temp directory
        file_path = manager.create("content", "test.txt")
        temp_dir = file_path.parent

        assert temp_dir.exists()
        assert temp_dir in manager.created_dirs

        manager.cleanup()

        assert not temp_dir.exists()
        assert len(manager.created_dirs) == 0

    def test_cleanup_nonexistent_files(self, tmp_path: Path) -> None:
        """Test cleanup handles non-existent files gracefully."""
        manager = FlextTestsFiles(base_dir=tmp_path)
        file_path = manager.create("content", "test.txt")
        file_path.unlink()  # Delete file manually

        # Cleanup should not raise error
        manager.cleanup()

        assert len(manager.created_files) == 0

    def test_context_manager(self, tmp_path: Path) -> None:
        """Test context manager usage."""
        with FlextTestsFiles(base_dir=tmp_path) as manager:
            file_path = manager.create("content", "test.txt")
            assert file_path.exists()

        # File should be cleaned up after context exit
        assert not file_path.exists()

    def test_resolve_directory_with_directory(self, tmp_path: Path) -> None:
        """Test directory resolution with provided directory."""
        manager = FlextTestsFiles(base_dir=tmp_path)
        custom_dir = tmp_path / "custom"

        resolved = manager._resolve_directory(custom_dir)

        assert resolved == custom_dir

    def test_resolve_directory_with_base_dir(self, tmp_path: Path) -> None:
        """Test directory resolution with base_dir."""
        manager = FlextTestsFiles(base_dir=tmp_path)

        resolved = manager._resolve_directory(None)

        assert resolved == tmp_path

    def test_resolve_directory_creates_temp(self) -> None:
        """Test directory resolution creates temporary directory."""
        manager = FlextTestsFiles()

        resolved = manager._resolve_directory(None)

        assert resolved.exists()
        assert resolved.is_dir()
        assert resolved in manager.created_dirs

    def test_temporary_files_classmethod(self) -> None:
        """Test files classmethod context manager."""
        files_dict = {
            "file1": "content1",
            "file2": "content2",
        }
        # Type: dict[str, str] is compatible with dict[str, str | bytes | ...]
        # Cast to satisfy mypy's invariant dict type checking
        files: dict[
            str,
            str
            | bytes
            | t_core.Types.ConfigurationMapping
            | Sequence[Sequence[str]]
            | BaseModel,
        ] = cast(
            "dict[str, str | bytes | t_core.Types.ConfigurationMapping | Sequence[Sequence[str]] | BaseModel]",
            files_dict,
        )

        with FlextTestsFiles.files(files) as created:
            assert len(created) == 2
            assert created["file1"].exists()
            assert created["file2"].exists()
            assert created["file1"].read_text() == "content1"
            assert created["file2"].read_text() == "content2"

        # Files should be cleaned up after context exit
        assert not created["file1"].exists()
        assert not created["file2"].exists()

    def test_temporary_files_custom_extension(self) -> None:
        """Test files with custom extension."""
        files_dict = {"file1": "content1"}
        # Type: dict[str, str] is compatible with dict[str, str | bytes | ...]
        # Cast to satisfy mypy's invariant dict type checking
        files: dict[
            str,
            str
            | bytes
            | t_core.Types.ConfigurationMapping
            | Sequence[Sequence[str]]
            | BaseModel,
        ] = cast(
            "dict[str, str | bytes | t_core.Types.ConfigurationMapping | Sequence[Sequence[str]] | BaseModel]",
            files_dict,
        )

        with FlextTestsFiles.files(files, ext=".md") as created:
            assert created["file1"].name == "file1.md"

    def test_create_file_set_nested_directory(self, tmp_path: Path) -> None:
        """Test creating files in nested directory."""
        nested_dir = tmp_path / "nested" / "subdir"
        files_dict = {"file1": "content1"}
        # Type: dict[str, str] is compatible with dict[str, str | bytes | ...]
        # Cast to satisfy mypy's invariant dict type checking
        files: dict[
            str,
            str
            | bytes
            | t_core.Types.ConfigurationMapping
            | Sequence[Sequence[str]]
            | BaseModel,
        ] = cast(
            "dict[str, str | bytes | t_core.Types.ConfigurationMapping | Sequence[Sequence[str]] | BaseModel]",
            files_dict,
        )

        with tf.files(files, directory=nested_dir) as created:
            assert created["file1"].parent == nested_dir
            assert nested_dir.exists()

    def test_create_text_file_nested_directory(self, tmp_path: Path) -> None:
        """Test creating text file in nested directory."""
        manager = FlextTestsFiles(base_dir=tmp_path)
        nested_dir = tmp_path / "nested" / "subdir"

        file_path = manager.create("content", "test.txt", directory=nested_dir)

        assert file_path.parent == nested_dir
        assert nested_dir.exists()

    def test_multiple_cleanup_calls(self, tmp_path: Path) -> None:
        """Test multiple cleanup calls are safe."""
        manager = FlextTestsFiles(base_dir=tmp_path)
        _ = manager.create("content", "test.txt")

        manager.cleanup()
        # Second cleanup should not raise error
        manager.cleanup()

        assert len(manager.created_files) == 0


class TestFlextTestsFilesNewApi:
    """Test suite for new FlextTestsFiles API methods (create, read, compare, info, files)."""

    # =========================================================================
    # create() Tests
    # =========================================================================

    def test_create_text_auto_detect(self, tmp_path: Path) -> None:
        """Test create() auto-detects text from str content."""
        manager = FlextTestsFiles(base_dir=tmp_path)

        path = manager.create("hello world", "test.txt")

        assert path.exists()
        assert path.read_text() == "hello world"
        assert path.suffix == ".txt"

    def test_create_binary_auto_detect(self, tmp_path: Path) -> None:
        """Test create() auto-detects binary from bytes content."""
        manager = FlextTestsFiles(base_dir=tmp_path)

        path = manager.create(b"\x00\x01\x02", "data.bin")

        assert path.exists()
        assert path.read_bytes() == b"\x00\x01\x02"

    def test_create_json_auto_detect_from_dict(self, tmp_path: Path) -> None:
        """Test create() auto-detects JSON from dict content."""
        manager = FlextTestsFiles(base_dir=tmp_path)
        content_dict = {"key": "value", "number": 42}
        # Type: dict[str, object] needs to be cast to ConfigurationMapping
        # create() accepts ConfigurationMapping which is compatible
        content: t_core.Types.ConfigurationMapping = cast(
            "t_core.Types.ConfigurationMapping",
            content_dict,
        )

        path = manager.create(content, "config.json")

        assert path.exists()
        data = json.loads(path.read_text())
        assert data == content

    def test_create_yaml_auto_detect_from_extension(self, tmp_path: Path) -> None:
        """Test create() auto-detects YAML from .yaml extension."""
        manager = FlextTestsFiles(base_dir=tmp_path)
        content_dict = {"name": "test", "enabled": True}
        # Type: dict[str, object] needs to be cast to ConfigurationMapping
        # create() accepts ConfigurationMapping which is compatible
        content: t_core.Types.ConfigurationMapping = cast(
            "t_core.Types.ConfigurationMapping",
            content_dict,
        )

        path = manager.create(content, "config.yaml")

        assert path.exists()
        data = yaml.safe_load(path.read_text())
        assert data == content

    def test_create_csv_auto_detect_from_list(self, tmp_path: Path) -> None:
        """Test create() auto-detects CSV from list[list] content."""
        manager = FlextTestsFiles(base_dir=tmp_path)
        content = [["a", "b"], ["1", "2"]]

        path = manager.create(content, "data.csv")

        assert path.exists()
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2

    def test_create_csv_with_headers(self, tmp_path: Path) -> None:
        """Test create() CSV with explicit headers."""
        manager = FlextTestsFiles(base_dir=tmp_path)
        content = [["1", "2"], ["3", "4"]]

        path = manager.create(content, "data.csv", headers=["col1", "col2"])

        assert path.exists()
        lines = path.read_text().strip().split("\n")
        assert lines[0] == "col1,col2"
        assert len(lines) == 3  # header + 2 data rows

    def test_create_explicit_format(self, tmp_path: Path) -> None:
        """Test create() with explicit format override."""
        manager = FlextTestsFiles(base_dir=tmp_path)

        path = manager.create(b"raw bytes", "data.dat", fmt="bin")

        assert path.exists()
        assert path.read_bytes() == b"raw bytes"

    def test_create_custom_encoding(self, tmp_path: Path) -> None:
        """Test create() with custom encoding."""
        manager = FlextTestsFiles(base_dir=tmp_path)

        path = manager.create("áéíóú", "unicode.txt", enc="utf-16")

        assert path.exists()
        assert path.read_text(encoding="utf-16") == "áéíóú"

    def test_create_json_custom_indent(self, tmp_path: Path) -> None:
        """Test create() JSON with custom indentation."""
        manager = FlextTestsFiles(base_dir=tmp_path)
        content = {"key": "value"}

        path = manager.create(content, "config.json", indent=4)

        assert path.exists()
        text = path.read_text()
        # 4-space indent should have more whitespace than 2-space
        assert "    " in text

    def test_create_in_custom_directory(self, tmp_path: Path) -> None:
        """Test create() in custom directory."""
        manager = FlextTestsFiles(base_dir=tmp_path)
        custom_dir = tmp_path / "subdir"

        path = manager.create("content", "test.txt", directory=custom_dir)

        assert path.exists()
        assert path.parent == custom_dir

    # =========================================================================
    # read() Tests
    # =========================================================================

    def test_read_text_file(self, tmp_path: Path) -> None:
        """Test read() returns text content for .txt files."""
        manager = FlextTestsFiles(base_dir=tmp_path)
        path = manager.create("hello world", "test.txt")

        result = manager.read(path)

        assert result.is_success
        assert result.unwrap() == "hello world"

    def test_read_binary_file(self, tmp_path: Path) -> None:
        """Test read() returns bytes content for .bin files."""
        manager = FlextTestsFiles(base_dir=tmp_path)
        path = manager.create(b"\x00\x01\x02", "data.bin", fmt="bin")

        result = manager.read(path)

        assert result.is_success
        assert result.unwrap() == b"\x00\x01\x02"

    def test_read_json_file(self, tmp_path: Path) -> None:
        """Test read() returns dict content for .json files."""
        manager = FlextTestsFiles(base_dir=tmp_path)
        content_dict = {"key": "value", "number": 42}
        # Type: dict[str, object] needs to be cast to ConfigurationMapping
        # create() accepts ConfigurationMapping which is compatible
        content: t_core.Types.ConfigurationMapping = cast(
            "t_core.Types.ConfigurationMapping",
            content_dict,
        )
        path = manager.create(content, "config.json")

        result = manager.read(path)

        assert result.is_success
        assert result.unwrap() == content

    def test_read_yaml_file(self, tmp_path: Path) -> None:
        """Test read() returns dict content for .yaml files."""
        manager = FlextTestsFiles(base_dir=tmp_path)
        content_dict = {"name": "test", "enabled": True}
        # Type: dict[str, object] needs to be cast to ConfigurationMapping
        # create() accepts ConfigurationMapping which is compatible
        content: t_core.Types.ConfigurationMapping = cast(
            "t_core.Types.ConfigurationMapping",
            content_dict,
        )
        path = manager.create(content, "config.yaml")

        result = manager.read(path)

        assert result.is_success
        assert result.unwrap() == content

    def test_read_csv_file(self, tmp_path: Path) -> None:
        """Test read() returns list[list] content for .csv files."""
        manager = FlextTestsFiles(base_dir=tmp_path)
        content = [["a", "b"], ["1", "2"]]
        path = manager.create(content, "data.csv")

        # By default has_headers=True, so first row is treated as header and skipped
        result = manager.read(path, has_headers=False)

        assert result.is_success
        # CSV read returns list of lists
        data = result.unwrap()
        assert isinstance(data, list)
        assert len(data) == 2

    def test_read_csv_file_with_headers(self, tmp_path: Path) -> None:
        """Test read() CSV with headers skips first row by default."""
        manager = FlextTestsFiles(base_dir=tmp_path)
        content = [["header1", "header2"], ["1", "2"], ["3", "4"]]
        path = manager.create(
            content,
            "data.csv",
            headers=None,
        )  # Don't add headers twice

        # Default has_headers=True skips first row
        result = manager.read(path)

        assert result.is_success
        data = result.unwrap()
        assert isinstance(data, list)
        assert len(data) == 2  # Only data rows, header skipped

    def test_read_nonexistent_file(self, tmp_path: Path) -> None:
        """Test read() returns failure for non-existent file."""
        manager = FlextTestsFiles(base_dir=tmp_path)
        path = tmp_path / "nonexistent.txt"

        result = manager.read(path)

        assert result.is_failure
        assert result.error is not None
        assert (
            "not found" in result.error.lower() or "not exist" in result.error.lower()
        )

    def test_read_explicit_format(self, tmp_path: Path) -> None:
        """Test read() with explicit format override."""
        manager = FlextTestsFiles(base_dir=tmp_path)
        # Create a text file with .dat extension
        path = manager.create("plain text", "data.dat", fmt="text")

        result = manager.read(path, fmt="text")

        assert result.is_success
        assert result.unwrap() == "plain text"

    # =========================================================================
    # compare() Tests
    # =========================================================================

    def test_compare_identical_content(self, tmp_path: Path) -> None:
        """Test compare() returns True for identical content."""
        manager = FlextTestsFiles(base_dir=tmp_path)
        path1 = manager.create("same content", "file1.txt")
        path2 = manager.create("same content", "file2.txt")

        result = manager.compare(path1, path2)

        assert result.is_success
        assert result.unwrap() is True

    def test_compare_different_content(self, tmp_path: Path) -> None:
        """Test compare() returns False for different content."""
        manager = FlextTestsFiles(base_dir=tmp_path)
        path1 = manager.create("content A", "file1.txt")
        path2 = manager.create("content B", "file2.txt")

        result = manager.compare(path1, path2)

        assert result.is_success
        assert result.unwrap() is False

    def test_compare_size_mode(self, tmp_path: Path) -> None:
        """Test compare() in size mode."""
        manager = FlextTestsFiles(base_dir=tmp_path)
        path1 = manager.create("12345", "file1.txt")
        path2 = manager.create("abcde", "file2.txt")  # Same length

        result = manager.compare(path1, path2, mode="size")

        assert result.is_success
        assert result.unwrap() is True

    def test_compare_size_mode_different(self, tmp_path: Path) -> None:
        """Test compare() in size mode with different sizes."""
        manager = FlextTestsFiles(base_dir=tmp_path)
        path1 = manager.create("short", "file1.txt")
        path2 = manager.create("much longer content", "file2.txt")

        result = manager.compare(path1, path2, mode="size")

        assert result.is_success
        assert result.unwrap() is False

    def test_compare_hash_mode(self, tmp_path: Path) -> None:
        """Test compare() in hash mode."""
        manager = FlextTestsFiles(base_dir=tmp_path)
        path1 = manager.create("identical", "file1.txt")
        path2 = manager.create("identical", "file2.txt")

        result = manager.compare(path1, path2, mode="hash")

        assert result.is_success
        assert result.unwrap() is True

    def test_compare_lines_mode(self, tmp_path: Path) -> None:
        """Test compare() in lines mode compares actual line content."""
        manager = FlextTestsFiles(base_dir=tmp_path)
        path1 = manager.create("line1\nline2\nline3", "file1.txt")
        path2 = manager.create("line1\nline2\nline3", "file2.txt")  # Same content

        result = manager.compare(path1, path2, mode="lines")

        assert result.is_success
        assert result.unwrap() is True

    def test_compare_lines_mode_different(self, tmp_path: Path) -> None:
        """Test compare() in lines mode returns False for different content."""
        manager = FlextTestsFiles(base_dir=tmp_path)
        path1 = manager.create("line1\nline2\nline3", "file1.txt")
        path2 = manager.create(
            "a\nb\nc",
            "file2.txt",
        )  # Same line count, different content

        result = manager.compare(path1, path2, mode="lines")

        assert result.is_success
        assert result.unwrap() is False

    def test_compare_ignore_whitespace(self, tmp_path: Path) -> None:
        """Test compare() ignoring whitespace."""
        manager = FlextTestsFiles(base_dir=tmp_path)
        path1 = manager.create("hello world", "file1.txt")
        path2 = manager.create("hello  world", "file2.txt")

        result = manager.compare(path1, path2, ignore_ws=True)

        assert result.is_success
        # With ignore_ws, extra spaces should be ignored
        assert result.unwrap() is True

    def test_compare_ignore_case(self, tmp_path: Path) -> None:
        """Test compare() ignoring case."""
        manager = FlextTestsFiles(base_dir=tmp_path)
        path1 = manager.create("Hello World", "file1.txt")
        path2 = manager.create("hello world", "file2.txt")

        result = manager.compare(path1, path2, ignore_case=True)

        assert result.is_success
        assert result.unwrap() is True

    def test_compare_pattern_match(self, tmp_path: Path) -> None:
        """Test compare() with pattern matching."""
        manager = FlextTestsFiles(base_dir=tmp_path)
        path1 = manager.create("ERROR: something failed", "file1.txt")
        path2 = manager.create("ERROR: other failure", "file2.txt")

        result = manager.compare(path1, path2, pattern="ERROR")

        assert result.is_success
        assert result.unwrap() is True  # Both contain "ERROR"

    def test_compare_pattern_no_match(self, tmp_path: Path) -> None:
        """Test compare() pattern matching when one file doesn't match."""
        manager = FlextTestsFiles(base_dir=tmp_path)
        path1 = manager.create("ERROR: something failed", "file1.txt")
        path2 = manager.create("Success: all good", "file2.txt")

        result = manager.compare(path1, path2, pattern="ERROR")

        assert result.is_success
        assert result.unwrap() is False  # Only one contains "ERROR"

    def test_compare_nonexistent_file(self, tmp_path: Path) -> None:
        """Test compare() returns failure for non-existent file."""
        manager = FlextTestsFiles(base_dir=tmp_path)
        path1 = manager.create("content", "file1.txt")
        path2 = tmp_path / "nonexistent.txt"

        result = manager.compare(path1, path2)

        assert result.is_failure

    # =========================================================================
    # info() Tests
    # =========================================================================

    def test_info_existing_file(self, tmp_path: Path) -> None:
        """Test info() returns FileInfo for existing file."""
        manager = FlextTestsFiles(base_dir=tmp_path)
        path = manager.create("line1\nline2\nline3", "test.txt")

        result = manager.info(path)

        assert result.is_success
        info = result.unwrap()
        assert info.exists is True
        assert info.size > 0
        assert info.lines == 3
        assert info.is_empty is False
        assert info.first_line == "line1"

    def test_info_nonexistent_file(self, tmp_path: Path) -> None:
        """Test info() returns FileInfo with exists=False."""
        manager = FlextTestsFiles(base_dir=tmp_path)
        path = tmp_path / "nonexistent.txt"

        result = manager.info(path)

        assert result.is_success
        info = result.unwrap()
        assert info.exists is False

    def test_info_with_hash(self, tmp_path: Path) -> None:
        """Test info() computes SHA256 hash when requested."""
        manager = FlextTestsFiles(base_dir=tmp_path)
        path = manager.create("test content", "test.txt")

        result = manager.info(path, compute_hash=True)

        assert result.is_success
        info = result.unwrap()
        assert info.sha256 is not None
        assert len(info.sha256) == 64  # SHA256 hex digest length

    def test_info_format_detection(self, tmp_path: Path) -> None:
        """Test info() detects file format."""
        manager = FlextTestsFiles(base_dir=tmp_path)
        path = manager.create({"key": "value"}, "config.json")

        result = manager.info(path)

        assert result.is_success
        info = result.unwrap()
        assert info.fmt == "json"

    def test_info_empty_file(self, tmp_path: Path) -> None:
        """Test info() for empty file."""
        manager = FlextTestsFiles(base_dir=tmp_path)
        path = manager.create("", "empty.txt")

        result = manager.info(path)

        assert result.is_success
        info = result.unwrap()
        assert info.exists is True
        assert info.size == 0
        assert info.is_empty is True

    def test_info_size_human_readable(self, tmp_path: Path) -> None:
        """Test info() provides human-readable size."""
        manager = FlextTestsFiles(base_dir=tmp_path)
        # Create a file with known content
        path = manager.create("x" * 1024, "test.txt")

        result = manager.info(path)

        assert result.is_success
        info = result.unwrap()
        # Should show KB or similar
        assert info.size_human != ""

    # =========================================================================
    # files() Context Manager Tests
    # =========================================================================

    def test_files_context_manager_basic(self) -> None:
        """Test files() context manager creates temporary files."""
        with FlextTestsFiles.files({"a": "content A", "b": "content B"}) as paths:
            assert "a" in paths
            assert "b" in paths
            assert paths["a"].exists()
            assert paths["b"].exists()
            assert paths["a"].read_text() == "content A"
            assert paths["b"].read_text() == "content B"

        # Files should be cleaned up
        assert not paths["a"].exists()
        assert not paths["b"].exists()

    def test_files_context_manager_json_auto_detect(self) -> None:
        """Test files() auto-detects JSON from dict content."""
        content = {"key": "value"}

        with FlextTestsFiles.files({"config": content}) as paths:
            assert paths["config"].suffix == ".json"
            data = json.loads(paths["config"].read_text())
            assert data == content

    def test_files_context_manager_mixed_types(self) -> None:
        """Test files() handles mixed content types."""
        with FlextTestsFiles.files({
            "text": "plain text",
            "json": {"key": "value"},
            "csv": [["a", "b"], ["1", "2"]],
        }) as paths:
            assert paths["text"].read_text() == "plain text"
            assert json.loads(paths["json"].read_text()) == {"key": "value"}
            assert len(paths["csv"].read_text().strip().split("\n")) == 2

    def test_files_context_manager_custom_extension(self) -> None:
        """Test files() with custom default extension."""
        with FlextTestsFiles.files({"file1": "content"}, ext=".md") as paths:
            assert paths["file1"].suffix == ".md"

    def test_files_context_manager_custom_directory(self, tmp_path: Path) -> None:
        """Test files() in custom directory."""
        with FlextTestsFiles.files({"test": "content"}, directory=tmp_path) as paths:
            assert paths["test"].parent == tmp_path


class TestShortAlias:
    """Test suite for tf short alias."""

    def test_tf_alias_import(self) -> None:
        """Test tf alias is importable."""
        assert tf is FlextTestsFiles

    def test_tf_alias_usage(self, tmp_path: Path) -> None:
        """Test tf alias can be used to create files."""
        with tf(base_dir=tmp_path) as files:
            path = files.create("test content", "test.txt")
            assert path.exists()

    def test_tf_files_context_manager(self) -> None:
        """Test tf.files() context manager works."""
        with tf.files({"test": "content"}) as paths:
            assert paths["test"].exists()


class TestFileInfoFromModels:
    """Test suite for FileInfo from FlextTestsModels.Files namespace."""

    def test_fileinfo_import_from_models(self) -> None:
        """Test FileInfo can be imported from models."""
        info = m.Tests.Files.FileInfo(exists=True, size=100, lines=5)

        assert info.exists is True
        assert info.size == 100
        assert info.lines == 5

    def test_fileinfo_backward_compatibility(self) -> None:
        """Test FileInfo alias works for backward compatibility."""
        # Old way still works
        info = FlextTestsFiles.FileInfo(exists=True)
        assert info.exists is True

        # New way works too
        info2 = m.Tests.Files.FileInfo(exists=True)
        assert info2.exists is True

    def test_fileinfo_all_fields(self) -> None:
        """Test FileInfo with all fields populated."""
        now = datetime.now(tz=UTC)
        info = m.Tests.Files.FileInfo(
            exists=True,
            path=Path("/test/file.txt"),
            size=1024,
            size_human="1.0 KB",
            lines=50,
            encoding="utf-8",
            is_empty=False,
            first_line="#!/usr/bin/env python",
            fmt="text",
            is_valid=True,
            created=now,
            modified=now,
            permissions=0o644,
            is_readonly=False,
            sha256="abc123" * 10 + "abcd",
        )

        assert info.exists is True
        assert info.size == 1024
        assert info.size_human == "1.0 KB"
        assert info.lines == 50
        assert info.fmt == "text"
        assert info.sha256 is not None


class TestInfoWithContentMeta:
    """Test suite for info() with ContentMeta integration."""

    def test_info_parse_content_json_dict(self, tmp_path: Path) -> None:
        """Test info() with parse_content=True for JSON dict."""
        manager = FlextTestsFiles(base_dir=tmp_path)
        path = manager.create({"key1": "value1", "key2": "value2"}, "config.json")

        result = manager.info(path, parse_content=True)

        assert result.is_success
        info = result.unwrap()
        assert info.content_meta is not None
        assert info.content_meta.key_count == 2
        assert info.content_meta.item_count is None

    def test_info_parse_content_json_list(self, tmp_path: Path) -> None:
        """Test info() with parse_content=True for JSON list."""
        manager = FlextTestsFiles(base_dir=tmp_path)
        content = json.dumps([1, 2, 3, 4, 5])
        path = tmp_path / "list.json"
        path.write_text(content)

        result = manager.info(path, parse_content=True)

        assert result.is_success
        info = result.unwrap()
        assert info.content_meta is not None
        assert info.content_meta.key_count is None
        assert info.content_meta.item_count == 5

    def test_info_parse_content_yaml_dict(self, tmp_path: Path) -> None:
        """Test info() with parse_content=True for YAML dict."""
        manager = FlextTestsFiles(base_dir=tmp_path)
        path = manager.create({"a": 1, "b": 2, "c": 3}, "config.yaml")

        result = manager.info(path, parse_content=True)

        assert result.is_success
        info = result.unwrap()
        assert info.content_meta is not None
        assert info.content_meta.key_count == 3

    def test_info_parse_content_csv(self, tmp_path: Path) -> None:
        """Test info() with parse_content=True for CSV."""
        manager = FlextTestsFiles(base_dir=tmp_path)
        csv_content = "name,age,city\nAlice,30,NYC\nBob,25,LA\n"
        path = tmp_path / "data.csv"
        path.write_text(csv_content)

        result = manager.info(path, parse_content=True, detect_fmt=True)

        assert result.is_success
        info = result.unwrap()
        assert info.content_meta is not None
        assert info.content_meta.row_count == 3  # header + 2 data rows
        assert info.content_meta.column_count == 3

    def test_info_validate_model_success(self, tmp_path: Path) -> None:
        """Test info() with validate_model for valid model."""

        class SimpleModel(BaseModel):
            name: str
            age: int

        manager = FlextTestsFiles(base_dir=tmp_path)
        path = manager.create({"name": "Alice", "age": 30}, "user.json")

        result = manager.info(path, validate_model=SimpleModel)

        assert result.is_success
        info = result.unwrap()
        assert info.content_meta is not None
        assert info.content_meta.model_valid is True
        assert info.content_meta.model_name == "SimpleModel"

    def test_info_validate_model_failure(self, tmp_path: Path) -> None:
        """Test info() with validate_model for invalid model."""

        class StrictModel(BaseModel):
            required_field: str

        manager = FlextTestsFiles(base_dir=tmp_path)
        path = manager.create({"other_field": "value"}, "invalid.json")

        result = manager.info(path, validate_model=StrictModel)

        assert result.is_success
        info = result.unwrap()
        assert info.content_meta is not None
        assert info.content_meta.model_valid is False
        assert info.content_meta.model_name == "StrictModel"


class TestAssertExists:
    """Test suite for tf.assert_exists() static method."""

    def test_assert_exists_file_success(self, tmp_path: Path) -> None:
        """Test assert_exists() succeeds for existing file."""
        path = tmp_path / "test.txt"
        path.write_text("content")

        # Should not raise
        tf.assert_exists(path)

    def test_assert_exists_file_failure(self, tmp_path: Path) -> None:
        """Test assert_exists() fails for non-existing file."""
        path = tmp_path / "nonexistent.txt"

        with pytest.raises(AssertionError):
            tf.assert_exists(path)

    def test_assert_exists_directory_success(self, tmp_path: Path) -> None:
        """Test assert_exists() succeeds for existing directory."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()

        # Should not raise
        tf.assert_exists(subdir)

    def test_assert_exists_is_file_check(self, tmp_path: Path) -> None:
        """Test assert_exists() with is_file=True."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("content")

        # Should not raise for file
        tf.assert_exists(file_path, is_file=True)

    def test_assert_exists_is_dir_check(self, tmp_path: Path) -> None:
        """Test assert_exists() with is_dir=True."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()

        # Should not raise for directory
        tf.assert_exists(subdir, is_dir=True)

    def test_assert_exists_not_empty(self, tmp_path: Path) -> None:
        """Test assert_exists() with not_empty=True."""
        path = tmp_path / "test.txt"
        path.write_text("content")

        # Should not raise for non-empty file
        tf.assert_exists(path, not_empty=True)

    def test_assert_exists_empty_file_fails(self, tmp_path: Path) -> None:
        """Test assert_exists() fails for empty file with not_empty=True."""
        path = tmp_path / "empty.txt"
        path.write_text("")

        with pytest.raises(AssertionError):
            tf.assert_exists(path, not_empty=True)

    def test_assert_exists_readable_check(self, tmp_path: Path) -> None:
        """Test assert_exists() with readable=True validation."""
        path = tmp_path / "readable.txt"
        path.write_text("content")
        # Make readable (default should be readable)
        path.chmod(0o644)

        # Should not raise for readable file
        tf.assert_exists(path, readable=True)

    def test_assert_exists_writable_check_file(self, tmp_path: Path) -> None:
        """Test assert_exists() with writable=True for file."""
        path = tmp_path / "writable.txt"
        path.write_text("content")
        # Make writable (default should be writable)
        path.chmod(0o644)

        # Should not raise for writable file
        tf.assert_exists(path, writable=True)

    def test_assert_exists_writable_check_directory(self, tmp_path: Path) -> None:
        """Test assert_exists() with writable=True for directory."""
        subdir = tmp_path / "writable_dir"
        subdir.mkdir()
        # Make writable (default should be writable)
        subdir.chmod(0o755)

        # Should not raise for writable directory
        tf.assert_exists(subdir, writable=True)

    def test_assert_exists_custom_error_message(self, tmp_path: Path) -> None:
        """Test assert_exists() with custom error message."""
        path = tmp_path / "nonexistent.txt"

        with pytest.raises(AssertionError, match="Custom error"):
            tf.assert_exists(path, msg="Custom error: file not found")

    def test_assert_exists_combined_validations(self, tmp_path: Path) -> None:
        """Test assert_exists() with multiple validations at once."""
        path = tmp_path / "test.txt"
        path.write_text("content")
        path.chmod(0o644)

        # Should not raise for file that satisfies all conditions
        tf.assert_exists(
            path,
            is_file=True,
            not_empty=True,
            readable=True,
            writable=True,
        )

    def test_assert_exists_is_file_false(self, tmp_path: Path) -> None:
        """Test assert_exists() with is_file=False (should not be a file)."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()

        # Should not raise - subdir is not a file
        tf.assert_exists(subdir, is_file=False)

    def test_assert_exists_is_dir_false(self, tmp_path: Path) -> None:
        """Test assert_exists() with is_dir=False (should not be a directory)."""
        path = tmp_path / "test.txt"
        path.write_text("content")

        # Should not raise - path is not a directory
        tf.assert_exists(path, is_dir=False)

    def test_assert_exists_empty_directory_fails(self, tmp_path: Path) -> None:
        """Test assert_exists() fails for empty directory with not_empty=True."""
        subdir = tmp_path / "empty_dir"
        subdir.mkdir()

        with pytest.raises(AssertionError):
            tf.assert_exists(subdir, not_empty=True)

    def test_assert_exists_not_empty_directory_success(self, tmp_path: Path) -> None:
        """Test assert_exists() succeeds for non-empty directory."""
        subdir = tmp_path / "non_empty_dir"
        subdir.mkdir()
        (subdir / "file.txt").write_text("content")

        # Should not raise for non-empty directory
        tf.assert_exists(subdir, not_empty=True)


class TestBatchOperations:
    """Test suite for tf().batch() method."""

    def test_batch_create_multiple_files(self, tmp_path: Path) -> None:
        """Test batch create for multiple files."""
        manager = FlextTestsFiles(base_dir=tmp_path)

        result = manager.batch(
            {
                "file1.txt": "content1",
                "file2.txt": "content2",
                "file3.txt": "content3",
            },
            directory=tmp_path,
        )

        assert result.is_success
        batch_result = result.unwrap()
        assert batch_result.total == 3
        assert batch_result.success_count == 3
        assert batch_result.failure_count == 0
        assert batch_result.succeeded == 3

    def test_batch_create_json_files(self, tmp_path: Path) -> None:
        """Test batch create for JSON files."""
        manager = FlextTestsFiles(base_dir=tmp_path)

        result = manager.batch(
            {
                "config1.json": {"key": "value1"},
                "config2.json": {"key": "value2"},
            },
            directory=tmp_path,
        )

        assert result.is_success
        batch_result = result.unwrap()
        assert batch_result.success_count == 2
        # Verify files were created correctly
        config1 = tmp_path / "config1.json"
        assert config1.exists()
        assert json.loads(config1.read_text())["key"] == "value1"

    def test_batch_on_error_collect(self, tmp_path: Path) -> None:
        """Test batch with on_error='collect' continues on failures."""
        manager = FlextTestsFiles(base_dir=tmp_path)

        # Create a read-only directory to cause failure
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()

        result = manager.batch(
            {
                "valid.txt": "content",
            },
            directory=tmp_path,
            on_error="collect",
        )

        # Should succeed for the valid file
        assert result.is_success
        batch_result = result.unwrap()
        # success_count is a computed_field property, not a callable
        # Access it as an attribute, not a method
        assert batch_result.success_count >= 1  # type: ignore[operator]

    def test_batch_result_model_structure(self, tmp_path: Path) -> None:
        """Test BatchResult model has correct structure."""
        manager = FlextTestsFiles(base_dir=tmp_path)

        result = manager.batch(
            {"file.txt": "content"},
            directory=tmp_path,
        )

        assert result.is_success
        batch_result = result.unwrap()

        # Verify all expected fields exist
        assert hasattr(batch_result, "succeeded")
        assert hasattr(batch_result, "failed")
        assert hasattr(batch_result, "total")
        assert hasattr(batch_result, "success_count")
        assert hasattr(batch_result, "failure_count")

        # Verify types
        assert isinstance(batch_result.succeeded, int)
        assert isinstance(batch_result.failed, int)
        assert isinstance(batch_result.total, int)


class TestCreateInStatic:
    """Test suite for tf.create_in() static method."""

    def test_create_in_text_content(self, tmp_path: Path) -> None:
        """Test create_in() for text content."""
        path = tf.create_in("hello world", "test.txt", tmp_path)

        assert path.exists()
        assert path.read_text() == "hello world"

    def test_create_in_dict_content(self, tmp_path: Path) -> None:
        """Test create_in() for dict content (JSON)."""
        path = tf.create_in({"key": "value"}, "config.json", tmp_path)

        assert path.exists()
        content = json.loads(path.read_text())
        assert content == {"key": "value"}

    def test_create_in_yaml_content(self, tmp_path: Path) -> None:
        """Test create_in() for YAML file."""
        path = tf.create_in({"setting": True}, "config.yaml", tmp_path)

        assert path.exists()
        content = yaml.safe_load(path.read_text())
        assert content == {"setting": True}

    def test_create_in_pydantic_model(self, tmp_path: Path) -> None:
        """Test create_in() for Pydantic model content."""

        class UserModel(BaseModel):
            name: str
            age: int

        user = UserModel(name="Alice", age=30)
        path = tf.create_in(user, "user.json", tmp_path)

        assert path.exists()
        content = json.loads(path.read_text())
        assert content == {"name": "Alice", "age": 30}

    def test_create_in_format_detection(self, tmp_path: Path) -> None:
        """Test create_in() format auto-detection from extension."""
        # JSON from .json extension
        path1 = tf.create_in({"key": "value"}, "config.json", tmp_path)
        assert path1.exists()
        assert json.loads(path1.read_text()) == {"key": "value"}

        # YAML from .yaml extension
        path2 = tf.create_in({"key": "value"}, "config.yaml", tmp_path)
        assert path2.exists()
        assert yaml.safe_load(path2.read_text()) == {"key": "value"}

        # CSV from .csv extension
        path3 = tf.create_in([["a", "b"], ["1", "2"]], "data.csv", tmp_path)
        assert path3.exists()
        lines = path3.read_text().strip().split("\n")
        assert len(lines) >= 2

    def test_create_in_with_flextresult(self, tmp_path: Path) -> None:
        """Test create_in() with FlextResult content extraction."""
        result = r[t_core.Types.ConfigurationMapping].ok({"status": "success"})
        path = tf.create_in(result, "result.json", tmp_path)

        assert path.exists()
        content = json.loads(path.read_text())
        assert content == {"status": "success"}

    def test_create_in_custom_format(self, tmp_path: Path) -> None:
        """Test create_in() with explicit format override."""
        path = tf.create_in(b"binary data", "data.dat", tmp_path, fmt="bin")

        assert path.exists()
        assert path.read_bytes() == b"binary data"

    def test_create_in_custom_encoding(self, tmp_path: Path) -> None:
        """Test create_in() with custom encoding."""
        path = tf.create_in("áéíóú", "unicode.txt", tmp_path, enc="utf-16")

        assert path.exists()
        assert path.read_text(encoding="utf-16") == "áéíóú"

    def test_create_in_json_indent(self, tmp_path: Path) -> None:
        """Test create_in() with custom JSON indentation."""
        content_dict = {"key": "value", "nested": {"a": 1}}
        # Type: dict[str, object] needs to be cast to ConfigurationMapping
        # create_in() accepts ConfigurationMapping which is compatible
        content: t_core.Types.ConfigurationMapping = cast(
            "t_core.Types.ConfigurationMapping",
            content_dict,
        )
        path = tf.create_in(content, "config.json", tmp_path, indent=4)

        assert path.exists()
        text = path.read_text()
        # 4-space indent should have more whitespace
        assert "    " in text

    def test_create_in_csv_with_headers(self, tmp_path: Path) -> None:
        """Test create_in() CSV with explicit headers."""
        content = [["1", "2"], ["3", "4"]]
        path = tf.create_in(
            content,
            "data.csv",
            tmp_path,
            headers=["col1", "col2"],
        )

        assert path.exists()
        lines = path.read_text().strip().split("\n")
        assert lines[0] == "col1,col2"
        assert len(lines) == 3  # header + 2 data rows
