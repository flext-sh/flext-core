"""Unit tests for flext_tests.files module.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from pathlib import Path

from flext_tests.files import FlextTestsFileManager

# Access nested class
FileInfo = FlextTestsFileManager.FileInfo


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


class TestFlextTestsFileManager:
    """Test suite for FlextTestsFileManager class."""

    def test_init_without_base_dir(self) -> None:
        """Test initialization without base directory."""
        manager = FlextTestsFileManager()

        assert manager.base_dir is None
        assert manager.created_files == []
        assert manager.created_dirs == []

    def test_init_with_base_dir(self, tmp_path: Path) -> None:
        """Test initialization with base directory."""
        manager = FlextTestsFileManager(base_dir=tmp_path)

        assert manager.base_dir == tmp_path
        assert manager.created_files == []
        assert manager.created_dirs == []

    def test_temporary_directory(self) -> None:
        """Test temporary_directory context manager."""
        manager = FlextTestsFileManager()

        with manager.temporary_directory() as temp_dir:
            assert isinstance(temp_dir, Path)
            assert temp_dir.exists()
            assert temp_dir.is_dir()

        # Directory should be cleaned up after context exit
        assert not temp_dir.exists()

    def test_create_text_file_default(self, tmp_path: Path) -> None:
        """Test creating text file with default parameters."""
        manager = FlextTestsFileManager(base_dir=tmp_path)
        content = "test content"

        file_path = manager.create_text_file(content)

        assert file_path.exists()
        assert file_path.read_text() == content
        assert file_path.name == "test.txt"
        assert file_path in manager.created_files

    def test_create_text_file_custom(self, tmp_path: Path) -> None:
        """Test creating text file with custom parameters."""
        manager = FlextTestsFileManager(base_dir=tmp_path)
        content = "custom content"
        filename = "custom.txt"
        custom_dir = tmp_path / "subdir"

        file_path = manager.create_text_file(content, filename, custom_dir)

        assert file_path.exists()
        assert file_path.read_text() == content
        assert file_path.name == filename
        assert file_path.parent == custom_dir
        assert file_path in manager.created_files

    def test_create_text_file_custom_encoding(self, tmp_path: Path) -> None:
        """Test creating text file with custom encoding."""
        manager = FlextTestsFileManager(base_dir=tmp_path)
        content = "test content"
        encoding = "utf-16"

        file_path = manager.create_text_file(content, encoding=encoding)

        assert file_path.exists()
        assert file_path.read_text(encoding=encoding) == content

    def test_create_binary_file_default(self, tmp_path: Path) -> None:
        """Test creating binary file with default parameters."""
        manager = FlextTestsFileManager(base_dir=tmp_path)
        content = b"binary content"

        file_path = manager.create_binary_file(content)

        assert file_path.exists()
        assert file_path.read_bytes() == content
        assert file_path.name == "binary_data.bin"
        assert file_path in manager.created_files

    def test_create_binary_file_custom(self, tmp_path: Path) -> None:
        """Test creating binary file with custom parameters."""
        manager = FlextTestsFileManager(base_dir=tmp_path)
        content = b"custom binary"
        filename = "custom.bin"
        custom_dir = tmp_path / "subdir"

        file_path = manager.create_binary_file(content, filename, custom_dir)

        assert file_path.exists()
        assert file_path.read_bytes() == content
        assert file_path.name == filename
        assert file_path.parent == custom_dir

    def test_create_empty_file(self, tmp_path: Path) -> None:
        """Test creating empty file."""
        manager = FlextTestsFileManager(base_dir=tmp_path)

        file_path = manager.create_empty_file()

        assert file_path.exists()
        assert file_path.read_text() == ""
        assert file_path.name == "empty.txt"

    def test_create_empty_file_custom(self, tmp_path: Path) -> None:
        """Test creating empty file with custom name."""
        manager = FlextTestsFileManager(base_dir=tmp_path)
        filename = "custom_empty.txt"

        file_path = manager.create_empty_file(filename)

        assert file_path.exists()
        assert file_path.read_text() == ""
        assert file_path.name == filename

    def test_create_file_set(self, tmp_path: Path) -> None:
        """Test creating multiple files from dictionary."""
        manager = FlextTestsFileManager(base_dir=tmp_path)
        files = {
            "file1": "content1",
            "file2": "content2",
            "file3.txt": "content3",
        }

        created = manager.create_file_set(files)

        assert len(created) == 3
        assert created["file1"].read_text() == "content1"
        assert created["file2"].read_text() == "content2"
        assert created["file3.txt"].read_text() == "content3"
        assert created["file1"].name == "file1.txt"
        assert created["file2"].name == "file2.txt"
        assert created["file3.txt"].name == "file3.txt"

    def test_create_file_set_custom_extension(self, tmp_path: Path) -> None:
        """Test creating file set with custom extension."""
        manager = FlextTestsFileManager(base_dir=tmp_path)
        files = {"file1": "content1"}
        extension = ".md"

        created = manager.create_file_set(files, extension=extension)

        assert created["file1"].name == "file1.md"

    def test_get_file_info_not_exists(self, tmp_path: Path) -> None:
        """Test getting file info for non-existent file."""
        manager = FlextTestsFileManager()
        non_existent = tmp_path / "non_existent.txt"

        info = manager.get_file_info(non_existent)

        assert isinstance(info, FileInfo)
        assert info.exists is False

    def test_get_file_info_exists(self, tmp_path: Path) -> None:
        """Test getting file info for existing file."""
        manager = FlextTestsFileManager(base_dir=tmp_path)
        content = "line1\nline2\nline3"
        file_path = manager.create_text_file(content, "test.txt")

        info = manager.get_file_info(file_path)

        assert isinstance(info, FileInfo)
        assert info.exists is True
        assert info.size > 0
        assert info.lines == 3
        assert info.encoding == "utf-8"
        assert info.is_empty is False
        assert info.first_line == "line1"

    def test_get_file_info_empty_file(self, tmp_path: Path) -> None:
        """Test getting file info for empty file."""
        manager = FlextTestsFileManager(base_dir=tmp_path)
        file_path = manager.create_empty_file("empty.txt")

        info = manager.get_file_info(file_path)

        assert isinstance(info, FileInfo)
        assert info.exists is True
        assert info.size == 0
        assert info.is_empty is True
        assert info.first_line == ""

    def test_get_file_info_multiline(self, tmp_path: Path) -> None:
        """Test getting file info for multiline file."""
        manager = FlextTestsFileManager(base_dir=tmp_path)
        content = "first line\nsecond line\nthird line"
        file_path = manager.create_text_file(content, "multiline.txt")

        info = manager.get_file_info(file_path)

        assert info.lines == 3
        assert info.first_line == "first line"

    def test_cleanup_files(self, tmp_path: Path) -> None:
        """Test cleaning up created files."""
        manager = FlextTestsFileManager(base_dir=tmp_path)
        file1 = manager.create_text_file("content1", "file1.txt")
        file2 = manager.create_text_file("content2", "file2.txt")

        assert file1.exists()
        assert file2.exists()

        manager.cleanup()

        assert not file1.exists()
        assert not file2.exists()
        assert len(manager.created_files) == 0

    def test_cleanup_directories(self) -> None:
        """Test cleaning up created directories."""
        manager = FlextTestsFileManager()
        # Create a file which will create a temp directory
        file_path = manager.create_text_file("content", "test.txt")
        temp_dir = file_path.parent

        assert temp_dir.exists()
        assert temp_dir in manager.created_dirs

        manager.cleanup()

        assert not temp_dir.exists()
        assert len(manager.created_dirs) == 0

    def test_cleanup_nonexistent_files(self, tmp_path: Path) -> None:
        """Test cleanup handles non-existent files gracefully."""
        manager = FlextTestsFileManager(base_dir=tmp_path)
        file_path = manager.create_text_file("content", "test.txt")
        file_path.unlink()  # Delete file manually

        # Cleanup should not raise error
        manager.cleanup()

        assert len(manager.created_files) == 0

    def test_context_manager(self, tmp_path: Path) -> None:
        """Test context manager usage."""
        with FlextTestsFileManager(base_dir=tmp_path) as manager:
            file_path = manager.create_text_file("content", "test.txt")
            assert file_path.exists()

        # File should be cleaned up after context exit
        assert not file_path.exists()

    def test_resolve_directory_with_directory(self, tmp_path: Path) -> None:
        """Test directory resolution with provided directory."""
        manager = FlextTestsFileManager(base_dir=tmp_path)
        custom_dir = tmp_path / "custom"

        resolved = manager._resolve_directory(custom_dir)

        assert resolved == custom_dir

    def test_resolve_directory_with_base_dir(self, tmp_path: Path) -> None:
        """Test directory resolution with base_dir."""
        manager = FlextTestsFileManager(base_dir=tmp_path)

        resolved = manager._resolve_directory(None)

        assert resolved == tmp_path

    def test_resolve_directory_creates_temp(self) -> None:
        """Test directory resolution creates temporary directory."""
        manager = FlextTestsFileManager()

        resolved = manager._resolve_directory(None)

        assert resolved.exists()
        assert resolved.is_dir()
        assert resolved in manager.created_dirs

    def test_temporary_files_classmethod(self) -> None:
        """Test temporary_files classmethod context manager."""
        files = {
            "file1": "content1",
            "file2": "content2",
        }

        with FlextTestsFileManager.temporary_files(files) as created:
            assert len(created) == 2
            assert created["file1"].exists()
            assert created["file2"].exists()
            assert created["file1"].read_text() == "content1"
            assert created["file2"].read_text() == "content2"

        # Files should be cleaned up after context exit
        assert not created["file1"].exists()
        assert not created["file2"].exists()

    def test_temporary_files_custom_extension(self) -> None:
        """Test temporary_files with custom extension."""
        files = {"file1": "content1"}

        with FlextTestsFileManager.temporary_files(files, extension=".md") as created:
            assert created["file1"].name == "file1.md"

    def test_create_file_set_nested_directory(self, tmp_path: Path) -> None:
        """Test creating files in nested directory."""
        manager = FlextTestsFileManager(base_dir=tmp_path)
        nested_dir = tmp_path / "nested" / "subdir"
        files = {"file1": "content1"}

        created = manager.create_file_set(files, directory=nested_dir)

        assert created["file1"].parent == nested_dir
        assert nested_dir.exists()

    def test_create_text_file_nested_directory(self, tmp_path: Path) -> None:
        """Test creating text file in nested directory."""
        manager = FlextTestsFileManager(base_dir=tmp_path)
        nested_dir = tmp_path / "nested" / "subdir"

        file_path = manager.create_text_file("content", "test.txt", nested_dir)

        assert file_path.parent == nested_dir
        assert nested_dir.exists()

    def test_multiple_cleanup_calls(self, tmp_path: Path) -> None:
        """Test multiple cleanup calls are safe."""
        manager = FlextTestsFileManager(base_dir=tmp_path)
        manager.create_text_file("content", "test.txt")

        manager.cleanup()
        # Second cleanup should not raise error
        manager.cleanup()

        assert len(manager.created_files) == 0
