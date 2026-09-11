"""FLEXT CLI file and format constants.

Owns file-format detection, extension mapping, and generic file metadata
constants reused across CLI and test infrastructure.
"""

from __future__ import annotations

from enum import StrEnum, unique
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from flext_core import t


class FlextCliConstantsFiles:
    """File format and filesystem metadata constants for CLI consumers."""

    @unique
    class FileFormat(StrEnum):
        """Canonical file format labels."""

        AUTO = "auto"
        TEXT = "text"
        BIN = "bin"
        JSON = "json"
        YAML = "yaml"
        TOML = "toml"
        CSV = "csv"
        UNKNOWN = "unknown"

    FILE_FORMAT_AUTO: Final[FileFormat] = FileFormat.AUTO
    FILE_FORMAT_TEXT: Final[FileFormat] = FileFormat.TEXT
    FILE_FORMAT_BIN: Final[FileFormat] = FileFormat.BIN
    FILE_FORMAT_JSON: Final[FileFormat] = FileFormat.JSON
    FILE_FORMAT_YAML: Final[FileFormat] = FileFormat.YAML
    FILE_FORMAT_TOML: Final[FileFormat] = FileFormat.TOML
    FILE_FORMAT_CSV: Final[FileFormat] = FileFormat.CSV
    FILE_FORMAT_UNKNOWN: Final[FileFormat] = FileFormat.UNKNOWN

    KNOWN_FORMATS: Final[frozenset[str]] = frozenset({
        "auto",
        "text",
        "bin",
        "json",
        "yaml",
        "csv",
    })

    EXT_TO_FMT: Final[t.StrMapping] = MappingProxyType({
        ".txt": "text",
        ".log": "text",
        ".md": "text",
        ".rst": "text",
        ".bin": "bin",
        ".dat": "bin",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".csv": "csv",
        ".tsv": "csv",
    })

    DEFAULT_FILENAME: Final[str] = "file"
    DEFAULT_EXTENSION: Final[str] = ".txt"
    DEFAULT_JSON_INDENT: Final[int] = 2
    DEFAULT_CSV_DELIMITER: Final[str] = ","

    SIZE_UNITS: Final[t.StrSequence] = ("B", "KB", "MB", "GB", "TB", "PB")
    SIZE_THRESHOLD: Final[int] = 1024

    @classmethod
    def format_size(cls, size: int) -> str:
        """Return a human-readable byte size string.

        Args:
            size: Size in bytes.

        Returns:
            Human-readable size like ``"1.2 KB"``.

        """
        for unit in cls.SIZE_UNITS:
            if size < cls.SIZE_THRESHOLD:
                return f"{size:.1f} {unit}" if unit != "B" else f"{size} {unit}"
            size //= cls.SIZE_THRESHOLD
        return f"{size:.1f} PB"

    @classmethod
    def format_for_extension(cls, extension: str) -> str:
        """Return the canonical format label for a file extension.

        Args:
            extension: File extension, e.g. ``".json"``.

        Returns:
            Format label or ``"text"`` as the default.

        """
        format_name = cls.EXT_TO_FMT.get(extension.lower())
        return format_name if isinstance(format_name, str) else "text"


__all__: list[str] = ["FlextCliConstantsFiles"]
