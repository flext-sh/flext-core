"""FLEXT CLI file operations utilities."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from flext_cli import c, m, p, r, s, t, u
from flext_cli.services._file_tools_atomic import FlextCliFileToolsAtomicMixin


class FlextCliFileTools(FlextCliFileToolsAtomicMixin, s):
    """File operations with r."""

    @staticmethod
    def ensure_dir(file_path: t.Cli.TextPath) -> p.Result[Path]:
        """Create a directory tree when missing and return the path."""
        return u.Cli.ensure_dir(Path(file_path))

    @staticmethod
    def read_json_file(file_path: t.Cli.TextPath) -> p.Result[t.JsonValue]:
        """Read one JSON file into a validated JSON-compatible value."""
        return u.Cli.files_read_json(Path(file_path))

    @staticmethod
    def read_text_file(file_path: t.Cli.TextPath) -> p.Result[str]:
        """Read a UTF-8 text file via the public CLI file surface."""
        return u.Cli.files_read_text(Path(file_path))

    @staticmethod
    def read_json_model[M: t.Cli.ModelLike](
        file_path: t.Cli.TextPath, model_type: t.ModelClass[M]
    ) -> p.Result[M]:
        """Read JSON into the canonical structural model-class contract."""
        return u.Cli.files_read_json_model(Path(file_path), model_type)

    @staticmethod
    def read_yaml_file(file_path: t.Cli.TextPath) -> p.Result[t.JsonValue]:
        """Read one YAML file into a validated JSON-compatible value."""
        normalized_path = u.Cli.normalize_optional_text(file_path)
        if normalized_path is None:
            return r[t.JsonValue].fail(c.Cli.ERR_FILE_PATH_EMPTY)
        return u.Cli.files_read_yaml(Path(normalized_path))

    @staticmethod
    def read_yaml_model[M: t.Cli.ModelLike](
        file_path: t.Cli.TextPath, model_type: t.ModelClass[M]
    ) -> p.Result[M]:
        """Read YAML and validate it once into the requested model type."""
        return u.Cli.files_read_yaml_model(Path(file_path), model_type)

    @staticmethod
    def read_yaml_model_chain[M: t.Cli.ModelLike](
        file_paths: Sequence[t.Cli.TextPath], model_type: t.ModelClass[M]
    ) -> p.Result[M]:
        """Merge ordered YAML sources and validate the final payload once."""
        return u.Cli.files_read_yaml_model_chain(file_paths, model_type)

    @staticmethod
    def write_json_file(
        file_path: t.Cli.TextPath,
        data: t.Cli.JsonWriteData,
        options: m.Cli.JsonWriteOptions | None = None,
    ) -> p.Result[bool]:
        """Write JSON data using the canonical serialization options."""
        return u.Cli.json_write(Path(file_path), data, options=options)

    @staticmethod
    def write_yaml_file(
        file_path: t.Cli.TextPath, data: t.Cli.JsonWriteData
    ) -> p.Result[bool]:
        """Write JSON-compatible data as YAML."""
        return u.Cli.yaml_dump(Path(file_path), data)

    @staticmethod
    def write_csv_file(
        file_path: t.Cli.TextPath, rows: t.SequenceOf[t.StrSequence]
    ) -> p.Result[bool]:
        """Write rows to a CSV file."""
        return u.Cli.files_write_csv(Path(file_path), rows)

    @staticmethod
    def read_csv_file_with_headers(
        file_path: t.Cli.TextPath,
    ) -> p.Result[t.SequenceOf[t.StrMapping]]:
        """Read CSV rows as string mappings keyed by the header row."""
        return u.Cli.files_read_csv_with_headers(Path(file_path))

    @staticmethod
    def read_binary_file(file_path: t.Cli.TextPath) -> p.Result[bytes]:
        """Read a file as bytes."""
        return u.Cli.files_read_binary(Path(file_path))

    @staticmethod
    def write_binary_file(file_path: t.Cli.TextPath, data: bytes) -> p.Result[bool]:
        """Write bytes to a file."""
        return u.Cli.files_write_binary(Path(file_path), data)

    @staticmethod
    def copy_file(
        source_path: t.Cli.TextPath, destination_path: t.Cli.TextPath
    ) -> p.Result[bool]:
        """Copy one file to another path."""
        return u.Cli.files_copy(Path(source_path), Path(destination_path))

    @staticmethod
    def detect_file_format(file_path: t.Cli.TextPath) -> p.Result[str]:
        """Detect the supported serialization format for a file."""
        return u.Cli.files_detect_format(Path(file_path))

    @staticmethod
    def delete_path(file_path: t.Cli.TextPath) -> p.Result[bool]:
        """Delete a file or directory via the public CLI file surface."""
        return u.Cli.files_delete(Path(file_path))

    @staticmethod
    def list_directory_names(file_path: t.Cli.TextPath) -> p.Result[Sequence[str]]:
        """Return sorted child directory names for one path."""
        return u.Cli.files_list_directory_names(Path(file_path))

    @staticmethod
    def load_file_auto_dict(file_path: t.Cli.TextPath) -> p.Result[t.JsonMapping]:
        """Load a supported file format as a JSON-compatible mapping."""
        return u.Cli.files_load_auto_mapping(Path(file_path))


__all__: t.MutableSequenceOf[str] = ["FlextCliFileTools"]
