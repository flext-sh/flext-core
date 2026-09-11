"""Generic filesystem helpers shared through ``u.Cli``."""

from __future__ import annotations

import csv
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path

from flext_cli import c, m, p, r, t
from flext_cli._utilities._files_parts.flextcliutilitiesfiles_part_01 import (
    FlextCliUtilitiesFiles as FlextCliUtilitiesFilesPart01,
)
from flext_cli._utilities._files_parts.flextcliutilitiesfiles_part_02 import (
    FlextCliUtilitiesFiles as FlextCliUtilitiesFilesPart02,
)
from flext_cli._utilities.json import FlextCliUtilitiesJson as uj
from flext_cli._utilities.yaml import FlextCliUtilitiesYaml as uy
from flext_core import u


class FlextCliUtilitiesFiles:
    """Implementation part for FlextCliUtilitiesFiles."""

    @staticmethod
    def files_detect_format(file_path: t.Cli.TextPath) -> p.Result[str]:
        """Detect one file format from extension using canonical output enums."""
        suffix = Path(file_path).suffix.lower()
        detected_format = FlextCliUtilitiesFilesPart01.FORMAT_BY_SUFFIX.get(suffix)
        if detected_format is not None:
            return r[str].ok(detected_format)
        if not suffix:
            return r[str].fail("Unable to detect file format without an extension")
        return r[str].fail(f"Unsupported format: {suffix}")

    @staticmethod
    def files_detect_format_from_content(
        content: str
        | bytes
        | m.ConfigMap
        | m.Dict
        | Mapping[str, object]
        | t.SequenceOf[t.StrSequence],
        name: str,
        fmt: str = c.Cli.FILE_FORMAT_AUTO,
    ) -> str:
        """Detect file format from explicit hint, content shape, or extension.

        Uses the test-domain convention (``text`` as the default fallback) so
        callers from ``flext-tests`` keep their existing contract.
        """
        if fmt != c.Cli.FILE_FORMAT_AUTO:
            return fmt
        if isinstance(content, bytes):
            return str(c.Cli.FILE_FORMAT_BIN)
        if isinstance(content, (m.ConfigMap, m.Dict, Mapping)):
            ext = Path(name).suffix.lower()
            return (
                str(c.Cli.FILE_FORMAT_YAML)
                if ext in {".yaml", ".yml"}
                else str(c.Cli.FILE_FORMAT_JSON)
            )
        if isinstance(content, list):
            return str(c.Cli.FILE_FORMAT_CSV)
        return c.Cli.format_for_extension(Path(name).suffix)

    @staticmethod
    def files_detect_format_from_path(
        path: t.Cli.TextPath, fmt: str = c.Cli.FILE_FORMAT_AUTO
    ) -> str:
        """Detect file format from a path extension.

        Uses the test-domain convention (``text`` as the default fallback).
        """
        if fmt != c.Cli.FILE_FORMAT_AUTO:
            return fmt
        return c.Cli.format_for_extension(Path(path).suffix)

    @staticmethod
    def files_load_auto_mapping(file_path: t.Cli.TextPath) -> p.Result[t.JsonMapping]:
        """Load JSON/YAML file and normalize to one mapping payload."""
        path = Path(file_path)
        if path.suffix.lower() == ".json":
            read_result = uj.json_read(path)
        elif path.suffix.lower() in {".yaml", ".yml"}:
            read_result = uy.yaml_safe_load(path)
        else:
            return r[t.JsonMapping].fail(
                f"Unsupported format: {path.suffix or '<none>'}"
            )
        loaded = read_result.map_error(lambda err: err or c.Cli.ERR_AUTO_LOAD_FAILED)
        if loaded.failure:
            return r[t.JsonMapping].fail(loaded.error or c.Cli.ERR_AUTO_LOAD_FAILED)
        payload = loaded.value
        normalized_payload: t.JsonMapping = {
            key: u.normalize_to_json_value(value) for key, value in payload.items()
        }
        return r[t.JsonMapping].ok(normalized_payload)

    @staticmethod
    def csv_loads(text: str, *, delimiter: str = ",") -> p.Result[list[list[str]]]:
        """Parse a CSV-encoded string into a list of rows."""
        try:
            rows = list(csv.reader(text.splitlines(), delimiter=delimiter))
        except csv.Error as exc:
            return r[list[list[str]]].fail(f"csv_loads: {exc}")
        return r[list[list[str]]].ok(rows)

    @staticmethod
    def files_copy_directory(
        source_path: t.Cli.TextPath,
        dest_path: t.Cli.TextPath,
        *,
        dirs_exist_ok: bool = False,
    ) -> p.Result[Path]:
        """Recursively copy *source_path* to *dest_path*, returning the destination."""
        destination = Path(dest_path)

        def _copy() -> Path:
            return Path(
                shutil.copytree(
                    Path(source_path), destination, dirs_exist_ok=dirs_exist_ok
                )
            )

        return FlextCliUtilitiesFilesPart02.files_execute(
            _copy, "copy_directory: {error}"
        )

    @staticmethod
    def files_create_temporary_directory(
        *,
        prefix: str = "flext-cli-",
        suffix: str = "",
        parent_path: t.Cli.TextPath | None = None,
    ) -> p.Result[Path]:
        """Create one temporary directory and return its path for caller cleanup."""

        def _create() -> Path:
            return Path(tempfile.mkdtemp(prefix=prefix, suffix=suffix, dir=parent_path))

        return FlextCliUtilitiesFilesPart02.files_execute(
            _create, "create_temporary_directory: {error}"
        )

    @staticmethod
    def files_remove_directory(directory_path: t.Cli.TextPath) -> p.Result[bool]:
        """Remove one directory tree while rejecting non-directory paths."""
        path = Path(directory_path)
        if not path.exists() and not path.is_symlink():
            return r[bool].ok(True)
        if not path.is_dir() or path.is_symlink():
            return r[bool].fail(f"remove_directory: not a directory: {path}")

        def _remove() -> bool:
            shutil.rmtree(path)
            return True

        return FlextCliUtilitiesFilesPart02.files_execute(
            _remove, "remove_directory: {error}"
        )


__all__: list[str] = ["FlextCliUtilitiesFiles"]
