"""Generic filesystem helpers shared through ``u.Cli``."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path
from types import MappingProxyType
from typing import ClassVar

import flext_core
from flext_cli import c, p, r, t
from flext_cli._utilities._files_parts.flextcliutilitiesfiles_part_02 import (
    FlextCliUtilitiesFiles as FlextCliUtilitiesFilesPart02,
)
from flext_cli._utilities.yaml import FlextCliUtilitiesYaml as uy


class FlextCliUtilitiesFiles:
    """Implementation part for FlextCliUtilitiesFiles."""

    FORMAT_BY_SUFFIX: ClassVar[t.MappingKV[str, str]] = MappingProxyType({
        ".json": c.Cli.OutputFormats.JSON,
        ".yaml": c.Cli.OutputFormats.YAML,
        ".yml": c.Cli.OutputFormats.YAML,
        ".csv": c.Cli.OutputFormats.CSV,
        ".txt": c.Cli.OutputFormats.TEXT,
        ".log": c.Cli.OutputFormats.TEXT,
    })

    @staticmethod
    def files_delete(file_path: t.Cli.TextPath) -> p.Result[bool]:
        """Delete one file-system path using canonical error handling."""
        path = Path(file_path)

        def _delete() -> bool:
            if not path.exists() and not path.is_symlink():
                return True
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()
            return True

        return FlextCliUtilitiesFilesPart02.files_execute_bool(
            _delete, c.Cli.ERR_FILE_DELETION_FAILED
        )

    @staticmethod
    def files_read_text(file_path: t.Cli.TextPath) -> p.Result[str]:
        """Read one UTF-8 text file."""
        return FlextCliUtilitiesFilesPart02.files_execute(
            lambda: Path(file_path).read_text(encoding=c.Cli.ENCODING_DEFAULT),
            c.Cli.ERR_TEXT_READ_FAILED,
        )

    @staticmethod
    def files_write_text(file_path: t.Cli.TextPath, content: str) -> p.Result[bool]:
        """Write one UTF-8 text file."""

        def _write() -> bool:
            Path(file_path).write_text(content, encoding=c.Cli.ENCODING_DEFAULT)
            return True

        return FlextCliUtilitiesFilesPart02.files_execute(
            _write, c.Cli.ERR_TEXT_WRITE_FAILED
        )

    @staticmethod
    def files_read_json(file_path: t.Cli.TextPath) -> p.Result[t.JsonValue]:
        """Read one JSON file and validate to canonical JSON value."""

        def _load() -> t.JsonValue:
            raw = Path(file_path).read_text(encoding=c.Cli.ENCODING_DEFAULT)
            return t.Cli.JSON_VALUE_ADAPTER.validate_json(raw)

        return FlextCliUtilitiesFilesPart02.files_execute(
            _load, c.Cli.ERR_JSON_LOAD_FAILED
        )

    @staticmethod
    def files_read_json_model[M: t.Cli.ModelLike](
        file_path: t.Cli.TextPath, model_type: t.ModelClass[M]
    ) -> p.Result[M]:
        """Read one JSON file directly into one Pydantic model."""
        # NOTE (multi-agent): Model classes use the canonical t.ModelClass alias.

        def _load() -> M:
            raw = Path(file_path).read_bytes()
            loaded: M = model_type.model_validate_json(raw, strict=False)
            return loaded

        return FlextCliUtilitiesFilesPart02.files_execute(
            _load, c.Cli.ERR_JSON_LOAD_FAILED
        )

    @staticmethod
    def files_read_first_json_model[M: t.Cli.ModelLike](
        file_path: t.Cli.TextPath, model_type: t.ModelClass[M]
    ) -> p.Result[M]:
        """Stream and validate the first non-empty JSON line into one model."""

        def _load() -> M:
            with Path(file_path).open(
                mode="r", encoding=c.Cli.ENCODING_DEFAULT
            ) as handle:
                for line in handle:
                    if line.strip():
                        loaded: M = model_type.model_validate_json(line, strict=False)
                        return loaded
            msg = f"JSON-lines file has no records: {file_path}"
            raise ValueError(msg)

        return FlextCliUtilitiesFilesPart02.files_execute(
            _load, c.Cli.ERR_JSON_LOAD_FAILED
        )

    @staticmethod
    def files_read_json_lines_model[M: t.Cli.ModelLike](
        file_path: t.Cli.TextPath, model_type: t.ModelClass[M]
    ) -> p.Result[tuple[M, ...]]:
        """Stream every non-empty JSON line and validate each into one model."""

        def _load() -> tuple[M, ...]:
            with Path(file_path).open(
                mode="r", encoding=c.Cli.ENCODING_DEFAULT
            ) as handle:
                return tuple(
                    model_type.model_validate_json(line, strict=False)
                    for line in handle
                    if line.strip()
                )

        return FlextCliUtilitiesFilesPart02.files_execute(
            _load, c.Cli.ERR_JSON_LOAD_FAILED
        )

    @staticmethod
    def files_read_yaml(file_path: t.Cli.TextPath) -> p.Result[t.JsonValue]:
        """Read one YAML file and validate to canonical JSON value."""
        return uy.yaml_safe_load(Path(file_path)).map(
            t.Cli.JSON_VALUE_ADAPTER.validate_python
        )

    @staticmethod
    def files_read_yaml_model[M: t.Cli.ModelLike](
        file_path: t.Cli.TextPath, model_type: t.ModelClass[M]
    ) -> p.Result[M]:
        """Read YAML directly into one caller-supplied validated model."""
        return uy.yaml_safe_load(Path(file_path)).map(model_type.model_validate)

    @staticmethod
    def files_read_yaml_model_chain[M: t.Cli.ModelLike](
        file_paths: t.SequenceOf[t.Cli.TextPath], model_type: t.ModelClass[M]
    ) -> p.Result[M]:
        """Merge ordered YAML sources and validate the final payload once."""
        sources = tuple(Path(file_path) for file_path in file_paths)
        if not sources:
            return r[M].fail(c.Cli.ERR_FILE_PATH_EMPTY)
        first = uy.yaml_safe_load(sources[0])
        if first.failure:
            return first.map(model_type.model_validate)
        merged: t.JsonMapping = first.value
        for source in sources[1:]:
            loaded = uy.yaml_safe_load(source)
            if loaded.failure:
                return loaded.map(model_type.model_validate)
            merged = flext_core.u.config_merge(merged, loaded.value)
        return r[t.JsonMapping].ok(merged).map(model_type.model_validate)

    @staticmethod
    def files_write_csv(
        file_path: t.Cli.TextPath, rows: t.SequenceOf[t.StrSequence]
    ) -> p.Result[bool]:
        """Write one CSV file from row sequence."""

        def _write() -> bool:
            with Path(file_path).open(
                mode="w", encoding=c.Cli.ENCODING_DEFAULT, newline=""
            ) as handle:
                writer = csv.writer(handle)
                for row in rows:
                    writer.writerow(list(row))
            return True

        return FlextCliUtilitiesFilesPart02.files_execute(
            _write, c.Cli.ERR_CSV_WRITE_FAILED
        )


__all__: list[str] = ["FlextCliUtilitiesFiles"]
