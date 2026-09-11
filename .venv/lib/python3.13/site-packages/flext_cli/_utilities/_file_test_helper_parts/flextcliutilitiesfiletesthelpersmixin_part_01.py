"""Test-oriented file helpers generalized for reuse through ``u.Cli``.

These operations are generic enough to be used by tests, examples, and
maintenance scripts, but were originally duplicated in ``flext-tests``.
They live here so ``flext-tests`` can delegate to ``u.Cli`` instead of
reimplementing them.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import cast

from flext_cli import c, p, r, t
from flext_cli._utilities.files import FlextCliUtilitiesFiles
from flext_cli._utilities.json import FlextCliUtilitiesJson as uj
from flext_cli._utilities.yaml import FlextCliUtilitiesYaml as uy


class FlextCliUtilitiesFileTestHelpersMixin:
    """Implementation part for FlextCliUtilitiesFileTestHelpersMixin."""

    @classmethod
    @contextmanager
    def files_context(
        cls,
        content: Mapping[str, str | bytes | t.JsonValue | t.SequenceOf[t.StrSequence]],
        *,
        directory: Path | None = None,
        ext: str | None = None,
        cleanup: bool = True,
    ) -> Generator[Mapping[str, Path]]:
        """Create a temporary bundle of files and yield the path mapping.

        Args:
            content: Mapping from file name to raw content.
            directory: Optional base directory; uses a temp directory if omitted.
            ext: Optional extension appended to every file name.
            cleanup: Remove created files/directories on exit when True.

        Yields:
            Mapping[str, Path] with resolved file paths.

        """
        base_dir = directory or Path(tempfile.mkdtemp())
        created: dict[str, Path] = {}
        try:
            for name, raw in content.items():
                file_name = f"{name}{ext or ''}"
                file_path = base_dir / file_name
                FlextCliUtilitiesFiles.ensure_dir(file_path.parent)
                if isinstance(raw, bytes):
                    file_path.write_bytes(raw)
                elif isinstance(raw, str):
                    file_path.write_text(raw, encoding=c.Cli.ENCODING_DEFAULT)
                elif isinstance(raw, Mapping):
                    fmt = (
                        c.Cli.FILE_FORMAT_YAML
                        if file_path.suffix in {".yaml", ".yml"}
                        else c.Cli.FILE_FORMAT_JSON
                    )
                    cls._files_write_structured(file_path, raw, fmt)
                elif isinstance(raw, list):
                    FlextCliUtilitiesFiles.files_write_csv(
                        file_path, cast("t.SequenceOf[t.StrSequence]", raw)
                    )
                else:
                    file_path.write_text(str(raw), encoding=c.Cli.ENCODING_DEFAULT)
                created[name] = file_path
            yield created
        finally:
            if cleanup:
                for path in created.values():
                    if path.exists() or path.is_symlink():
                        if path.is_dir():
                            shutil.rmtree(path)
                        else:
                            path.unlink()
                if directory is None and base_dir.exists():
                    shutil.rmtree(base_dir)

    @staticmethod
    def _files_write_structured(
        path: Path, data: t.JsonValue, fmt: str
    ) -> p.Result[bool]:
        """Write a structured payload as JSON or YAML."""
        validated = t.Cli.JSON_VALUE_ADAPTER.validate_python(data)
        if fmt == c.Cli.FILE_FORMAT_YAML:
            dumped = uy.yaml_dump_str(validated)
            return FlextCliUtilitiesFiles.files_write_text(path, dumped)
        dumped_result = uj.json_dumps(validated)
        if dumped_result.failure:
            return r[bool].fail(dumped_result.error or "json_dumps failed")
        return FlextCliUtilitiesFiles.files_write_text(path, dumped_result.unwrap())


__all__: list[str] = ["FlextCliUtilitiesFileTestHelpersMixin"]
