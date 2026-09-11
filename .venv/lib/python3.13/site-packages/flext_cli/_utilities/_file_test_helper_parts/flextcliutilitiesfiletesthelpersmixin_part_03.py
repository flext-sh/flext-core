"""Test-oriented file helpers generalized for reuse through ``u.Cli``.

These operations are generic enough to be used by tests, examples, and
maintenance scripts, but were originally duplicated in ``flext-tests``.
They live here so ``flext-tests`` can delegate to ``u.Cli`` instead of
reimplementing them.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import TYPE_CHECKING

from flext_cli import c, p, r
from flext_cli._utilities._file_test_helper_parts.flextcliutilitiesfiletesthelpersmixin_part_04 import (
    FlextCliUtilitiesFileTestHelpersMixin as FlextCliUtilitiesFileTestHelpersMixinPart04,
)
from flext_cli._utilities.files import FlextCliUtilitiesFiles

if TYPE_CHECKING:
    from pathlib import Path


class FlextCliUtilitiesFileTestHelpersMixin:
    """Implementation part for FlextCliUtilitiesFileTestHelpersMixin."""

    @staticmethod
    def files_compare(
        file1: Path,
        file2: Path,
        *,
        mode: str = "content",
        ignore_ws: bool = False,
        ignore_case: bool = False,
    ) -> p.Result[bool]:
        """Compare two files by content, size, hash, or lines.

        Args:
            file1: First file.
            file2: Second file.
            mode: ``content``, ``size``, ``hash``, or ``lines``.
            ignore_ws: Ignore whitespace for content/lines comparison.
            ignore_case: Case-insensitive comparison.

        Returns:
            ``r.ok(True)`` when equal, ``r.ok(False)`` when different, or
            ``r.fail(msg)`` on error.

        """
        if mode == "size":
            return r[bool].ok(file1.stat().st_size == file2.stat().st_size)
        if mode == "hash":
            return r[bool].ok(
                FlextCliUtilitiesFiles.sha256_file(file1)
                == FlextCliUtilitiesFiles.sha256_file(file2)
            )
        if mode == "lines":
            try:
                lines1 = file1.read_text(encoding=c.Cli.ENCODING_DEFAULT).splitlines()
                lines2 = file2.read_text(encoding=c.Cli.ENCODING_DEFAULT).splitlines()
            except OSError as exc:
                msg = f"Compare lines failed: {exc}"
                return r[bool].fail(msg)
            if ignore_ws:
                lines1 = ["".join(line.split()) for line in lines1]
                lines2 = ["".join(line.split()) for line in lines2]
            if ignore_case:
                lines1 = [line.lower() for line in lines1]
                lines2 = [line.lower() for line in lines2]
            return r[bool].ok(lines1 == lines2)
        try:
            text1 = file1.read_text(encoding=c.Cli.ENCODING_DEFAULT)
            text2 = file2.read_text(encoding=c.Cli.ENCODING_DEFAULT)
        except OSError as exc:
            msg = f"Compare content failed: {exc}"
            return r[bool].fail(msg)
        if ignore_ws:
            text1 = "".join(text1.split())
            text2 = "".join(text2.split())
        if ignore_case:
            text1 = text1.lower()
            text2 = text2.lower()
        return r[bool].ok(text1 == text2)

    @staticmethod
    def files_info(
        path: Path, *, compute_hash: bool = False, parse_content: bool = False
    ) -> p.Result[Mapping[str, object]]:
        """Return generic file metadata.

        Args:
            path: File path.
            compute_hash: Include SHA-256 hex digest.
            parse_content: Parse JSON/YAML content and include as ``parsed``.

        Returns:
            ``r.ok(mapping)`` with keys such as ``exists``, ``size``,
            ``is_file``, ``is_dir``, ``format``, ``hash``, ``parsed``.

        """
        try:
            stat = path.stat()
        except OSError as exc:
            msg = f"files_info failed: {exc}"
            return r[Mapping[str, object]].fail(msg)

        info: MutableMapping[str, object] = {
            "exists": path.exists(),
            "path": str(path),
            "size": stat.st_size,
            "is_file": path.is_file(),
            "is_dir": path.is_dir(),
            "format": FlextCliUtilitiesFiles.files_detect_format_from_path(path),
        }
        if compute_hash:
            info["hash"] = FlextCliUtilitiesFiles.sha256_file(path)
        if parse_content and path.is_file():
            parsed_result = (
                FlextCliUtilitiesFileTestHelpersMixinPart04.files_parse_content(
                    path, str(info["format"])
                )
            )
            info["parsed"] = parsed_result
        return r[Mapping[str, object]].ok(info)


__all__: list[str] = ["FlextCliUtilitiesFileTestHelpersMixin"]
