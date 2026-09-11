"""Generic filesystem helpers shared through ``u.Cli``."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

from flext_cli import c, p, r, t
from flext_cli._utilities._atomic_file import write_atomic_bytes
from flext_cli._utilities._atomic_file_delete import remove_guarded_file


class FlextCliUtilitiesFiles:
    """Implementation part for FlextCliUtilitiesFiles."""

    @staticmethod
    def files_read_csv_with_headers(
        file_path: t.Cli.TextPath,
    ) -> p.Result[t.SequenceOf[t.StrMapping]]:
        """Read one CSV file into mapping rows using header row."""

        def _load() -> t.SequenceOf[t.StrMapping]:
            with Path(file_path).open(
                encoding=c.Cli.ENCODING_DEFAULT, newline=""
            ) as handle:
                return [dict(row) for row in csv.DictReader(handle)]

        return FlextCliUtilitiesFiles.files_execute(_load, c.Cli.ERR_CSV_READ_FAILED)

    @staticmethod
    def files_read_binary(file_path: t.Cli.TextPath) -> p.Result[bytes]:
        """Read one binary file."""
        return FlextCliUtilitiesFiles.files_execute(
            lambda: Path(file_path).read_bytes(), c.Cli.ERR_BINARY_READ_FAILED
        )

    @staticmethod
    def files_write_binary(file_path: t.Cli.TextPath, data: bytes) -> p.Result[bool]:
        """Write one binary file atomically in its destination directory."""
        path = Path(file_path).absolute()
        ensure_result = FlextCliUtilitiesFiles.ensure_dir(path.parent)
        if ensure_result.failure:
            return r[bool].fail(
                ensure_result.error or c.Cli.ERR_ENSURE_DIR_GENERIC_FAILED
            )
        try:
            write_atomic_bytes(path, data)
        except OSError as exc:
            return r[bool].fail(c.Cli.ERR_BINARY_WRITE_FAILED.format(error=exc))
        return r[bool].ok(True)

    @staticmethod
    def atomic_write_text_file(
        file_path: t.Cli.TextPath, content: str
    ) -> p.Result[bool]:
        """Write a text file atomically via the shared byte primitive."""
        path = Path(file_path).absolute()
        ensure_result = FlextCliUtilitiesFiles.ensure_dir(path.parent)
        if ensure_result.failure:
            return r[bool].fail(
                ensure_result.error or c.Cli.ERR_ENSURE_DIR_GENERIC_FAILED
            )
        try:
            write_atomic_bytes(path, content.encode(c.Cli.ENCODING_DEFAULT))
        except OSError as exc:
            return r[bool].fail(
                c.Cli.ERR_ATOMIC_WRITE_TEXT_FILE_FAILED.format(error=exc)
            )
        return r[bool].ok(True)

    @staticmethod
    def atomic_write_text_file_guarded(
        file_path: t.Cli.TextPath, content: str, *, expected_bytes: bytes | None
    ) -> p.Result[bool]:
        """Publish under a caller-held lock after an exact raw-byte precondition.

        Cooperative writers must hold one exclusive lock from planning through
        this call.  This portable operation is not compare-and-swap against actors
        that ignore that lock and does not promise power-loss durability.  ``None``
        requires absence; bytes require an existing uniquely owned regular,
        non-reparse destination with exactly that content.  The immediate parent
        must already exist as a real directory and is never created by this call.
        """
        path = Path(file_path).absolute()
        try:
            write_atomic_bytes(
                path,
                content.encode(c.Cli.ENCODING_DEFAULT),
                expected_bytes=expected_bytes,
            )
        except OSError as exc:
            return r[bool].fail(
                c.Cli.ERR_ATOMIC_WRITE_TEXT_FILE_FAILED.format(error=exc)
            )
        return r[bool].ok(True)

    @staticmethod
    def atomic_write_binary_file_guarded(
        file_path: t.Cli.TextPath,
        data: bytes,
        *,
        expected_bytes: bytes | None,
        expected_mode: int | None,
        permission_mode: int,
    ) -> p.Result[bool]:
        """Publish exact bytes and mode under a caller-held exclusive lock.

        Cooperative writers must hold the same lock through planning and this
        call.  This is not CAS against an actor that ignores that lock and does
        not promise parent-directory power-loss durability.  The immediate real
        parent must exist and is never created here.
        """
        path = Path(file_path).absolute()
        try:
            write_atomic_bytes(
                path,
                data,
                expected_bytes=expected_bytes,
                expected_mode=expected_mode,
                permission_mode=permission_mode,
            )
        except OSError as exc:
            return r[bool].fail(c.Cli.ERR_BINARY_WRITE_FAILED.format(error=exc))
        return r[bool].ok(True)

    @staticmethod
    def atomic_delete_binary_file_guarded(
        file_path: t.Cli.TextPath, *, expected_bytes: bytes, expected_mode: int
    ) -> p.Result[bool]:
        """Delete the exact byte-and-mode version under the caller's lock.

        The portable check-then-unlink operation depends on every writer sharing
        that lock; it is not CAS against an actor that ignores it.  The immediate
        real parent must exist.  Directory power-loss durability is not promised.
        """
        path = Path(file_path).absolute()
        try:
            remove_guarded_file(
                path, expected_bytes=expected_bytes, expected_mode=expected_mode
            )
        except OSError as exc:
            return r[bool].fail(c.Cli.ERR_FILE_DELETION_FAILED.format(error=exc))
        return r[bool].ok(True)

    @staticmethod
    def files_copy(
        source_path: t.Cli.TextPath, destination_path: t.Cli.TextPath
    ) -> p.Result[bool]:
        """Copy one file preserving metadata."""

        def _copy() -> bool:
            shutil.copy2(source_path, destination_path)
            return True

        return FlextCliUtilitiesFiles.files_execute(_copy, c.Cli.ERR_FILE_COPY_FAILED)

    @staticmethod
    def files_execute[T](
        operation_func: t.Cli.NullaryOperation[T],
        error_template: str,
        **format_kwargs: t.Scalar,
    ) -> p.Result[T]:
        """Execute one operation and map common runtime errors to ``r``."""
        try:
            return r[T].ok(operation_func())
        except c.EXC_BROAD_RUNTIME_OS as exc:
            return r[T].fail(error_template.format(error=exc, **format_kwargs))

    @staticmethod
    def files_execute_bool[T](
        operation_func: t.Cli.NullaryOperation[T],
        error_template: str,
        **format_kwargs: t.Scalar,
    ) -> p.Result[bool]:
        """Execute one operation that should return a success boolean."""

        def _run() -> bool:
            _ = operation_func()
            return True

        return FlextCliUtilitiesFiles.files_execute(
            _run, error_template, **format_kwargs
        )

    @staticmethod
    def ensure_dir(path: t.Cli.TextPath) -> p.Result[Path]:
        """Create a directory tree when missing and return the resolved path."""
        target = Path(path)
        try:
            target.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return r[Path].fail(c.Cli.ERR_ENSURE_DIR_FAILED.format(error=exc))
        return r[Path].ok(target)


__all__: list[str] = ["FlextCliUtilitiesFiles"]
