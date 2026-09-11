"""File-assertion helpers for FlextTestsFiles.

Generalized file/directory existence and property checks.
"""

from __future__ import annotations

from pathlib import Path

from flext_infra import u
from flext_tests import c, m
from flext_tests._utilities._files._batch import FlextTestsFilesBatchMixin


class FlextTestsFilesAssertionsMixin(FlextTestsFilesBatchMixin):
    """Generalized file existence and property assertions."""

    @staticmethod
    def assert_exists(
        path: Path,
        msg: str | None = None,
        options: m.Tests.AssertExistsParams | None = None,
        *,
        is_file: bool | None = None,
        is_dir: bool | None = None,
        not_empty: bool | None = None,
        readable: bool | None = None,
        writable: bool | None = None,
    ) -> Path:
        """Generalized file existence assertion - ALL file validations in ONE method.

        Consolidates: assert_exists(), assert_file(), assert_dir(), assert_not_empty()
        into single method with optional parameters.

        Args:
            path: File or directory path to check
            msg: Custom error message
            options: Pre-built assertion parameters; when provided, the
                keyword flags below are ignored
            is_file: Assert is file (True) or not file (False)
            is_dir: Assert is directory (True) or not directory (False)
            not_empty: Assert file/dir is not empty (True) or empty (False)
            readable: Assert is readable (True)
            writable: Assert is writable (True)

        Returns:
            Path if all validations pass

        """
        params = (
            options
            if options is not None
            else m.Tests.AssertExistsParams.model_validate({
                "is_file": is_file,
                "is_dir": is_dir,
                "not_empty": not_empty,
                "readable": readable,
                "writable": writable,
            })
        )
        if not path.exists():
            error_msg = msg or c.Tests.ERROR_FILE_NOT_FOUND.format(path=path)
            raise AssertionError(error_msg)
        try:
            u.Cli.files_assert_exists(
                path,
                is_file=params.is_file,
                is_dir=params.is_dir,
                not_empty=params.not_empty,
                readable=params.readable,
                writable=params.writable,
            )
        except AssertionError as exc:
            raise AssertionError(msg or str(exc)) from exc
        return path


__all__: list[str] = ["FlextTestsFilesAssertionsMixin"]
