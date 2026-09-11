"""Test-oriented file helpers generalized for reuse through ``u.Cli``.

These operations are generic enough to be used by tests, examples, and
maintenance scripts, but were originally duplicated in ``flext-tests``.
They live here so ``flext-tests`` can delegate to ``u.Cli`` instead of
reimplementing them.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


class FlextCliUtilitiesFileTestHelpersMixin:
    """Implementation part for FlextCliUtilitiesFileTestHelpersMixin."""

    @staticmethod
    def files_assert_exists(
        path: Path,
        *,
        is_file: bool | None = None,
        is_dir: bool | None = None,
        not_empty: bool | None = None,
        readable: bool | None = None,
        writable: bool | None = None,
    ) -> Path:
        """Assert file-system properties on ``path``.

        Args:
            path: Path to validate.
            is_file: Assert (or deny) that ``path`` is a regular file.
            is_dir: Assert (or deny) that ``path`` is a directory.
            not_empty: Assert that a file has content or a directory has entries.
            readable: Assert that ``path`` is readable.
            writable: Assert that ``path`` is writable.

        Returns:
            The validated ``path``.

        Raises:
            AssertionError: when any predicate fails.

        """
        if is_file is True and not path.is_file():
            msg = f"Expected file: {path}"
            raise AssertionError(msg)
        if is_file is False and path.is_file():
            msg = f"Expected non-file: {path}"
            raise AssertionError(msg)
        if is_dir is True and not path.is_dir():
            msg = f"Expected directory: {path}"
            raise AssertionError(msg)
        if is_dir is False and path.is_dir():
            msg = f"Expected non-directory: {path}"
            raise AssertionError(msg)
        if not_empty is True:
            if path.is_file() and path.stat().st_size == 0:
                msg = f"Expected non-empty file: {path}"
                raise AssertionError(msg)
            if path.is_dir() and not any(path.iterdir()):
                msg = f"Expected non-empty directory: {path}"
                raise AssertionError(msg)
        if readable is True and not os.access(path, os.R_OK):
            msg = f"Expected readable path: {path}"
            raise AssertionError(msg)
        if writable is True and not os.access(path, os.W_OK):
            msg = f"Expected writable path: {path}"
            raise AssertionError(msg)
        return path


__all__: list[str] = ["FlextCliUtilitiesFileTestHelpersMixin"]
