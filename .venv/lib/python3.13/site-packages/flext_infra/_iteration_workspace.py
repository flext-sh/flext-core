"""Workspace-scoped Python file iteration mixin.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from flext_core import r
from flext_infra import config, t
from flext_infra._iteration_directory import FlextInfraUtilitiesIterationDirectory

if TYPE_CHECKING:
    from flext_infra import p


class FlextInfraUtilitiesIterationWorkspace:
    """Static helpers for discovering Python files across workspace projects."""

    @classmethod
    def iter_python_files(
        cls, request: p.Infra.SourceScanRequest
    ) -> p.Result[t.SequenceOf[Path]]:
        """Return Python files from the exact production roots in ``request``.

        Args:
            request: Field-only request containing already-resolved project roots.

        Returns:
            Result[t.SequenceOf[Path]] - Success contains sorted unique file paths.
            Failure if: workspace inaccessible, discovery fails, or OSError.

        """
        invalid_root = next(
            (root for root in request.project_roots if not root.is_dir()), None
        )
        if invalid_root is not None:
            return r[t.SequenceOf[Path]].fail(
                f"python file iteration failed: project root is not a directory: {invalid_root}"
            )
        try:
            # NOTE (multi-agent, flext-wkii.17.24 / agent: codex): the scanner
            # consumes one exact request and the validated config singleton;
            # it never discovers projects or enables alternate source trees.
            files = {
                file_path
                for project_root in request.project_roots
                for directory_name in config.Infra.source_scan.roots
                if (directory := project_root / directory_name).is_dir()
                for file_path in FlextInfraUtilitiesIterationDirectory.iter_directory_python_files(
                    directory
                )
            }
            return r[t.SequenceOf[Path]].ok(tuple(sorted(files)))
        except OSError as exc:
            return r[t.SequenceOf[Path]].fail_op("python file iteration", exc)


__all__: list[str] = ["FlextInfraUtilitiesIterationWorkspace"]
